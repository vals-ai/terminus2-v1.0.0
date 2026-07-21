from __future__ import annotations

import hashlib
import json
import logging
import re
import shlex
import subprocess
from pathlib import Path, PurePosixPath

MAX_MODEL_PATCH_BYTES = 10 * 1024 * 1024
_BINARY_MARKERS = (b"GIT binary patch", b"Binary files ")
_DIRECT_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
)
_ASSIGNED_SECRET_RE = re.compile(
    r"(?i)\b(?:api[_-]?key|client[_-]?secret|password|private[_-]?key|secret|token)\b"
    + r"\s*[:=]\s*[\"']?([^\s\"'`]+)"
)
_REDACTED_VALUES = frozenset({"[REDACTED]", "<REDACTED>", "REDACTED", "***", "xxxxx"})


def _git(
    repo: Path,
    *args: str,
    accepted_returncodes: tuple[int, ...] = (0,),
    max_output_bytes: int | None = None,
) -> bytes:
    process = subprocess.Popen(
        ["git", *args],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    if max_output_bytes is None:
        stdout, stderr = process.communicate()
    else:
        stdout = process.stdout.read(max_output_bytes + 1)
        if len(stdout) > max_output_bytes:
            process.kill()
            _ = process.communicate()
            raise ValueError("model patch exceeds 10 MiB")
        remaining_stdout, stderr = process.communicate()
        stdout += remaining_stdout
    if process.returncode not in accepted_returncodes:
        detail = stderr.decode(errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return stdout


def current_commit(repo: Path) -> str | None:
    """Return the immutable starting commit, or None outside a Git worktree."""
    try:
        return _git(repo, "rev-parse", "HEAD").decode().strip()
    except Exception:
        return None


def _untracked_patch(repo: Path, excluded_path: str | None, max_bytes: int) -> bytes:
    paths = _git(
        repo,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        max_output_bytes=MAX_MODEL_PATCH_BYTES,
    ).split(b"\x00")
    chunks: list[bytes] = []
    total_bytes = 0
    for raw_path in paths:
        if not raw_path:
            continue
        path = raw_path.decode("utf-8")
        if excluded_path and (path == excluded_path or path.startswith(f"{excluded_path}/")):
            continue
        remaining_bytes = max_bytes - total_bytes
        if remaining_bytes <= 0 or (repo / path).lstat().st_size > remaining_bytes:
            raise ValueError("model patch exceeds 10 MiB")
        chunk = _git(
            repo,
            "diff",
            "--no-index",
            "--no-ext-diff",
            "--no-textconv",
            "--unified=3",
            "--",
            "/dev/null",
            path,
            accepted_returncodes=(0, 1),
            max_output_bytes=remaining_bytes,
        )
        chunks.append(chunk)
        total_bytes += len(chunk)
    return b"".join(chunks)


def _safe_path(path: str, prefix: str) -> bool:
    if not path.startswith(prefix):
        return False
    value = path.removeprefix(prefix)
    parts = value.split("/")
    return all(part not in {"", ".", ".."} for part in parts) and not PurePosixPath(value).is_absolute()


def _stats_and_paths(text: str) -> tuple[int, int, int]:
    file_count = additions = deletions = 0
    in_hunk = False
    for line in text.splitlines():
        if line.startswith("diff --git "):
            parts = shlex.split(line)
            if len(parts) != 4 or not _safe_path(parts[2], "a/") or not _safe_path(parts[3], "b/"):
                raise ValueError("unsafe model patch path")
            file_count += 1
            in_hunk = False
        elif line.startswith("@@ "):
            in_hunk = True
        elif in_hunk and line.startswith("+"):
            additions += 1
        elif in_hunk and line.startswith("-"):
            deletions += 1
    if file_count == 0:
        raise ValueError("model patch contains no file diff")
    return file_count, additions, deletions


def _has_unredacted_secret(text: str) -> bool:
    if any(pattern.search(text) for pattern in _DIRECT_SECRET_PATTERNS):
        return True
    for match in _ASSIGNED_SECRET_RE.finditer(text):
        value = match.group(1).rstrip(",;)")
        if value not in _REDACTED_VALUES and len(value) >= 8:
            return True
    return False


def _write_reference(trajectory_path: Path, reference: dict[str, str | int]) -> None:
    trajectory = json.loads(trajectory_path.read_text())
    extra = trajectory.setdefault("extra", {})
    if not isinstance(extra, dict):
        raise ValueError("ATIF extra must be an object")
    vals_extra = extra.setdefault("vals", {})
    if not isinstance(vals_extra, dict):
        raise ValueError("ATIF extra.vals must be an object")
    vals_extra["model_patch"] = reference
    temporary_path = trajectory_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(trajectory, indent=2, ensure_ascii=False) + "\n")
    temporary_path.replace(trajectory_path)


def write_model_patch(repo: Path, logs_dir: Path, trajectory_path: Path, base_commit: str) -> bool:
    """Write an optional validated model patch and attach its ATIF reference."""
    patch_path = logs_dir / "artifacts/model.patch"
    try:
        try:
            relative_logs = logs_dir.resolve().relative_to(repo.resolve()).as_posix()
        except ValueError:
            relative_logs = None
        if relative_logs == ".":
            raise ValueError("model patch requires logs outside the repository root")

        diff_paths = ["."]
        if relative_logs is not None:
            diff_paths.append(f":(exclude){relative_logs}/**")
        tracked = _git(
            repo,
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--unified=3",
            "--no-color",
            base_commit,
            "--",
            *diff_paths,
            max_output_bytes=MAX_MODEL_PATCH_BYTES,
        )
        patch = tracked + _untracked_patch(repo, relative_logs, MAX_MODEL_PATCH_BYTES - len(tracked))
        if not patch:
            return False
        if len(patch) > MAX_MODEL_PATCH_BYTES:
            raise ValueError("model patch exceeds 10 MiB")
        if b"\x00" in patch or any(marker in patch for marker in _BINARY_MARKERS):
            raise ValueError("binary model patches are not allowed")
        text = patch.decode("utf-8")
        file_count, additions, deletions = _stats_and_paths(text)
        if _has_unredacted_secret(text):
            raise ValueError("model patch contains a potential unredacted secret")

        patch_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path.write_bytes(patch)
        _write_reference(
            trajectory_path,
            {
                "path": "artifacts/model.patch",
                "media_type": "text/x-diff",
                "sha256": hashlib.sha256(patch).hexdigest(),
                "base_commit": base_commit,
                "file_count": file_count,
                "additions": additions,
                "deletions": deletions,
            },
        )
        return True
    except Exception as error:
        patch_path.unlink(missing_ok=True)
        logging.getLogger(__name__).warning("Model patch omitted: %s", error)
        return False
