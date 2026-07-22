from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
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
    r"(?i)(?<![A-Za-z0-9])"
    r"[\"']?(?P<key>[A-Za-z0-9_.-]*(?:api[_-]?key|apikey|access[_-]?token|accesstoken|"
    r"authorization|auth|client[_-]?secret|password|private[_-]?key|secret|token)[A-Za-z0-9_.-]*)[\"']?"
    r"\s*[:=]\s*(?P<value>(?:bearer|basic)\s+(?:\[[^\]\n]+\]|[^\s,}]+)|"
    r"\"[^\"\n]*\"|'[^'\n]*'|[^\s,}]+)"
)
_CREDENTIALED_URL_RE = re.compile(
    r"(?i)\b[a-z][a-z0-9+.-]*://[^/\s:@]+:(?P<secret>[^@\s/]+)@"
)
_REDACTED_VALUES = frozenset({"[REDACTED]", "<REDACTED>", "REDACTED", "***", "xxxxx"})
_NON_SECRET_TOKEN_KEYS = frozenset(
    {
        "cachedtokens",
        "completiontokens",
        "inputtokens",
        "maxtokens",
        "numtokens",
        "outputtokens",
        "prompttokens",
        "reasoningtokens",
        "tokencount",
        "tokenids",
        "totaltokens",
    }
)
_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


@dataclass(frozen=True)
class ModelPatchBaseline:
    base_commit: str
    tree: str
    excluded_paths: tuple[str, ...] = ()


def _git(
    repo: Path,
    *args: str,
    max_output_bytes: int | None = None,
    env: dict[str, str] | None = None,
) -> bytes:
    process = subprocess.Popen(
        ["git", *args],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE if max_output_bytes is None else subprocess.DEVNULL,
        env=env,
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
        remaining_stdout, _ = process.communicate()
        stdout += remaining_stdout
        stderr = b""
    if process.returncode != 0:
        detail = stderr.decode(errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return stdout


def _relative_excluded_paths(repo: Path, paths: tuple[Path, ...]) -> tuple[str, ...]:
    repo_root = Path(os.path.abspath(repo))
    excluded: set[str] = set()
    for path in paths:
        candidate = path if path.is_absolute() else repo_root / path
        try:
            relative = (
                Path(os.path.abspath(candidate)).relative_to(repo_root).as_posix()
            )
        except ValueError:
            continue
        if relative == ".":
            raise ValueError("cannot exclude the repository root")
        excluded.add(relative)
    return tuple(sorted(excluded))


def _worktree_tree(
    repo: Path, base_commit: str, excluded_paths: tuple[str, ...]
) -> str:
    with tempfile.TemporaryDirectory(prefix="vals-model-patch-index-") as temporary_dir:
        index_path = Path(temporary_dir) / "index"
        env = {**os.environ, "GIT_INDEX_FILE": str(index_path)}
        _ = _git(repo, "read-tree", base_commit, env=env)
        pathspecs = ["."]
        for path in excluded_paths:
            pathspecs.append(f":(exclude){path}")
        _ = _git(repo, "add", "-A", "--", *pathspecs, env=env)
        tree = _git(repo, "write-tree", env=env).decode().strip()
    if not _COMMIT_RE.fullmatch(tree):
        raise ValueError("invalid model patch tree id")
    return tree


def capture_model_patch_baseline(
    repo: Path,
    *,
    excluded_paths: tuple[Path, ...] = (),
) -> ModelPatchBaseline | None:
    """Capture the exact pre-model worktree without changing its real index."""
    try:
        base_commit = (
            _git(repo, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
        )
        if not _COMMIT_RE.fullmatch(base_commit):
            raise ValueError("invalid model patch base commit")
        relative_exclusions = _relative_excluded_paths(repo, excluded_paths)
        return ModelPatchBaseline(
            base_commit=base_commit,
            tree=_worktree_tree(repo, base_commit, relative_exclusions),
            excluded_paths=relative_exclusions,
        )
    except Exception as error:
        logging.getLogger(__name__).warning("Model patch baseline omitted: %s", error)
        return None


def _safe_path(path: str, prefix: str) -> bool:
    if not path.startswith(prefix):
        return False
    value = path.removeprefix(prefix)
    parts = value.split("/")
    return (
        all(part not in {"", ".", ".."} for part in parts)
        and not PurePosixPath(value).is_absolute()
    )


def _stats_and_paths(text: str) -> tuple[int, int, int]:
    file_count = additions = deletions = 0
    in_hunk = False
    for line in text.splitlines():
        if line.startswith("diff --git "):
            parts = shlex.split(line)
            if (
                len(parts) != 4
                or not _safe_path(parts[2], "a/")
                or not _safe_path(parts[3], "b/")
            ):
                raise ValueError("unsafe model patch path")
            file_count += 1
            in_hunk = False
        elif line.startswith("--- ") or line.startswith("+++ "):
            parts = shlex.split(line)
            expected_prefix = "a/" if line.startswith("--- ") else "b/"
            if len(parts) != 2 or (
                parts[1] != "/dev/null" and not _safe_path(parts[1], expected_prefix)
            ):
                raise ValueError("unsafe model patch path")
        elif line.startswith(("rename from ", "rename to ", "copy from ", "copy to ")):
            parts = shlex.split(line)
            if len(parts) != 3 or not _safe_path(parts[2], ""):
                raise ValueError("unsafe model patch path")
        elif line.startswith("@@ "):
            in_hunk = True
        elif in_hunk and line.startswith("+"):
            additions += 1
        elif in_hunk and line.startswith("-"):
            deletions += 1
    if file_count == 0:
        raise ValueError("model patch contains no file diff")
    return file_count, additions, deletions


def _is_explicitly_redacted(value: str) -> bool:
    value = value.strip().rstrip(",;)}")
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()
    if value in _REDACTED_VALUES:
        return True
    authorization = value.split(maxsplit=1)
    return (
        len(authorization) == 2
        and authorization[0].lower() in {"bearer", "basic"}
        and authorization[1] in _REDACTED_VALUES
    )


def _has_unredacted_secret(text: str) -> bool:
    if any(pattern.search(text) for pattern in _DIRECT_SECRET_PATTERNS):
        return True
    for match in _ASSIGNED_SECRET_RE.finditer(text):
        normalized_key = re.sub(r"[^a-z0-9]", "", match.group("key").lower())
        if normalized_key in _NON_SECRET_TOKEN_KEYS:
            continue
        if not _is_explicitly_redacted(match.group("value")):
            return True
    for match in _CREDENTIALED_URL_RE.finditer(text):
        if not _is_explicitly_redacted(match.group("secret")):
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
    temporary_path.write_text(
        json.dumps(trajectory, indent=2, ensure_ascii=False) + "\n"
    )
    temporary_path.replace(trajectory_path)


def write_model_patch(
    repo: Path,
    logs_dir: Path,
    trajectory_path: Path,
    baseline: ModelPatchBaseline,
) -> bool:
    """Write an optional validated model patch and attach its ATIF reference."""
    patch_path = logs_dir / "artifacts/model.patch"
    try:
        if not _COMMIT_RE.fullmatch(baseline.base_commit) or not _COMMIT_RE.fullmatch(
            baseline.tree
        ):
            raise ValueError("invalid model patch baseline")
        try:
            relative_logs = logs_dir.resolve().relative_to(repo.resolve()).as_posix()
        except ValueError:
            relative_logs = None
        if relative_logs == ".":
            raise ValueError("model patch requires logs outside the repository root")

        final_tree = _worktree_tree(repo, baseline.base_commit, baseline.excluded_paths)
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
            baseline.tree,
            final_tree,
            "--",
            *diff_paths,
            max_output_bytes=MAX_MODEL_PATCH_BYTES,
        )
        patch = tracked
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
                "base_commit": baseline.base_commit,
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
