from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

MAX_MODEL_PATCH_BYTES = 10 * 1024 * 1024
_STATE_DIRECTORY_PREFIX = "vals-model-patch-"
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
    state_dir: str | None = None


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


def _repository_git_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in (
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_NAMESPACE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_PREFIX",
        "GIT_WORK_TREE",
    ):
        env.pop(key, None)
    return env


def _real_object_directory(repo: Path) -> Path:
    raw_path = (
        _git(repo, "rev-parse", "--git-path", "objects", env=_repository_git_env())
        .decode()
        .strip()
    )
    object_directory = Path(raw_path)
    if not object_directory.is_absolute():
        object_directory = repo / object_directory
    return object_directory.resolve()


def _isolated_git_env(repo: Path, state_dir: Path, index_name: str) -> dict[str, str]:
    object_directory = state_dir / "objects"
    object_directory.mkdir(parents=True, exist_ok=True)
    env = _repository_git_env()
    env["GIT_INDEX_FILE"] = str(state_dir / index_name)
    env["GIT_OBJECT_DIRECTORY"] = str(object_directory)
    env["GIT_ALTERNATE_OBJECT_DIRECTORIES"] = str(_real_object_directory(repo))
    return env


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


def _is_excluded(path: str, excluded_paths: tuple[str, ...]) -> bool:
    return any(
        path == excluded or path.startswith(f"{excluded}/")
        for excluded in excluded_paths
    )


def _candidate_worktree_paths(
    repo: Path, base_commit: str, excluded_paths: tuple[str, ...]
) -> tuple[Path, ...]:
    env = _repository_git_env()
    commands = (
        ("diff", "--cached", "--name-only", "--no-renames", "-z", base_commit, "--"),
        ("diff", "--name-only", "--no-renames", "--no-ext-diff", "-z", "--"),
        ("ls-files", "--others", "--exclude-standard", "-z", "--"),
    )
    paths: set[str] = set()
    for command in commands:
        output = _git(repo, *command, max_output_bytes=MAX_MODEL_PATCH_BYTES, env=env)
        for raw_path in output.split(b"\x00"):
            if not raw_path:
                continue
            path = raw_path.decode("utf-8")
            if not _safe_path(path, ""):
                raise ValueError("unsafe model patch candidate path")
            if not _is_excluded(path, excluded_paths):
                paths.add(path)
    return tuple(repo / path for path in sorted(paths))


def _preflight_worktree(
    repo: Path,
    base_commit: str,
    excluded_paths: tuple[str, ...],
    *,
    validate_content: bool,
) -> None:
    candidates = _candidate_worktree_paths(repo, base_commit, excluded_paths)
    total_bytes = 0
    regular_files: list[Path] = []
    for path in candidates:
        try:
            file_stat = path.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError("model patch candidates must be regular files")
        total_bytes += file_stat.st_size
        if total_bytes > MAX_MODEL_PATCH_BYTES:
            raise ValueError("model patch candidate files exceed 10 MiB")
        regular_files.append(path)

    if not validate_content:
        return
    bytes_read = 0
    for path in regular_files:
        content = path.read_bytes()
        bytes_read += len(content)
        if bytes_read > MAX_MODEL_PATCH_BYTES:
            raise ValueError("model patch candidate files exceed 10 MiB")
        if b"\x00" in content:
            raise ValueError("binary model patches are not allowed")
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("model patch candidates must be UTF-8 text") from error
        if _has_unredacted_secret(text):
            raise ValueError("model patch contains a potential unredacted secret")


def _worktree_tree(
    repo: Path,
    base_commit: str,
    excluded_paths: tuple[str, ...],
    state_dir: Path,
    index_name: str,
    *,
    validate_content: bool,
) -> str:
    _preflight_worktree(
        repo,
        base_commit,
        excluded_paths,
        validate_content=validate_content,
    )
    env = _isolated_git_env(repo, state_dir, index_name)
    _ = _git(repo, "read-tree", base_commit, env=env)
    pathspecs = ["."]
    for path in excluded_paths:
        pathspecs.append(f":(exclude){path}")
    _ = _git(repo, "add", "-A", "--", *pathspecs, env=env)
    tree = _git(repo, "write-tree", env=env).decode().strip()
    if not _COMMIT_RE.fullmatch(tree):
        raise ValueError("invalid model patch tree id")
    return tree


def _best_effort_remove(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def _safe_model_patch_state_dir(baseline: ModelPatchBaseline) -> Path | None:
    if baseline.state_dir is None:
        return None
    temporary_root = Path(os.path.abspath(tempfile.gettempdir()))
    state_dir = Path(os.path.abspath(baseline.state_dir))
    if state_dir.parent != temporary_root or not state_dir.name.startswith(
        _STATE_DIRECTORY_PREFIX
    ):
        return None
    return state_dir


def cleanup_model_patch_baseline(baseline: ModelPatchBaseline | None) -> None:
    """Best-effort removal of isolated Git state; never affect agent scoring."""
    if baseline is None:
        return
    state_dir = _safe_model_patch_state_dir(baseline)
    if state_dir is None:
        return
    try:
        shutil.rmtree(state_dir, ignore_errors=True)
    except Exception:
        pass


def capture_model_patch_baseline(
    repo: Path,
    *,
    excluded_paths: tuple[Path, ...] = (),
) -> ModelPatchBaseline | None:
    """Capture the exact pre-model worktree without changing its real index."""
    state_dir: Path | None = None
    try:
        base_commit = (
            _git(
                repo,
                "rev-parse",
                "--verify",
                "HEAD^{commit}",
                env=_repository_git_env(),
            )
            .decode()
            .strip()
        )
        if not _COMMIT_RE.fullmatch(base_commit):
            raise ValueError("invalid model patch base commit")
        relative_exclusions = _relative_excluded_paths(repo, excluded_paths)
        state_dir = Path(tempfile.mkdtemp(prefix=_STATE_DIRECTORY_PREFIX))
        return ModelPatchBaseline(
            base_commit=base_commit,
            tree=_worktree_tree(
                repo,
                base_commit,
                relative_exclusions,
                state_dir,
                "baseline-index",
                validate_content=False,
            ),
            excluded_paths=relative_exclusions,
            state_dir=str(state_dir),
        )
    except Exception as error:
        if state_dir is not None:
            cleanup_model_patch_baseline(
                ModelPatchBaseline("", "", state_dir=str(state_dir))
            )
        logging.getLogger(__name__).warning("Model patch baseline omitted: %s", error)
        return None


def _safe_path(path: str, prefix: str) -> bool:
    if not path.startswith(prefix) or any(
        ord(character) < 32 or ord(character) == 127 for character in path
    ):
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
    patch_temp_path = logs_dir / "artifacts/.model.patch.tmp"
    reference_temp_path = trajectory_path.with_suffix(".json.tmp")
    try:
        if not _COMMIT_RE.fullmatch(baseline.base_commit) or not _COMMIT_RE.fullmatch(
            baseline.tree
        ):
            raise ValueError("invalid model patch baseline")
        state_dir = _safe_model_patch_state_dir(baseline)
        if state_dir is None or not state_dir.is_dir():
            raise ValueError("model patch isolated state is unavailable")
        try:
            relative_logs = logs_dir.resolve().relative_to(repo.resolve()).as_posix()
        except ValueError:
            relative_logs = None
        if relative_logs == ".":
            raise ValueError("model patch requires logs outside the repository root")

        final_tree = _worktree_tree(
            repo,
            baseline.base_commit,
            baseline.excluded_paths,
            state_dir,
            "final-index",
            validate_content=True,
        )
        diff_paths = ["."]
        if relative_logs is not None:
            diff_paths.append(f":(exclude){relative_logs}/**")
        isolated_env = _isolated_git_env(repo, state_dir, "final-index")
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
            env=isolated_env,
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
        patch_temp_path.write_bytes(patch)
        patch_temp_path.replace(patch_path)
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
        _best_effort_remove(patch_path)
        _best_effort_remove(patch_temp_path)
        _best_effort_remove(reference_temp_path)
        logging.getLogger(__name__).warning("Model patch omitted: %s", error)
        return False
    finally:
        cleanup_model_patch_baseline(baseline)
