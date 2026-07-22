from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import re
import secrets
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
_REGULAR_FILE_MODES = frozenset({"100644", "100755"})
_PATH_METADATA_PREFIXES = ("rename from ", "rename to ", "copy from ", "copy to ")


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


@dataclass(frozen=True)
class _FileSnapshot:
    device: int
    inode: int
    size: int
    modified_ns: int


def _snapshot(file_stat: os.stat_result) -> _FileSnapshot:
    return _FileSnapshot(
        device=file_stat.st_dev,
        inode=file_stat.st_ino,
        size=file_stat.st_size,
        modified_ns=file_stat.st_mtime_ns,
    )


def _same_file(left: _FileSnapshot, right: _FileSnapshot) -> bool:
    return left.device == right.device and left.inode == right.inode


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _regular_file_open_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)


def _stat_at(directory_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


@dataclass(frozen=True)
class _PinnedDirectory:
    file_descriptors: tuple[int, ...]
    component_names: tuple[str, ...]
    component_snapshots: tuple[_FileSnapshot, ...]

    @property
    def fd(self) -> int:
        return self.file_descriptors[-1]


@dataclass(frozen=True)
class _PinnedChildDirectory:
    parent: _PinnedDirectory
    name: str
    fd: int
    snapshot: _FileSnapshot


@dataclass(frozen=True)
class _PublishedReplacement:
    destination_name: str
    published_snapshot: _FileSnapshot
    backup_name: str
    backup_snapshot: _FileSnapshot


def _open_safe_directory(path: Path) -> _PinnedDirectory:
    absolute_path = Path(os.path.abspath(path))
    if not absolute_path.anchor:
        raise ValueError(f"unsafe model patch directory: {path}")

    file_descriptors: list[int] = []
    component_names: list[str] = []
    component_snapshots: list[_FileSnapshot] = []
    try:
        root_fd = os.open(absolute_path.anchor, _directory_open_flags())
        file_descriptors.append(root_fd)
        if not stat.S_ISDIR(os.fstat(root_fd).st_mode):
            raise ValueError(f"unsafe model patch directory root: {path}")

        for component in absolute_path.parts[1:]:
            parent_fd = file_descriptors[-1]
            before = _stat_at(parent_fd, component)
            if before is None or not stat.S_ISDIR(before.st_mode):
                raise ValueError(f"unsafe model patch path component: {component}")
            child_fd = os.open(component, _directory_open_flags(), dir_fd=parent_fd)
            file_descriptors.append(child_fd)
            opened = os.fstat(child_fd)
            after = _stat_at(parent_fd, component)
            if (
                after is None
                or not stat.S_ISDIR(opened.st_mode)
                or not stat.S_ISDIR(after.st_mode)
                or not _same_file(_snapshot(before), _snapshot(opened))
                or not _same_file(_snapshot(opened), _snapshot(after))
            ):
                raise ValueError(
                    f"model patch path component changed while opening: {component}"
                )
            component_names.append(component)
            component_snapshots.append(_snapshot(opened))

        return _PinnedDirectory(
            file_descriptors=tuple(file_descriptors),
            component_names=tuple(component_names),
            component_snapshots=tuple(component_snapshots),
        )
    except Exception:
        for file_fd in reversed(file_descriptors):
            _best_effort_close(file_fd)
        raise


def _revalidate_pinned_directory(directory: _PinnedDirectory) -> None:
    if len(directory.file_descriptors) != len(directory.component_names) + 1:
        raise ValueError("invalid pinned model patch directory")
    for index, (name, expected) in enumerate(
        zip(directory.component_names, directory.component_snapshots, strict=True)
    ):
        parent_fd = directory.file_descriptors[index]
        child_fd = directory.file_descriptors[index + 1]
        current = _stat_at(parent_fd, name)
        opened = os.fstat(child_fd)
        if (
            current is None
            or not stat.S_ISDIR(current.st_mode)
            or not stat.S_ISDIR(opened.st_mode)
            or not _same_file(expected, _snapshot(current))
            or not _same_file(expected, _snapshot(opened))
        ):
            raise ValueError(f"model patch directory component changed: {name}")


def _open_or_create_child_directory(
    parent: _PinnedDirectory, name: str
) -> _PinnedChildDirectory:
    if not _safe_path(name, "") or Path(name).name != name:
        raise ValueError("unsafe model patch directory name")
    _revalidate_pinned_directory(parent)
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent.fd)
    except FileExistsError:
        pass
    child_stat = _stat_at(parent.fd, name)
    if child_stat is None or not stat.S_ISDIR(child_stat.st_mode):
        raise ValueError(f"unsafe model patch directory component: {name}")
    child_fd = os.open(name, _directory_open_flags(), dir_fd=parent.fd)
    try:
        opened = os.fstat(child_fd)
        after = _stat_at(parent.fd, name)
        if (
            after is None
            or not stat.S_ISDIR(opened.st_mode)
            or not stat.S_ISDIR(after.st_mode)
            or not _same_file(_snapshot(child_stat), _snapshot(opened))
            or not _same_file(_snapshot(opened), _snapshot(after))
        ):
            raise ValueError(f"model patch directory component changed: {name}")
        return _PinnedChildDirectory(
            parent=parent,
            name=name,
            fd=child_fd,
            snapshot=_snapshot(opened),
        )
    except Exception:
        _best_effort_close(child_fd)
        raise


def _revalidate_child_directory(directory: _PinnedChildDirectory) -> None:
    _revalidate_pinned_directory(directory.parent)
    current = _stat_at(directory.parent.fd, directory.name)
    opened = os.fstat(directory.fd)
    if (
        current is None
        or not stat.S_ISDIR(current.st_mode)
        or not stat.S_ISDIR(opened.st_mode)
        or not _same_file(directory.snapshot, _snapshot(current))
        or not _same_file(directory.snapshot, _snapshot(opened))
    ):
        raise ValueError(f"model patch directory component changed: {directory.name}")


def _require_missing_at(directory_fd: int, name: str) -> None:
    if _stat_at(directory_fd, name) is not None:
        raise ValueError(f"unsafe pre-existing model patch output: {name}")


def _read_regular_file_at(directory_fd: int, name: str) -> tuple[bytes, _FileSnapshot]:
    before = _stat_at(directory_fd, name)
    if before is None or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"model patch input must be a regular file: {name}")
    file_fd = os.open(name, _regular_file_open_flags(), dir_fd=directory_fd)
    try:
        opened = os.fstat(file_fd)
        if not stat.S_ISREG(opened.st_mode) or not _same_file(
            _snapshot(before), _snapshot(opened)
        ):
            raise ValueError(f"model patch input changed while opening: {name}")
        source = os.fdopen(file_fd, "rb", closefd=True)
        file_fd = -1
        with source:
            content = source.read()
    finally:
        if file_fd >= 0:
            try:
                os.close(file_fd)
            except Exception:
                pass
    after = _stat_at(directory_fd, name)
    if after is None or _snapshot(after) != _snapshot(before):
        raise ValueError(f"model patch input changed while reading: {name}")
    return content, _snapshot(before)


def _create_secure_temp_file(
    directory_fd: int, prefix: str, content: bytes
) -> tuple[str, _FileSnapshot]:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_BINARY", 0)
    )
    for _ in range(64):
        name = f"{prefix}{secrets.token_hex(16)}.tmp"
        try:
            file_fd = os.open(name, flags, 0o600, dir_fd=directory_fd)
        except FileExistsError:
            continue
        try:
            view = memoryview(content)
            while view:
                written = os.write(file_fd, view)
                if written <= 0:
                    raise OSError("failed to write model patch temporary file")
                view = view[written:]
            os.fsync(file_fd)
            file_stat = os.fstat(file_fd)
            if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size != len(content):
                raise ValueError("model patch temporary output is not a regular file")
            return name, _snapshot(file_stat)
        except Exception:
            _best_effort_unlink_at(directory_fd, name)
            raise
        finally:
            try:
                os.close(file_fd)
            except Exception:
                pass
    raise FileExistsError("could not allocate a unique model patch temporary file")


def _best_effort_unlink_at(
    directory_fd: int | None,
    name: str | None,
    expected: _FileSnapshot | None = None,
) -> None:
    if directory_fd is None or name is None:
        return
    try:
        current = _stat_at(directory_fd, name)
        if current is None:
            return
        if expected is not None and _snapshot(current) != expected:
            return
        os.unlink(name, dir_fd=directory_fd)
    except Exception:
        pass


def _best_effort_close(file_fd: int | None) -> None:
    if file_fd is None:
        return
    try:
        os.close(file_fd)
    except Exception:
        pass


def _best_effort_close_pinned(directory: _PinnedDirectory | None) -> None:
    if directory is None:
        return
    for file_fd in reversed(directory.file_descriptors):
        _best_effort_close(file_fd)


def _unique_missing_name_at(directory_fd: int, prefix: str) -> str:
    for _ in range(64):
        name = f"{prefix}{secrets.token_hex(16)}.tmp"
        if _stat_at(directory_fd, name) is None:
            return name
    raise FileExistsError("could not allocate a unique model patch publication name")


def _restore_entry_without_clobbering(
    directory_fd: int,
    backup_name: str,
    destination_name: str,
    expected: _FileSnapshot,
) -> None:
    backup = _stat_at(directory_fd, backup_name)
    if backup is None or _snapshot(backup) != expected:
        return
    try:
        os.link(
            backup_name,
            destination_name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except Exception:
        return
    restored = _stat_at(directory_fd, destination_name)
    if restored is not None and _same_file(_snapshot(restored), expected):
        _best_effort_unlink_at(directory_fd, backup_name, expected)


def _publish_missing_regular_at(
    directory_fd: int,
    temporary_name: str,
    temporary_snapshot: _FileSnapshot,
    destination_name: str,
) -> _FileSnapshot:
    current_temporary = _stat_at(directory_fd, temporary_name)
    if (
        current_temporary is None
        or not stat.S_ISREG(current_temporary.st_mode)
        or _snapshot(current_temporary) != temporary_snapshot
    ):
        raise ValueError("model patch temporary file changed during publication")
    _require_missing_at(directory_fd, destination_name)
    os.link(
        temporary_name,
        destination_name,
        src_dir_fd=directory_fd,
        dst_dir_fd=directory_fd,
        follow_symlinks=False,
    )
    final = _stat_at(directory_fd, destination_name)
    if (
        final is None
        or not stat.S_ISREG(final.st_mode)
        or _snapshot(final) != temporary_snapshot
    ):
        _best_effort_unlink_at(directory_fd, destination_name, temporary_snapshot)
        raise ValueError("model patch artifact changed during publication")
    os.unlink(temporary_name, dir_fd=directory_fd)
    return _snapshot(final)


def _replace_existing_regular_at(
    directory_fd: int,
    temporary_name: str,
    temporary_snapshot: _FileSnapshot,
    destination_name: str,
    destination_snapshot: _FileSnapshot,
) -> _PublishedReplacement:
    backup_name = _unique_missing_name_at(
        directory_fd, f".{destination_name}.original."
    )
    moved_snapshot: _FileSnapshot | None = None
    published_snapshot: _FileSnapshot | None = None
    try:
        # Moving the destination first lets us inspect the exact directory entry
        # captured at the publication boundary without following a replacement
        # symlink or overwriting a concurrently-created entry.
        os.rename(
            destination_name,
            backup_name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        moved = _stat_at(directory_fd, backup_name)
        if moved is None:
            raise ValueError("trajectory disappeared during publication")
        moved_snapshot = _snapshot(moved)
        if not stat.S_ISREG(moved.st_mode) or moved_snapshot != destination_snapshot:
            raise ValueError("trajectory changed during publication")

        current_temporary = _stat_at(directory_fd, temporary_name)
        if (
            current_temporary is None
            or not stat.S_ISREG(current_temporary.st_mode)
            or _snapshot(current_temporary) != temporary_snapshot
        ):
            raise ValueError("trajectory temporary file changed during publication")
        os.link(
            temporary_name,
            destination_name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        published = _stat_at(directory_fd, destination_name)
        if (
            published is None
            or not stat.S_ISREG(published.st_mode)
            or _snapshot(published) != temporary_snapshot
        ):
            raise ValueError("trajectory changed during publication")
        published_snapshot = _snapshot(published)
        os.unlink(temporary_name, dir_fd=directory_fd)
        return _PublishedReplacement(
            destination_name=destination_name,
            published_snapshot=published_snapshot,
            backup_name=backup_name,
            backup_snapshot=destination_snapshot,
        )
    except Exception:
        if published_snapshot is not None:
            _best_effort_unlink_at(directory_fd, destination_name, published_snapshot)
        if moved_snapshot is not None:
            _restore_entry_without_clobbering(
                directory_fd,
                backup_name,
                destination_name,
                moved_snapshot,
            )
        raise


def _rollback_replacement_at(
    directory_fd: int | None, replacement: _PublishedReplacement | None
) -> None:
    if directory_fd is None or replacement is None:
        return
    _best_effort_unlink_at(
        directory_fd,
        replacement.destination_name,
        replacement.published_snapshot,
    )
    _restore_entry_without_clobbering(
        directory_fd,
        replacement.backup_name,
        replacement.destination_name,
        replacement.backup_snapshot,
    )


def _finalize_replacement_at(
    directory_fd: int, replacement: _PublishedReplacement
) -> None:
    _best_effort_unlink_at(
        directory_fd,
        replacement.backup_name,
        replacement.backup_snapshot,
    )


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


def _decode_git_path(value: str) -> str:
    if not value.startswith('"'):
        return value
    try:
        decoded = ast.literal_eval(value)
    except (SyntaxError, ValueError) as error:
        raise ValueError("unsafe model patch path") from error
    if not isinstance(decoded, str):
        raise ValueError("unsafe model patch path")
    return decoded


def _patch_header_path(value: str) -> str:
    if not value.startswith('"'):
        return value.split("\t", 1)[0]
    try:
        parts = shlex.split(value, posix=False)
    except ValueError as error:
        raise ValueError("unsafe model patch path") from error
    if len(parts) != 1:
        raise ValueError("unsafe model patch path")
    return _decode_git_path(parts[0])


def _diff_header_paths(line: str) -> tuple[str, str]:
    try:
        parts = shlex.split(line, posix=False)
    except ValueError as error:
        raise ValueError("unsafe model patch path") from error
    if len(parts) == 4:
        return _decode_git_path(parts[2]), _decode_git_path(parts[3])

    payload = line.removeprefix("diff --git ")
    if '"' in payload:
        raise ValueError("unsafe model patch path")
    candidates: list[tuple[str, str]] = []
    offset = 0
    while (delimiter := payload.find(" b/", offset)) != -1:
        old_path = payload[:delimiter]
        new_path = payload[delimiter + 1 :]
        if _safe_path(old_path, "a/") and _safe_path(new_path, "b/"):
            candidates.append((old_path, new_path))
        offset = delimiter + 1
    if len(candidates) != 1:
        raise ValueError("unsafe model patch path")
    return candidates[0]


def _stats_and_paths(text: str) -> tuple[int, int, int]:
    file_count = additions = deletions = 0
    in_hunk = False
    entry_open = False
    regular_mode_proven = False
    for line in text.splitlines():
        if line.startswith("diff --git "):
            if entry_open and not regular_mode_proven:
                raise ValueError("model patch entries must prove a regular file mode")
            old_path, new_path = _diff_header_paths(line)
            if not _safe_path(old_path, "a/") or not _safe_path(new_path, "b/"):
                raise ValueError("unsafe model patch path")
            file_count += 1
            entry_open = True
            regular_mode_proven = False
            in_hunk = False
        elif line.startswith("--- ") or line.startswith("+++ "):
            path = _patch_header_path(line[4:])
            expected_prefix = "a/" if line.startswith("--- ") else "b/"
            if path != "/dev/null" and not _safe_path(path, expected_prefix):
                raise ValueError("unsafe model patch path")
        elif line.startswith(
            ("new file mode ", "deleted file mode ", "old mode ", "new mode ")
        ):
            mode = line.rsplit(" ", 1)[-1]
            if mode not in _REGULAR_FILE_MODES:
                raise ValueError(
                    f"model patch entries must be regular files, got mode {mode!r}"
                )
            regular_mode_proven = True
        elif line.startswith("index "):
            parts = line.split()
            if len(parts) == 3 and parts[-1] not in _REGULAR_FILE_MODES:
                raise ValueError(
                    f"model patch entries must be regular files, got mode {parts[-1]!r}"
                )
            if len(parts) == 3:
                regular_mode_proven = True
        elif line.startswith(_PATH_METADATA_PREFIXES):
            path = _decode_git_path(line.split(" ", 2)[2])
            if not _safe_path(path, ""):
                raise ValueError("unsafe model patch path")
        elif line.startswith("@@ "):
            in_hunk = True
        elif in_hunk and line.startswith("+"):
            additions += 1
        elif in_hunk and line.startswith("-"):
            deletions += 1
    if file_count == 0:
        raise ValueError("model patch contains no file diff")
    if entry_open and not regular_mode_proven:
        raise ValueError("model patch entries must prove a regular file mode")
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


def _trajectory_with_reference(
    content: bytes, reference: dict[str, str | int]
) -> bytes:
    trajectory = json.loads(content.decode("utf-8"))
    extra = trajectory.setdefault("extra", {})
    if not isinstance(extra, dict):
        raise ValueError("ATIF extra must be an object")
    vals_extra = extra.setdefault("vals", {})
    if not isinstance(vals_extra, dict):
        raise ValueError("ATIF extra.vals must be an object")
    vals_extra["model_patch"] = reference
    return (json.dumps(trajectory, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def write_model_patch(
    repo: Path,
    logs_dir: Path,
    trajectory_path: Path,
    baseline: ModelPatchBaseline,
) -> bool:
    """Write an optional validated model patch and attach its ATIF reference."""
    logs_directory: _PinnedDirectory | None = None
    artifacts_directory: _PinnedChildDirectory | None = None
    patch_temp_name: str | None = None
    trajectory_temp_name: str | None = None
    published_patch: _FileSnapshot | None = None
    published_trajectory: _PublishedReplacement | None = None
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
            "--no-renames",
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

        absolute_logs_dir = Path(os.path.abspath(logs_dir))
        absolute_trajectory_path = Path(os.path.abspath(trajectory_path))
        if absolute_trajectory_path.parent != absolute_logs_dir or not _safe_path(
            trajectory_path.name, ""
        ):
            raise ValueError(
                "trajectory path must be a direct child of the logs directory"
            )

        logs_directory = _open_safe_directory(logs_dir)
        logs_fd = logs_directory.fd
        trajectory_name = trajectory_path.name
        legacy_reference_temp_name = trajectory_path.with_suffix(".json.tmp").name
        _require_missing_at(logs_fd, legacy_reference_temp_name)
        trajectory_content, original_trajectory = _read_regular_file_at(
            logs_fd, trajectory_name
        )
        reference = {
            "path": "artifacts/model.patch",
            "media_type": "text/x-diff",
            "sha256": hashlib.sha256(patch).hexdigest(),
            "base_commit": baseline.base_commit,
            "file_count": file_count,
            "additions": additions,
            "deletions": deletions,
        }
        updated_trajectory = _trajectory_with_reference(trajectory_content, reference)

        artifacts_directory = _open_or_create_child_directory(
            logs_directory, "artifacts"
        )
        artifacts_fd = artifacts_directory.fd
        _require_missing_at(artifacts_fd, ".model.patch.tmp")
        _require_missing_at(artifacts_fd, "model.patch")
        patch_temp_name, patch_temp = _create_secure_temp_file(
            artifacts_fd, ".model.patch.", patch
        )
        trajectory_temp_name, trajectory_temp = _create_secure_temp_file(
            logs_fd,
            f".{trajectory_name}.",
            updated_trajectory,
        )

        _revalidate_pinned_directory(logs_directory)
        _revalidate_child_directory(artifacts_directory)
        _require_missing_at(logs_fd, legacy_reference_temp_name)
        _require_missing_at(artifacts_fd, ".model.patch.tmp")
        _require_missing_at(artifacts_fd, "model.patch")
        current_trajectory = _stat_at(logs_fd, trajectory_name)
        current_patch_temp = _stat_at(artifacts_fd, patch_temp_name)
        if (
            current_trajectory is None
            or _snapshot(current_trajectory) != original_trajectory
        ):
            raise ValueError("trajectory changed during model patch collection")
        if (
            current_patch_temp is None
            or not stat.S_ISREG(current_patch_temp.st_mode)
            or _snapshot(current_patch_temp) != patch_temp
        ):
            raise ValueError("model patch temporary file changed during collection")

        published_patch = _publish_missing_regular_at(
            artifacts_fd,
            patch_temp_name,
            patch_temp,
            "model.patch",
        )
        patch_temp_name = None
        final_patch = _stat_at(artifacts_fd, "model.patch")
        if final_patch is None or not stat.S_ISREG(final_patch.st_mode):
            raise ValueError("final model patch artifact is not a regular file")
        if published_patch != patch_temp:
            raise ValueError("model patch artifact changed during publication")

        _revalidate_pinned_directory(logs_directory)
        _revalidate_child_directory(artifacts_directory)
        current_patch = _stat_at(artifacts_fd, "model.patch")
        current_trajectory = _stat_at(logs_fd, trajectory_name)
        current_trajectory_temp = _stat_at(logs_fd, trajectory_temp_name)
        if current_patch is None or _snapshot(current_patch) != published_patch:
            raise ValueError(
                "model patch artifact changed before trajectory publication"
            )
        if (
            current_trajectory is None
            or _snapshot(current_trajectory) != original_trajectory
        ):
            raise ValueError("trajectory changed before model patch publication")
        if (
            current_trajectory_temp is None
            or not stat.S_ISREG(current_trajectory_temp.st_mode)
            or _snapshot(current_trajectory_temp) != trajectory_temp
        ):
            raise ValueError("trajectory temporary file changed during publication")
        published_trajectory = _replace_existing_regular_at(
            logs_fd,
            trajectory_temp_name,
            trajectory_temp,
            trajectory_name,
            original_trajectory,
        )
        trajectory_temp_name = None
        _revalidate_pinned_directory(logs_directory)
        _revalidate_child_directory(artifacts_directory)
        _finalize_replacement_at(logs_fd, published_trajectory)
        published_trajectory = None
        return True
    except Exception as error:
        logs_fd = logs_directory.fd if logs_directory is not None else None
        artifacts_fd = (
            artifacts_directory.fd if artifacts_directory is not None else None
        )
        _best_effort_unlink_at(artifacts_fd, patch_temp_name)
        _best_effort_unlink_at(logs_fd, trajectory_temp_name)
        _rollback_replacement_at(logs_fd, published_trajectory)
        if published_patch is not None:
            _best_effort_unlink_at(artifacts_fd, "model.patch", published_patch)
        logging.getLogger(__name__).warning("Model patch omitted: %s", error)
        return False
    finally:
        if artifacts_directory is not None:
            _best_effort_close(artifacts_directory.fd)
        _best_effort_close_pinned(logs_directory)
        cleanup_model_patch_baseline(baseline)
