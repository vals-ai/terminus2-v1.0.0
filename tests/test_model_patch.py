from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest
import terminus2.model_patch as model_patch_module

from terminus2.model_patch import (
    MAX_MODEL_PATCH_BYTES,
    _stats_and_paths,
    capture_model_patch_baseline,
    write_model_patch,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _ = _git(repo, "init")
    _ = _git(repo, "config", "user.email", "test@example.com")
    _ = _git(repo, "config", "user.name", "Test")
    (repo / "example.py").write_text("value = 1\n")
    _ = _git(repo, "add", "example.py")
    _ = _git(repo, "commit", "-m", "base")
    return repo, _git(repo, "rev-parse", "HEAD")


def _trajectory(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.6",
                "agent": {"name": "terminus-2", "version": "2.0.0"},
                "steps": [{"step_id": 1, "source": "user", "message": "task"}],
            }
        )
    )


def _object_inventory(repo: Path) -> tuple[str, tuple[tuple[str, int, str], ...]]:
    object_dir = Path(_git(repo, "rev-parse", "--git-path", "objects"))
    if not object_dir.is_absolute():
        object_dir = repo / object_dir
    files = tuple(
        (
            path.relative_to(object_dir).as_posix(),
            path.stat().st_size,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(object_dir.rglob("*"))
        if path.is_file()
    )
    return _git(repo, "count-objects", "-v"), files


def test_baseline_uses_isolated_index_and_diffs_model_changes_only(
    tmp_path: Path,
) -> None:
    repo, base_commit = _repo(tmp_path)
    (repo / "example.py").write_text("value = 'setup'\n")
    _ = _git(repo, "add", "example.py")
    (repo / "setup-only.txt").write_text("pre-model harness setup\n")
    status_before = _git(repo, "status", "--short")
    cached_before = _git(repo, "diff", "--cached")

    baseline = capture_model_patch_baseline(repo)

    assert baseline is not None
    assert baseline.base_commit == base_commit
    assert _git(repo, "status", "--short") == status_before
    assert _git(repo, "diff", "--cached") == cached_before

    (repo / "example.py").write_text("value = 'model'\n")
    status_after_model = _git(repo, "status", "--short")
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    trajectory_path = logs_dir / "trajectory.json"
    _trajectory(trajectory_path)

    assert write_model_patch(repo, logs_dir, trajectory_path, baseline)

    patch = (logs_dir / "artifacts/model.patch").read_text()
    assert "-value = 'setup'" in patch
    assert "+value = 'model'" in patch
    assert "value = 1" not in patch
    assert "setup-only.txt" not in patch
    assert _git(repo, "status", "--short") == status_after_model
    assert _git(repo, "diff", "--cached") == cached_before


def test_writes_text_patch_and_atif_reference(tmp_path: Path) -> None:
    repo, base_commit = _repo(tmp_path)
    objects_before = _object_inventory(repo)
    baseline = capture_model_patch_baseline(repo)
    assert baseline is not None
    assert baseline.state_dir is not None
    state_dir = Path(baseline.state_dir)
    assert state_dir.is_dir()
    assert _object_inventory(repo) == objects_before
    (repo / "example.py").write_text("value = 2\n")
    (repo / "new.py").write_text("new_value = 3\n")
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    trajectory_path = logs_dir / "trajectory.json"
    _trajectory(trajectory_path)

    assert write_model_patch(repo, logs_dir, trajectory_path, baseline)

    patch = (logs_dir / "artifacts/model.patch").read_bytes()
    trajectory = json.loads(trajectory_path.read_text())
    reference = trajectory["extra"]["vals"]["model_patch"]
    assert reference == {
        "path": "artifacts/model.patch",
        "media_type": "text/x-diff",
        "sha256": hashlib.sha256(patch).hexdigest(),
        "base_commit": base_commit,
        "file_count": 2,
        "additions": 2,
        "deletions": 1,
    }
    assert _object_inventory(repo) == objects_before
    assert not state_dir.exists()
    assert (logs_dir / "artifacts/model.patch").is_file()
    assert not (logs_dir / "artifacts/model.patch").is_symlink()
    assert trajectory_path.is_file()
    assert not trajectory_path.is_symlink()
    assert not list(logs_dir.glob(".*.tmp"))
    assert not list((logs_dir / "artifacts").glob(".*.tmp"))


def test_diffs_preexisting_untracked_file_from_its_pre_model_content(
    tmp_path: Path,
) -> None:
    repo, _ = _repo(tmp_path)
    (repo / "notes.txt").write_text("setup content\n")
    baseline = capture_model_patch_baseline(repo)
    assert baseline is not None
    (repo / "notes.txt").write_text("model content\n")
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    trajectory_path = logs_dir / "trajectory.json"
    _trajectory(trajectory_path)

    assert write_model_patch(repo, logs_dir, trajectory_path, baseline)

    patch = (logs_dir / "artifacts/model.patch").read_text()
    assert "-setup content" in patch
    assert "+model content" in patch


def test_excludes_private_harness_paths_from_both_trees(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path)
    private_task = repo / "private-task.md"
    private_task.write_text("hidden question\n")
    baseline = capture_model_patch_baseline(repo, excluded_paths=(private_task,))
    assert baseline is not None
    private_task.write_text("hidden question with model notes\n")
    (repo / "example.py").write_text("value = 2\n")
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    trajectory_path = logs_dir / "trajectory.json"
    _trajectory(trajectory_path)

    assert write_model_patch(repo, logs_dir, trajectory_path, baseline)

    patch = (logs_dir / "artifacts/model.patch").read_text()
    assert "example.py" in patch
    assert "private-task.md" not in patch
    assert "hidden question" not in patch


def test_excludes_private_harness_symlink_without_following_it(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path)
    private_source = tmp_path / "private-source.md"
    private_source.write_text("hidden question\n")
    private_task = repo / "private-task.md"
    private_task.symlink_to(private_source)
    baseline = capture_model_patch_baseline(repo, excluded_paths=(private_task,))
    assert baseline is not None
    private_task.unlink()
    private_task.write_text("hidden question copied into worktree\n")
    (repo / "example.py").write_text("value = 2\n")
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    trajectory_path = logs_dir / "trajectory.json"
    _trajectory(trajectory_path)

    assert write_model_patch(repo, logs_dir, trajectory_path, baseline)

    patch = (logs_dir / "artifacts/model.patch").read_text()
    assert "example.py" in patch
    assert "private-task.md" not in patch
    assert "hidden question" not in patch


def test_excludes_private_harness_directory_recursively(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path)
    private_dir = repo / "harness"
    private_dir.mkdir()
    private_task = private_dir / "private-task.md"
    private_task.write_text("hidden question\n")
    baseline = capture_model_patch_baseline(repo, excluded_paths=(private_dir,))
    assert baseline is not None
    private_task.write_text("hidden question with model notes\n")
    (private_dir / "model-output.txt").write_text("private tool output\n")
    (repo / "example.py").write_text("value = 2\n")
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    trajectory_path = logs_dir / "trajectory.json"
    _trajectory(trajectory_path)

    assert write_model_patch(repo, logs_dir, trajectory_path, baseline)

    patch = (logs_dir / "artifacts/model.patch").read_text()
    assert "example.py" in patch
    assert "harness/" not in patch
    assert "hidden question" not in patch
    assert "private tool output" not in patch


@pytest.mark.parametrize(
    "secret_line",
    [
        'API_KEY = "sk-live-example-secret-value"',
        '"apiKey": "tiny"',
        "access_token='short'",
        'clientAuthToken = "abc"',
        'Authorization: Bearer "opaque-token"',
        'proxy_authorization = "Bearer another-token"',
        'endpoint = "https://user:password@example.com/api"',
        'DATABASE_API_KEY: "db-key"',
        '"foo_client_secret": "client-secret"',
        "service_access_token = abc",
        'customAuth = "Basic auth-value"',
    ],
)
def test_omits_patch_when_diff_contains_secret(
    tmp_path: Path, secret_line: str
) -> None:
    repo, _ = _repo(tmp_path)
    objects_before = _object_inventory(repo)
    baseline = capture_model_patch_baseline(repo)
    assert baseline is not None
    assert baseline.state_dir is not None
    state_dir = Path(baseline.state_dir)
    assert _object_inventory(repo) == objects_before
    (repo / "example.py").write_text(f"{secret_line}\n")
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    trajectory_path = logs_dir / "trajectory.json"
    _trajectory(trajectory_path)
    original = trajectory_path.read_text()

    assert not write_model_patch(repo, logs_dir, trajectory_path, baseline)
    assert not (logs_dir / "artifacts/model.patch").exists()
    assert trajectory_path.read_text() == original
    assert _object_inventory(repo) == objects_before
    assert not state_dir.exists()


@pytest.mark.parametrize(
    "redacted_line",
    [
        '"apiKey": "[REDACTED]"',
        "access_token='<REDACTED>'",
        "clientAuthToken = REDACTED",
        "Authorization: Bearer [REDACTED]",
        'endpoint = "https://user:[REDACTED]@example.com/api"',
        'DATABASE_API_KEY: "[REDACTED]"',
        '"foo_client_secret": "***"',
        "service_access_token = <REDACTED>",
        'customAuth = "Basic [REDACTED]"',
    ],
)
def test_allows_explicitly_redacted_secret_values(
    tmp_path: Path, redacted_line: str
) -> None:
    repo, _ = _repo(tmp_path)
    baseline = capture_model_patch_baseline(repo)
    assert baseline is not None
    (repo / "example.py").write_text(f"{redacted_line}\n")
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    trajectory_path = logs_dir / "trajectory.json"
    _trajectory(trajectory_path)

    assert write_model_patch(repo, logs_dir, trajectory_path, baseline)


@pytest.mark.parametrize(
    "metric_line",
    [
        "total_tokens = 123",
        '"prompt_tokens": 456',
        "token_count: 12",
        "max_tokens = 4096",
    ],
)
def test_allows_known_non_secret_token_metrics(
    tmp_path: Path, metric_line: str
) -> None:
    repo, _ = _repo(tmp_path)
    baseline = capture_model_patch_baseline(repo)
    assert baseline is not None
    (repo / "example.py").write_text(f"{metric_line}\n")
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    trajectory_path = logs_dir / "trajectory.json"
    _trajectory(trajectory_path)

    assert write_model_patch(repo, logs_dir, trajectory_path, baseline)


def test_omits_binary_patch(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path)
    objects_before = _object_inventory(repo)
    baseline = capture_model_patch_baseline(repo)
    assert baseline is not None
    assert baseline.state_dir is not None
    state_dir = Path(baseline.state_dir)
    (repo / "asset.bin").write_bytes(b"\x00\x01\x02")
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    trajectory_path = logs_dir / "trajectory.json"
    _trajectory(trajectory_path)

    assert not write_model_patch(repo, logs_dir, trajectory_path, baseline)
    assert not (logs_dir / "artifacts/model.patch").exists()
    assert _object_inventory(repo) == objects_before
    assert not state_dir.exists()


def test_omits_non_utf8_patch_without_writing_repository_objects(
    tmp_path: Path,
) -> None:
    repo, _ = _repo(tmp_path)
    objects_before = _object_inventory(repo)
    baseline = capture_model_patch_baseline(repo)
    assert baseline is not None
    assert baseline.state_dir is not None
    state_dir = Path(baseline.state_dir)
    (repo / "invalid.txt").write_bytes(b"\xff\xfe")
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    trajectory_path = logs_dir / "trajectory.json"
    _trajectory(trajectory_path)

    assert not write_model_patch(repo, logs_dir, trajectory_path, baseline)
    assert not (logs_dir / "artifacts/model.patch").exists()
    assert _object_inventory(repo) == objects_before
    assert not state_dir.exists()


def test_omits_unsafe_candidate_path_without_writing_repository_objects(
    tmp_path: Path,
) -> None:
    repo, _ = _repo(tmp_path)
    objects_before = _object_inventory(repo)
    baseline = capture_model_patch_baseline(repo)
    assert baseline is not None
    assert baseline.state_dir is not None
    state_dir = Path(baseline.state_dir)
    (repo / "unsafe\npath.txt").write_text("model output\n")
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    trajectory_path = logs_dir / "trajectory.json"
    _trajectory(trajectory_path)

    assert not write_model_patch(repo, logs_dir, trajectory_path, baseline)
    assert not (logs_dir / "artifacts/model.patch").exists()
    assert _object_inventory(repo) == objects_before
    assert not state_dir.exists()


def test_omits_symlink_candidate_without_writing_repository_objects(
    tmp_path: Path,
) -> None:
    repo, _ = _repo(tmp_path)
    objects_before = _object_inventory(repo)
    baseline = capture_model_patch_baseline(repo)
    assert baseline is not None
    assert baseline.state_dir is not None
    state_dir = Path(baseline.state_dir)
    target = tmp_path / "outside.txt"
    target.write_text("outside\n")
    (repo / "link.txt").symlink_to(target)
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    trajectory_path = logs_dir / "trajectory.json"
    _trajectory(trajectory_path)

    assert not write_model_patch(repo, logs_dir, trajectory_path, baseline)
    assert not (logs_dir / "artifacts/model.patch").exists()
    assert _object_inventory(repo) == objects_before
    assert not state_dir.exists()


def test_excludes_agent_logs_when_logs_are_inside_worktree(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path)
    baseline = capture_model_patch_baseline(repo)
    assert baseline is not None
    (repo / "example.py").write_text("value = 2\n")
    logs_dir = repo / "logs"
    logs_dir.mkdir()
    trajectory_path = logs_dir / "trajectory.json"
    _trajectory(trajectory_path)

    assert write_model_patch(repo, logs_dir, trajectory_path, baseline)

    trajectory = json.loads(trajectory_path.read_text())
    assert trajectory["extra"]["vals"]["model_patch"]["file_count"] == 1
    assert (
        b"logs/trajectory.json" not in (logs_dir / "artifacts/model.patch").read_bytes()
    )


def test_omits_oversized_patch_without_writing_artifact(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path)
    objects_before = _object_inventory(repo)
    baseline = capture_model_patch_baseline(repo)
    assert baseline is not None
    assert baseline.state_dir is not None
    state_dir = Path(baseline.state_dir)
    (repo / "large.txt").write_bytes(b"x" * (MAX_MODEL_PATCH_BYTES + 1))
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    trajectory_path = logs_dir / "trajectory.json"
    _trajectory(trajectory_path)

    assert not write_model_patch(repo, logs_dir, trajectory_path, baseline)
    assert not (logs_dir / "artifacts/model.patch").exists()
    assert _object_inventory(repo) == objects_before
    assert not state_dir.exists()


def test_omits_patch_for_invalid_base_commit(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path)
    baseline = capture_model_patch_baseline(repo)
    assert baseline is not None
    (repo / "example.py").write_text("value = 2\n")
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    trajectory_path = logs_dir / "trajectory.json"
    _trajectory(trajectory_path)
    original = trajectory_path.read_text()

    assert not write_model_patch(
        repo, logs_dir, trajectory_path, replace(baseline, base_commit="abc")
    )
    assert not (logs_dir / "artifacts/model.patch").exists()
    assert trajectory_path.read_text() == original


def test_reference_collection_failure_is_fail_open(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path)
    baseline = capture_model_patch_baseline(repo)
    assert baseline is not None
    (repo / "example.py").write_text("value = 2\n")
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    trajectory_path = logs_dir / "trajectory.json"
    trajectory_path.write_text('{"schema_version":"ATIF-v1.6","extra":[]}\n')
    original = trajectory_path.read_text()

    assert not write_model_patch(repo, logs_dir, trajectory_path, baseline)
    assert not (logs_dir / "artifacts/model.patch").exists()
    assert trajectory_path.read_text() == original


@pytest.mark.parametrize(
    "blocking_path",
    [
        "artifacts/model.patch",
        "artifacts/.model.patch.tmp",
        "trajectory.json.tmp",
    ],
)
def test_collection_cleanup_failure_is_fail_open(
    tmp_path: Path,
    blocking_path: str,
) -> None:
    repo, _ = _repo(tmp_path)
    objects_before = _object_inventory(repo)
    baseline = capture_model_patch_baseline(repo)
    assert baseline is not None
    assert baseline.state_dir is not None
    state_dir = Path(baseline.state_dir)
    (repo / "example.py").write_text("value = 2\n")
    logs_dir = tmp_path / "logs"
    blocked_path = logs_dir / blocking_path
    blocked_path.mkdir(parents=True)
    trajectory_path = logs_dir / "trajectory.json"
    _trajectory(trajectory_path)
    original = trajectory_path.read_text()

    assert not write_model_patch(repo, logs_dir, trajectory_path, baseline)
    assert blocked_path.is_dir()
    assert trajectory_path.read_text() == original
    assert _object_inventory(repo) == objects_before
    assert not state_dir.exists()


@pytest.mark.parametrize(
    "legacy_temp_path",
    [
        "artifacts/.model.patch.tmp",
        "trajectory.json.tmp",
    ],
)
def test_rejects_legacy_temp_symlink_without_touching_sentinel(
    tmp_path: Path,
    legacy_temp_path: str,
) -> None:
    repo, _ = _repo(tmp_path)
    baseline = capture_model_patch_baseline(repo)
    assert baseline is not None
    (repo / "example.py").write_text("value = 2\n")
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    trajectory_path = logs_dir / "trajectory.json"
    _trajectory(trajectory_path)
    original_trajectory = trajectory_path.read_text()
    sentinel = tmp_path / "scored-output.txt"
    sentinel.write_text("scored output\n")
    legacy_path = logs_dir / legacy_temp_path
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.symlink_to(sentinel)

    assert not write_model_patch(repo, logs_dir, trajectory_path, baseline)
    assert sentinel.read_text() == "scored output\n"
    assert legacy_path.is_symlink()
    assert trajectory_path.read_text() == original_trajectory
    assert not (logs_dir / "artifacts/model.patch").exists()


def test_rejects_final_patch_symlink_without_touching_sentinel(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path)
    baseline = capture_model_patch_baseline(repo)
    assert baseline is not None
    (repo / "example.py").write_text("value = 2\n")
    logs_dir = tmp_path / "logs"
    artifacts_dir = logs_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    trajectory_path = logs_dir / "trajectory.json"
    _trajectory(trajectory_path)
    original_trajectory = trajectory_path.read_text()
    sentinel = tmp_path / "scored-output.txt"
    sentinel.write_text("scored output\n")
    patch_path = artifacts_dir / "model.patch"
    patch_path.symlink_to(sentinel)

    assert not write_model_patch(repo, logs_dir, trajectory_path, baseline)
    assert sentinel.read_text() == "scored output\n"
    assert patch_path.is_symlink()
    assert trajectory_path.read_text() == original_trajectory


def test_rejects_trajectory_symlink_without_touching_scored_file(
    tmp_path: Path,
) -> None:
    repo, _ = _repo(tmp_path)
    baseline = capture_model_patch_baseline(repo)
    assert baseline is not None
    (repo / "example.py").write_text("value = 2\n")
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    scored_trajectory = tmp_path / "scored-trajectory.json"
    _trajectory(scored_trajectory)
    original_scored_trajectory = scored_trajectory.read_text()
    trajectory_path = logs_dir / "trajectory.json"
    trajectory_path.symlink_to(scored_trajectory)

    assert not write_model_patch(repo, logs_dir, trajectory_path, baseline)
    assert scored_trajectory.read_text() == original_scored_trajectory
    assert trajectory_path.is_symlink()
    assert not (logs_dir / "artifacts/model.patch").exists()


def test_rejects_symlinked_artifacts_parent_without_writing_outside_logs(
    tmp_path: Path,
) -> None:
    repo, _ = _repo(tmp_path)
    baseline = capture_model_patch_baseline(repo)
    assert baseline is not None
    (repo / "example.py").write_text("value = 2\n")
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    trajectory_path = logs_dir / "trajectory.json"
    _trajectory(trajectory_path)
    original_trajectory = trajectory_path.read_text()
    outside_dir = tmp_path / "scored-directory"
    outside_dir.mkdir()
    sentinel = outside_dir / "scored-output.txt"
    sentinel.write_text("scored output\n")
    (logs_dir / "artifacts").symlink_to(outside_dir, target_is_directory=True)

    assert not write_model_patch(repo, logs_dir, trajectory_path, baseline)
    assert sentinel.read_text() == "scored output\n"
    assert not (outside_dir / "model.patch").exists()
    assert trajectory_path.read_text() == original_trajectory


def test_rejects_symlinked_logs_directory_without_touching_scored_files(
    tmp_path: Path,
) -> None:
    repo, _ = _repo(tmp_path)
    baseline = capture_model_patch_baseline(repo)
    assert baseline is not None
    (repo / "example.py").write_text("value = 2\n")
    real_logs = tmp_path / "scored-logs"
    real_logs.mkdir()
    scored_trajectory = real_logs / "trajectory.json"
    _trajectory(scored_trajectory)
    original_scored_trajectory = scored_trajectory.read_text()
    logs_dir = tmp_path / "logs-link"
    logs_dir.symlink_to(real_logs, target_is_directory=True)
    trajectory_path = logs_dir / "trajectory.json"

    assert not write_model_patch(repo, logs_dir, trajectory_path, baseline)
    assert scored_trajectory.read_text() == original_scored_trajectory
    assert not (real_logs / "artifacts/model.patch").exists()


def test_rejects_symlinked_logs_parent_component_without_touching_scored_files(
    tmp_path: Path,
) -> None:
    repo, _ = _repo(tmp_path)
    baseline = capture_model_patch_baseline(repo)
    assert baseline is not None
    (repo / "example.py").write_text("value = 2\n")
    real_parent = tmp_path / "scored-parent"
    real_logs = real_parent / "logs"
    real_logs.mkdir(parents=True)
    scored_trajectory = real_logs / "trajectory.json"
    _trajectory(scored_trajectory)
    original_scored_trajectory = scored_trajectory.read_text()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    logs_dir = linked_parent / "logs"
    trajectory_path = logs_dir / "trajectory.json"

    assert not write_model_patch(repo, logs_dir, trajectory_path, baseline)
    assert scored_trajectory.read_text() == original_scored_trajectory
    assert not (real_logs / "artifacts/model.patch").exists()


def test_unique_temp_cleanup_failure_never_escapes_or_mutates_trajectory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _ = _repo(tmp_path)
    baseline = capture_model_patch_baseline(repo)
    assert baseline is not None
    assert baseline.state_dir is not None
    state_dir = Path(baseline.state_dir)
    (repo / "example.py").write_text("value = 2\n")
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    trajectory_path = logs_dir / "trajectory.json"
    _trajectory(trajectory_path)
    original_trajectory = trajectory_path.read_text()

    monkeypatch.setattr(
        model_patch_module.os,
        "replace",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            PermissionError("replace denied")
        ),
    )
    monkeypatch.setattr(
        model_patch_module.os,
        "unlink",
        lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("unlink denied")),
    )

    assert not write_model_patch(repo, logs_dir, trajectory_path, baseline)
    assert trajectory_path.read_text() == original_trajectory
    assert not (logs_dir / "artifacts/model.patch").exists()

    monkeypatch.undo()
    model_patch_module.cleanup_model_patch_baseline(baseline)
    assert not state_dir.exists()


def test_no_model_changes_cleans_isolated_state_without_writing_objects(
    tmp_path: Path,
) -> None:
    repo, _ = _repo(tmp_path)
    objects_before = _object_inventory(repo)
    baseline = capture_model_patch_baseline(repo)
    assert baseline is not None
    assert baseline.state_dir is not None
    state_dir = Path(baseline.state_dir)
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    trajectory_path = logs_dir / "trajectory.json"
    _trajectory(trajectory_path)
    original = trajectory_path.read_text()

    assert not write_model_patch(repo, logs_dir, trajectory_path, baseline)
    assert trajectory_path.read_text() == original
    assert _object_inventory(repo) == objects_before
    assert not state_dir.exists()


def test_failed_baseline_preflight_cleans_isolated_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _ = _repo(tmp_path)
    objects_before = _object_inventory(repo)
    (repo / "large-setup.txt").write_bytes(b"x" * (MAX_MODEL_PATCH_BYTES + 1))
    state_dir = Path(tempfile.mkdtemp(prefix="vals-model-patch-test-"))
    monkeypatch.setattr(
        model_patch_module.tempfile,
        "mkdtemp",
        lambda *, prefix: str(state_dir),
    )

    assert capture_model_patch_baseline(repo) is None
    assert _object_inventory(repo) == objects_before
    assert not state_dir.exists()


@pytest.mark.parametrize(
    "unsafe_header",
    [
        "diff --git a/../../secret b/../../secret\n",
        "diff --git a/safe b/safe\n--- ../../secret\n+++ b/safe\n",
        "diff --git a/safe b/safe\n--- a/safe\n+++ /absolute/path\n",
        "diff --git a/safe b/safe\nrename from ../secret\nrename to safe\n",
        'diff --git a/safe b/safe\nrename from "../secret"\nrename to safe\n',
        "diff --git a/safe b/safe\ncopy from safe\ncopy to /absolute/path\n",
    ],
)
def test_rejects_unsafe_patch_paths(unsafe_header: str) -> None:
    with pytest.raises(ValueError, match="unsafe model patch path"):
        _stats_and_paths(unsafe_header)
