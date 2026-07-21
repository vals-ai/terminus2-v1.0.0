from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from terminus2.model_patch import MAX_MODEL_PATCH_BYTES, write_model_patch


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


def test_writes_text_patch_and_atif_reference(tmp_path: Path) -> None:
    repo, base_commit = _repo(tmp_path)
    (repo / "example.py").write_text("value = 2\n")
    (repo / "new.py").write_text("new_value = 3\n")
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    trajectory_path = logs_dir / "trajectory.json"
    _trajectory(trajectory_path)

    assert write_model_patch(repo, logs_dir, trajectory_path, base_commit)

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


def test_omits_patch_when_diff_contains_secret(tmp_path: Path) -> None:
    repo, base_commit = _repo(tmp_path)
    (repo / "example.py").write_text('API_KEY = "sk-live-example-secret-value"\n')
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    trajectory_path = logs_dir / "trajectory.json"
    _trajectory(trajectory_path)
    original = trajectory_path.read_text()

    assert not write_model_patch(repo, logs_dir, trajectory_path, base_commit)
    assert not (logs_dir / "artifacts/model.patch").exists()
    assert trajectory_path.read_text() == original


def test_omits_binary_patch(tmp_path: Path) -> None:
    repo, base_commit = _repo(tmp_path)
    (repo / "asset.bin").write_bytes(b"\x00\x01\x02")
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    trajectory_path = logs_dir / "trajectory.json"
    _trajectory(trajectory_path)

    assert not write_model_patch(repo, logs_dir, trajectory_path, base_commit)
    assert not (logs_dir / "artifacts/model.patch").exists()


def test_excludes_agent_logs_when_logs_are_inside_worktree(tmp_path: Path) -> None:
    repo, base_commit = _repo(tmp_path)
    (repo / "example.py").write_text("value = 2\n")
    logs_dir = repo / "logs"
    logs_dir.mkdir()
    trajectory_path = logs_dir / "trajectory.json"
    _trajectory(trajectory_path)

    assert write_model_patch(repo, logs_dir, trajectory_path, base_commit)

    trajectory = json.loads(trajectory_path.read_text())
    assert trajectory["extra"]["vals"]["model_patch"]["file_count"] == 1
    assert b"logs/trajectory.json" not in (logs_dir / "artifacts/model.patch").read_bytes()


def test_omits_oversized_patch_without_writing_artifact(tmp_path: Path) -> None:
    repo, base_commit = _repo(tmp_path)
    (repo / "large.txt").write_bytes(b"x" * (MAX_MODEL_PATCH_BYTES + 1))
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    trajectory_path = logs_dir / "trajectory.json"
    _trajectory(trajectory_path)

    assert not write_model_patch(repo, logs_dir, trajectory_path, base_commit)
    assert not (logs_dir / "artifacts/model.patch").exists()
