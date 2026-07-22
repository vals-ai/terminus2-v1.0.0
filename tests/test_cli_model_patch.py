from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from terminus2.cli import _run_agent
from terminus2.model_patch import ModelPatchBaseline


def test_run_captures_pre_model_baseline_and_excludes_private_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    private_problem = repo / "private-task.md"
    private_problem.write_text("hidden task\n")
    logs_dir = repo / "private-logs"
    events: list[str] = []
    baseline = ModelPatchBaseline("a" * 40, "b" * 40)

    session = SimpleNamespace(send_keys=AsyncMock())
    agent = SimpleNamespace(
        _session=session,
        setup=AsyncMock(),
        run=AsyncMock(side_effect=lambda *args: events.append("run")),
    )
    environment = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
    capture = Mock(
        side_effect=lambda *args, **kwargs: (events.append("capture"), baseline)[1]
    )
    write = Mock(side_effect=lambda *args, **kwargs: events.append("write"))

    monkeypatch.setattr("terminus2.terminus_2.Terminus2", Mock(return_value=agent))
    monkeypatch.setattr(
        "terminus2.environment_local.LocalEnvironment", Mock(return_value=environment)
    )
    monkeypatch.setattr("terminus2.model_patch.capture_model_patch_baseline", capture)
    monkeypatch.setattr("terminus2.model_patch.write_model_patch", write)
    monkeypatch.setattr(os, "getcwd", lambda: str(repo))

    args = SimpleNamespace(
        logs_dir=logs_dir,
        raw_trajectory=False,
        linear_history=False,
        model="test/model",
        parser="json",
        max_turns=1,
        temperature=None,
        max_tokens=None,
        reasoning=None,
        reasoning_effort=None,
        api_base=None,
        no_summarize=True,
        instruction="do the task",
        problem_path=private_problem,
    )

    asyncio.run(_run_agent(args))

    assert events == ["capture", "run", "write"]
    capture.assert_called_once_with(
        repo,
        excluded_paths=(logs_dir.resolve(), private_problem.resolve()),
    )
    write.assert_called_once_with(
        repo, logs_dir.resolve(), logs_dir.resolve() / "trajectory.json", baseline
    )
