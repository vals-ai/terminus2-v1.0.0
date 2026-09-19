import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from model_library.base.output import QueryResult, QueryResultMetadata
from model_library.exceptions import MaxContextWindowExceededError
from terminus2.agent.context import AgentContext
from terminus2.llms.chat import Chat
from terminus2.terminus_2 import Terminus2


@pytest.mark.parametrize(
    ("duration", "summary_fails", "reasoning_only"),
    [(1.25, False, False), (0.0, False, False), (None, False, False),
     (None, True, False), (0.0, False, True)],
)
def test_short_summary_is_linked_without_changing_main_call_accounting(
    tmp_path, duration, summary_fails, reasoning_only, monkeypatch
):
    # The only model boundary is synthetic; reject any accidental network access.
    def no_network(*args, **kwargs):
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr("socket.socket.connect", no_network)
    monkeypatch.setattr("socket.socket.connect_ex", no_network)
    summary = QueryResult(
        output_text=None if reasoning_only else "finish the build",
        reasoning="summary reasoning",
        metadata=QueryResultMetadata(
            in_tokens=11, out_tokens=5, reasoning_tokens=3, duration_seconds=duration,
        ),
    )
    continuation = QueryResult(
        output_text=json.dumps({
            "analysis": "continue the build", "plan": "inspect the files",
            "commands": [{"keystrokes": "ls\n", "duration": 0}],
            "task_complete": False,
        }),
        metadata=QueryResultMetadata(in_tokens=2, out_tokens=4, duration_seconds=0.25),
    )
    results = iter([
        MaxContextWindowExceededError(),
        RuntimeError("short summary unavailable") if summary_fails else summary,
        continuation,
    ])
    calls = []

    async def query(input, *, history=()):
        calls.append((input, history))
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result

    llm = SimpleNamespace(query=query, count_tokens=AsyncMock(return_value=0))
    agent = Terminus2(
        logs_dir=tmp_path, model_name="synthetic/offline", llm=llm,
        max_turns=1, proactive_summarization_threshold=0, session_id="test-session",
    )
    agent._context = AgentContext()
    agent._summarization_count = 2
    monkeypatch.setattr(agent, "truncator", SimpleNamespace(unwind=AsyncMock(), context_limit=10000))
    agent._summarize = AsyncMock(side_effect=RuntimeError("full summary unavailable"))
    screen = "$ make\nmissing target"
    monkeypatch.setattr(agent, "_session", SimpleNamespace(
        is_session_alive=AsyncMock(return_value=True),
        capture_pane=AsyncMock(return_value=screen),
        send_keys=AsyncMock(),
        get_incremental_output=AsyncMock(return_value="model-visible terminal output"),
    ))
    chat = Chat(llm, metrics_dir=tmp_path)

    asyncio.run(agent._run_agent_loop("continue", chat, tmp_path, "Build the project"))

    assert len(calls) == 3
    assert calls[1] == (
        (
            f"Briefly continue this task: Build the project\n\nCurrent state: {screen}"
            "\n\nNext steps (2-3 sentences):"
        ),
        (),
    )
    expected_prompt = (
        f"Build the project\n\nCurrent state: {screen}" if summary_fails
        else "Build the project\n\nSummary: None" if reasoning_only
        else "Build the project\n\nSummary: finish the build"
    )
    assert calls[2][0] == expected_prompt
    assert agent._api_request_times == [250]
    assert agent._context.n_input_tokens == 2
    assert agent._context.n_output_tokens == 4
    turns = (tmp_path / "metrics_per_turn.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in turns] == [continuation.metadata.model_dump(mode="json")]
    root = json.loads((tmp_path / "trajectory.json").read_text())
    assert root["schema_version"] == "ATIF-v1.7"
    assert root["final_metrics"]["total_prompt_tokens"] == 2
    assert root["final_metrics"]["total_completion_tokens"] == 4
    main_step = root["steps"][-1]
    assert main_step["observation"]["results"][0]["content"] == "model-visible terminal output"
    assert main_step["metrics"]["prompt_tokens"] == 2
    assert main_step["metrics"]["completion_tokens"] == 4
    children = root.get("subagent_trajectories", [])
    if summary_fails:
        assert children == []
        assert len(root["steps"]) == 1
        assert not list(tmp_path.glob("trajectory.summarization-*.json"))
        return

    assert len(children) == 1
    child = children[0]
    ref = root["steps"][0]["observation"]["results"][0]["subagent_trajectory_ref"][0]
    assert ref["session_id"] == child["session_id"]
    assert ref["trajectory_id"] == child["trajectory_id"] == "summarization-2-short-summary"
    assert child == json.loads((tmp_path / "trajectory.summarization-2-short-summary.json").read_text())
    assert child["schema_version"] == "ATIF-v1.7"
    assert child["steps"][0]["message"] == calls[1][0]
    step = child["steps"][1]
    assert step["message"] == ("" if reasoning_only else "finish the build")
    assert summary.output_text == (None if reasoning_only else "finish the build")
    assert step["reasoning_content"] == summary.reasoning
    expected_extra = {"reasoning_tokens": 3}
    if duration is not None:
        expected_extra["duration_seconds"] = duration
    assert step["metrics"] == {
        "prompt_tokens": 11, "completion_tokens": 8, "extra": expected_extra,
    }
    assert "history" not in step
    assert "response_id" not in json.dumps(child)
    assert child["final_metrics"]["total_prompt_tokens"] == 11
    assert child["final_metrics"]["total_completion_tokens"] == 8
    assert child["final_metrics"]["total_steps"] == 2
