import asyncio
import hashlib
import json
import logging
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from model_library.base.output import QueryResult, QueryResultMetadata

import terminus2.terminus_2 as terminus_module
from terminus2.agent.context import AgentContext
from terminus2.environment_base import BaseEnvironment
from terminus2.terminus_2 import Command, Terminus2
from terminus2.tmux_session import TmuxSession


# Both byte-limit boundaries split a multibyte character. The middle must survive
# natively even though neither model-visible reduction retains it.
SNAPSHOT = "A" * 4999 + "€MIDDLE_SENTINEL" + "B" * 10000 + "é" + "Z" * 4999
ONCE_LIMITED = (
    "A" * 4999
    + "\n[... output limited to 10000 bytes; 10020 interior bytes omitted ...]\n"
    + "Z" * 4999
)
TWICE_LIMITED = (
    "A" * 4999
    + "\n\n[... output limited to 10000 bytes; 69 interior bytes omitted ...]\n\n"
    + "Z" * 4999
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr("socket.socket.connect", blocked)
    monkeypatch.setattr("socket.socket.connect_ex", blocked)


def _agent(tmp_path, responses=(), **kwargs):
    llm = SimpleNamespace(
        query=AsyncMock(side_effect=responses),
        count_tokens=AsyncMock(return_value=0),
        input_context_window=100000,
    )
    agent = Terminus2(
        logs_dir=tmp_path,
        model_name="synthetic/offline",
        llm=llm,
        session_id="original-session",
        enable_summarize=False,
        **kwargs,
    )
    agent._logger = logging.getLogger("test-native-capture")
    session = MagicMock(spec=TmuxSession)
    session.is_session_alive = AsyncMock(return_value=True)
    session.send_keys = AsyncMock()
    session.get_incremental_output = AsyncMock(return_value=SNAPSHOT)
    agent._session = session
    return agent, session, llm


def _response(commands=(), complete=False):
    return QueryResult(
        output_text=json.dumps(
            {
                "analysis": "inspect",
                "plan": "continue",
                "commands": list(commands),
                "task_complete": complete,
            }
        ),
        metadata=QueryResultMetadata(in_tokens=2, out_tokens=3, duration_seconds=0.25),
    )


def _records(tmp_path):
    return [
        json.loads(path.read_text())
        for path in sorted((tmp_path / "terminal-observations").glob("*.json"))
    ]


@pytest.mark.parametrize("raw_content", [False, True])
def test_run_retains_snapshots_and_preserves_double_limit_and_completion(
    tmp_path,
    raw_content,
):
    commands = [
        {"keystrokes": "first\n", "duration": 0},
        {"keystrokes": "second\n", "duration": 0},
    ]
    agent, session, llm = _agent(
        tmp_path,
        [
            _response(commands),
            _response(),
            _response(complete=True),
            _response(complete=True),
        ],
        max_turns=4,
        trajectory_config={"raw_content": raw_content},
    )
    context = AgentContext()
    asyncio.run(
        agent.run("synthetic instruction", MagicMock(spec=BaseEnvironment), context)
    )

    prompts = [c.kwargs["input"] for c in llm.query.await_args_list]
    assert prompts[:3] == [
        agent._prompt_template.format(
            instruction="synthetic instruction", terminal_state=ONCE_LIMITED
        ),
        TWICE_LIMITED,
        TWICE_LIMITED,
    ]
    assert prompts[3] == agent._get_completion_confirmation_message(ONCE_LIMITED)
    assert session.get_incremental_output.await_count == 5
    assert session.send_keys.await_args_list == [
        call("first\n", block=False, min_timeout_sec=0),
        call("second\n", block=False, min_timeout_sec=0),
    ]
    assert context.n_input_tokens == 8
    assert context.n_output_tokens == 12
    assert context.metadata is not None
    assert context.metadata["n_episodes"] == 4
    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    observations = [
        step["observation"]["results"][0] for step in trajectory["steps"][1:]
    ]
    assert [o["content"] for o in observations] == [
        TWICE_LIMITED,
        TWICE_LIMITED,
        prompts[3],
        ONCE_LIMITED,
    ]
    assert observations[0].get("source_call_id") is None
    records = _records(tmp_path)
    assert [r["sequence"] for r in records] == list(range(5))
    assert [r["episode"] for r in records] == [None, 0, 1, 2, 3]
    assert [r["phase"] for r in records] == ["initial"] + ["command_batch"] * 4
    assert [r["tool_call_ids"] for r in records] == [
        [],
        [] if raw_content else ["call_0_1", "call_0_2"],
        [],
        [],
        [],
    ]
    assert all(r["output"] == SNAPSHOT for r in records)
    assert all(r["capture_session_id"] == "original-session" for r in records)


def test_timeout_links_only_attempted_commands_and_keeps_original_result(tmp_path):
    agent, session, _ = _agent(tmp_path)
    session.send_keys.side_effect = [None, TimeoutError("synthetic timeout")]
    commands = [Command("first\n", 1), Command("second\n", 2), Command("not-sent\n", 3)]
    result = asyncio.run(agent._execute_commands(commands, session, episode=7))

    assert result == (
        True,
        agent._timeout_template.format(
            timeout_sec=2,
            command="second\n",
            terminal_state=ONCE_LIMITED,
        ),
    )
    assert session.send_keys.await_count == 2
    assert session.get_incremental_output.await_count == 1
    (record,) = _records(tmp_path)
    assert record["phase"] == "command_timeout"
    assert record["episode"] == 7
    assert record["tool_call_ids"] == ["call_7_1", "call_7_2"]
    assert record["output"] == SNAPSHOT


def test_completed_archive_records_remain_ordered_across_continuation(
    tmp_path,
    monkeypatch,
):
    logs = tmp_path / "logs" / "terminus2"
    logs.mkdir(parents=True)
    agent, session, llm = _agent(
        logs, max_turns=0, trajectory_config={"linear_history": True}
    )
    replace = Path.replace
    published = []

    def check_publish(path, target):
        assert not target.exists()
        record = json.loads(path.read_text())
        assert record["output"] == SNAPSHOT
        assert record["path"] == target.relative_to(logs).as_posix()
        published.append(record["path"])
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", check_publish)
    asyncio.run(
        agent.run(
            "synthetic instruction", MagicMock(spec=BaseEnvironment), AgentContext()
        )
    )
    agent._summarization_count = 1
    agent._split_trajectory_on_summarization("synthetic handoff")
    asyncio.run(agent._execute_commands([Command("next\n", 0)], session, episode=8))

    records = _records(logs)
    assert [r["sequence"] for r in records] == [0, 1]
    assert [r["session_id"] for r in records] == [
        "original-session",
        "original-session-cont-1",
    ]
    assert [r["capture_session_id"] for r in records] == ["original-session"] * 2
    assert records[1]["tool_call_ids"] == ["call_8_1"]
    assert llm.query.await_count == 0
    assert not list((logs / "terminal-observations").glob("*.tmp"))

    archive_path = tmp_path / "agent_output.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(logs, arcname="logs/terminus2")
    with tarfile.open(archive_path, "r:gz") as archive:
        for relative_path in published:
            entry = archive.extractfile("logs/terminus2/" + relative_path)
            assert entry is not None
            with entry:
                record = json.load(entry)
            data = record["output"].encode("utf-8")
            assert data == SNAPSHOT.encode("utf-8")
            assert record["byte_count"] == len(data)
            assert record["sha256"] == hashlib.sha256(data).hexdigest()
    sizes = [(logs / path).stat().st_size for path in published]
    print(
        json.dumps(
            {
                "native_records": len(sizes),
                "total_record_bytes": sum(sizes),
                "largest_record_bytes": max(sizes),
            }
        )
    )


@pytest.mark.parametrize("failure", ["mkdir", "open", "write", "publish"])
@pytest.mark.parametrize("timeout", [False, True])
def test_capture_io_failure_is_nonfatal_has_no_success_record_and_leaks_no_text(
    tmp_path,
    monkeypatch,
    caplog,
    failure,
    timeout,
):
    agent, session, _ = _agent(tmp_path)
    if timeout:
        session.send_keys.side_effect = TimeoutError("synthetic timeout")
    error = OSError("PRIVATE_ERROR_TEXT")

    def fail_io(*args, **kwargs):
        raise error

    with monkeypatch.context() as patch:
        if failure == "mkdir":
            patch.setattr(Path, "mkdir", fail_io)
        elif failure == "open":
            patch.setattr(terminus_module, "NamedTemporaryFile", fail_io)
        elif failure == "write":

            def fail_write(payload, stream, **kwargs):
                stream.write("PRIVATE_PARTIAL_OUTPUT")
                raise error

            patch.setattr(terminus_module.json, "dump", fail_write)
        else:
            patch.setattr(Path, "replace", fail_io)

        result = asyncio.run(
            agent._execute_commands([Command("first\n", 0)], session, episode=0)
        )

    expected_output = (
        agent._timeout_template.format(
            timeout_sec=0,
            command="first\n",
            terminal_state=ONCE_LIMITED,
        )
        if timeout
        else ONCE_LIMITED
    )
    assert result == (timeout, expected_output)
    assert _records(tmp_path) == []
    assert not list((tmp_path / "terminal-observations").glob("*.tmp"))
    assert "capture incomplete" in caplog.text
    assert "PRIVATE" not in caplog.text
    assert "MIDDLE_SENTINEL" not in caplog.text
    asyncio.run(agent._execute_commands([], session, episode=1))
    (record,) = _records(tmp_path)
    assert record["sequence"] == 1
    assert record["episode"] == 1
    assert record["output"] == SNAPSHOT


@pytest.mark.parametrize("output", ["", "ready\n", "é" * 5000])
def test_unclipped_snapshots_keep_exact_empty_small_and_boundary_output(
    tmp_path, output
):
    agent, session, _ = _agent(tmp_path)
    session.get_incremental_output.return_value = output
    assert asyncio.run(agent._execute_commands([], session, episode=0)) == (
        False,
        output,
    )
    (record,) = _records(tmp_path)
    assert record["output"] == output
    assert record["byte_count"] == len(output.encode("utf-8"))
    assert record["sha256"] == hashlib.sha256(output.encode("utf-8")).hexdigest()


def test_original_command_failure_is_not_replaced_by_capture(tmp_path):
    agent, session, _ = _agent(tmp_path)
    error = RuntimeError("synthetic transport failure")
    session.send_keys.side_effect = error
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(
            agent._execute_commands([Command("first\n", 0)], session, episode=0)
        )
    assert caught.value is error
    session.get_incremental_output.assert_not_awaited()
    assert _records(tmp_path) == []


def test_relative_logs_directory_keeps_archive_relative_snapshot_paths(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    logs = Path("logs")
    agent, session, _ = _agent(logs)
    asyncio.run(agent._execute_commands([], session, episode=0))
    (record,) = _records(logs)
    assert record["output"] == SNAPSHOT
    assert not Path(record["path"]).is_absolute()
    assert json.loads((logs / record["path"]).read_text()) == record
