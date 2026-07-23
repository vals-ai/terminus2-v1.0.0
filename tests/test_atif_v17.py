import json
import logging
from pathlib import Path

from model_library.base import QueryResultMetadata

from terminus2.agent.context import AgentContext
from terminus2.trajectories.agent import Agent
from terminus2.trajectories.final_metrics import FinalMetrics
from terminus2.trajectories.metrics import Metrics
from terminus2.trajectories.step import Step
from terminus2.trajectories.trajectory import Trajectory
from terminus2.terminus_2 import Terminus2, _model_routing


def test_trajectory_defaults_to_atif_v17() -> None:
    assert Trajectory.model_fields["schema_version"].default == "ATIF-v1.7"


def test_query_metadata_is_normalized_without_losing_extension_metrics() -> None:
    metadata = QueryResultMetadata(
        in_tokens=11,
        out_tokens=7,
        reasoning_tokens=3,
        cache_read_tokens=5,
        cache_write_tokens=2,
    )

    metrics = Metrics.from_query_result_metadata(metadata)

    assert metrics.model_dump(exclude_none=True) == {
        "prompt_tokens": 18,
        "completion_tokens": 10,
        "cached_tokens": 5,
        "extra": {
            "reasoning_tokens": 3,
            "cache_write_tokens": 2,
        },
    }


def test_final_metrics_are_normalized_without_losing_extension_metrics() -> None:
    metrics = FinalMetrics.from_token_counts(
        input_tokens=11,
        output_tokens=7,
        reasoning_tokens=3,
        cache_read_tokens=5,
        cache_write_tokens=2,
        cost_usd=1.25,
        total_steps=4,
    )

    assert metrics.model_dump(exclude_none=True) == {
        "total_prompt_tokens": 18,
        "total_completion_tokens": 10,
        "total_cached_tokens": 5,
        "total_cost_usd": 1.25,
        "total_steps": 4,
        "extra": {
            "total_reasoning_tokens": 3,
            "total_cache_write_tokens": 2,
        },
    }


def test_final_metrics_do_not_invent_missing_optional_token_counts() -> None:
    metrics = FinalMetrics.from_token_counts(
        input_tokens=11,
        output_tokens=7,
        reasoning_tokens=None,
        cache_read_tokens=None,
        cache_write_tokens=None,
        cost_usd=None,
        total_steps=2,
    )

    assert metrics.model_dump(exclude_none=True) == {
        "total_prompt_tokens": 11,
        "total_completion_tokens": 7,
        "total_steps": 2,
    }


def test_legacy_step_metric_inputs_are_normalized_for_compatibility() -> None:
    metrics = Metrics(
        prompt_tokens=11,
        completion_tokens=7,
        reasoning_tokens=3,
        cache_read_tokens=5,
        cache_write_tokens=2,
    )

    assert metrics.model_dump(exclude_none=True) == {
        "prompt_tokens": 18,
        "completion_tokens": 10,
        "cached_tokens": 5,
        "extra": {"reasoning_tokens": 3, "cache_write_tokens": 2},
    }


def test_legacy_final_metric_inputs_are_normalized_for_compatibility() -> None:
    metrics = FinalMetrics(
        total_prompt_tokens=11,
        total_completion_tokens=7,
        total_reasoning_tokens=3,
        total_cache_read_tokens=5,
        total_cache_write_tokens=2,
        total_steps=4,
    )

    assert metrics.model_dump(exclude_none=True) == {
        "total_prompt_tokens": 18,
        "total_completion_tokens": 10,
        "total_cached_tokens": 5,
        "total_steps": 4,
        "extra": {"total_reasoning_tokens": 3, "total_cache_write_tokens": 2},
    }


def test_agent_context_preserves_unknown_optional_aggregate_counts() -> None:
    context = AgentContext()
    context.accumulate(QueryResultMetadata(in_tokens=11, out_tokens=7))

    metrics = FinalMetrics.from_token_counts(
        input_tokens=context.n_input_tokens,
        output_tokens=context.n_output_tokens,
        cost_usd=context.optional_cost_total(),
        total_steps=2,
        **context.optional_token_totals(),
    )

    assert metrics.model_dump(exclude_none=True) == {
        "total_prompt_tokens": 11,
        "total_completion_tokens": 7,
        "total_steps": 2,
    }


def test_model_routing_records_confirmed_fallback() -> None:
    metadata = QueryResultMetadata(
        in_tokens=1,
        out_tokens=1,
        extra={
            "anthropic_response_model": "claude-opus-4-8",
            "fallback": True,
        },
    )

    assert _model_routing(metadata, "anthropic/claude-sonnet-4-6") == (
        "anthropic/claude-opus-4-8",
        {
            "vals": {
                "model_routing": {
                    "requested_model": "anthropic/claude-sonnet-4-6",
                    "resolved_model": "anthropic/claude-opus-4-8",
                    "fallback_used": True,
                }
            }
        },
    )


def _terminus_for_trajectory_test(tmp_path: Path) -> Terminus2:
    terminus = Terminus2.__new__(Terminus2)
    terminus.logs_dir = tmp_path
    terminus._logger = logging.getLogger(__name__)
    terminus._model_name = "anthropic/claude-sonnet-4-6"
    terminus._parser_name = "json"
    terminus._temperature = None
    terminus._session_id = "run-1-cont-1"
    terminus._summarization_count = 1
    terminus._linear_history = True
    terminus._subagent_trajectories = []
    return terminus


def test_subagent_reference_resolves_to_embedded_trajectory(tmp_path: Path) -> None:
    terminus = _terminus_for_trajectory_test(tmp_path)
    metadata = QueryResultMetadata(in_tokens=2, out_tokens=1)
    steps = [Step(step_id=1, source="user", message="Summarize")]

    reference = terminus._save_subagent_trajectory(
        session_id="run-1-summary",
        agent_name="terminus-2-summarization-summary",
        steps=steps,
        result_metadata=metadata,
        filename_suffix="summary",
        summary_text="Summary generation",
    )

    assert reference.trajectory_id == "summarization-1-summary"
    assert reference.trajectory_path is None
    assert terminus._subagent_trajectories[0].trajectory_id == reference.trajectory_id


def test_final_linear_trajectory_is_also_written_as_self_contained_root(
    tmp_path: Path,
) -> None:
    terminus = _terminus_for_trajectory_test(tmp_path)
    context = AgentContext()
    context.accumulate(QueryResultMetadata(in_tokens=2, out_tokens=1))
    terminus._context = context
    terminus._trajectory_steps = [Step(step_id=1, source="user", message="Continue")]
    terminus._subagent_trajectories = [
        Trajectory(
            trajectory_id="summarization-1-summary",
            session_id="run-1-summary",
            agent=Agent(name="summary", version="1", model_name=terminus._model_name),
            steps=[Step(step_id=1, source="user", message="Summarize")],
        )
    ]

    terminus._dump_trajectory()

    assert (tmp_path / "trajectory.cont-1.json").is_file()
    canonical = json.loads((tmp_path / "trajectory.json").read_text())
    assert "continued_trajectory_ref" not in canonical
    assert (
        canonical["subagent_trajectories"][0]["trajectory_id"]
        == "summarization-1-summary"
    )
