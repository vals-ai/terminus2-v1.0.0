from model_library.base import QueryResultMetadata

from terminus2.agent.context import AgentContext
from terminus2.trajectories.final_metrics import FinalMetrics
from terminus2.trajectories.metrics import Metrics
from terminus2.trajectories.trajectory import Trajectory


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
