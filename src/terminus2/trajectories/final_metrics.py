"""Final metrics model for ATIF trajectories."""

from typing import Any

from pydantic import BaseModel, Field, model_validator


class FinalMetrics(BaseModel):
    """Aggregate statistics for the entire trajectory.

    Token fields align with QueryResultMetadata from model-library.
    """

    total_prompt_tokens: int | None = Field(
        default=None,
        description="Sum of all prompt tokens across all steps, including cached and cache-write tokens",
    )
    total_completion_tokens: int | None = Field(
        default=None,
        description="Sum of all completion tokens across all steps, including reasoning tokens",
    )
    total_cached_tokens: int | None = Field(
        default=None,
        description="Sum of cached-token subsets across all steps",
    )
    total_reasoning_tokens: int | None = Field(default=None, exclude=True, repr=False)
    total_cache_read_tokens: int | None = Field(default=None, exclude=True, repr=False)
    total_cache_write_tokens: int | None = Field(default=None, exclude=True, repr=False)
    total_cost_usd: float | None = Field(
        default=None,
        description="Total real monetary cost for the entire trajectory, including cost for subagents, if any",
    )
    total_steps: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Total number of steps. If not equivalent to the number of steps in the "
            "trajectory, must be documented in the root-level notes field."
        ),
    )
    extra: dict[str, Any] | None = Field(
        default=None,
        description="Custom aggregate metrics",
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def normalize_legacy_token_fields(self) -> "FinalMetrics":
        """Accept the pre-v1.6 constructor fields while emitting v1.6 metrics."""
        if (
            self.total_reasoning_tokens is None
            and self.total_cache_read_tokens is None
            and self.total_cache_write_tokens is None
        ):
            return self

        if self.total_prompt_tokens is not None:
            self.total_prompt_tokens += (self.total_cache_read_tokens or 0) + (self.total_cache_write_tokens or 0)
        if self.total_completion_tokens is not None:
            self.total_completion_tokens += self.total_reasoning_tokens or 0
        if self.total_cached_tokens is None:
            self.total_cached_tokens = self.total_cache_read_tokens

        extra = dict(self.extra or {})
        if self.total_reasoning_tokens is not None:
            extra.setdefault("total_reasoning_tokens", self.total_reasoning_tokens)
        if self.total_cache_write_tokens is not None:
            extra.setdefault("total_cache_write_tokens", self.total_cache_write_tokens)
        self.extra = extra or None
        return self

    @classmethod
    def from_token_counts(
        cls,
        *,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int | None,
        cache_read_tokens: int | None,
        cache_write_tokens: int | None,
        cost_usd: float | None,
        total_steps: int | None = None,
    ) -> "FinalMetrics":
        """Normalize tracked token counters to ATIF v1.6 without dropping extensions."""
        extra = {
            key: value
            for key, value in {
                "total_reasoning_tokens": reasoning_tokens,
                "total_cache_write_tokens": cache_write_tokens,
            }.items()
            if value is not None
        }
        return cls(
            total_prompt_tokens=input_tokens + (cache_read_tokens or 0) + (cache_write_tokens or 0),
            total_completion_tokens=output_tokens + (reasoning_tokens or 0),
            total_cached_tokens=cache_read_tokens,
            total_cost_usd=cost_usd,
            total_steps=total_steps,
            extra=extra or None,
        )
