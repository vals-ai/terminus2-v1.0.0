from typing import Any

from model_library.base.output import QueryResultMetadata
from pydantic import BaseModel, Field, PrivateAttr

from terminus2.agent.rollout_detail import RolloutDetail


class AgentContext(BaseModel):
    """Token usage and cost metrics for an agent execution.

    Token fields align with QueryResultMetadata from model-library.
    Accumulates metadata from LLM calls via the accumulate() method.
    """

    n_input_tokens: int = Field(
        default=0,
        description="The number of input tokens used (excluding cache read tokens).",
    )
    n_output_tokens: int = Field(
        default=0,
        description="The number of output tokens used (excluding reasoning tokens).",
    )
    n_reasoning_tokens: int = Field(default=0, description="The number of reasoning/thinking tokens used.")
    n_cache_read_tokens: int = Field(default=0, description="The number of cache read tokens used.")
    n_cache_write_tokens: int = Field(default=0, description="The number of cache write tokens used.")
    cost_usd: float = Field(default=0.0, description="The cost in USD for the agent execution.")
    latency_seconds: float = Field(
        default=0.0,
        description="The total latency in seconds accumulated from LLM queries.",
    )
    rollout_details: list[RolloutDetail] | None = Field(
        default=None,
        description=(
            "Detailed information about each rollout trajectory including token IDs, "
            "loss masks, and logprobs. Each element represents one trajectory. For a "
            "linear chat history, there is only one rollout trajectory."
        ),
    )
    metadata: dict[str, Any] | None = Field(default=None, description="Additional metadata about the agent execution.")
    _llm_call_count: int = PrivateAttr(default=0)
    _reasoning_tokens_complete: bool = PrivateAttr(default=True)
    _cache_read_tokens_complete: bool = PrivateAttr(default=True)
    _cache_write_tokens_complete: bool = PrivateAttr(default=True)
    _cost_complete: bool = PrivateAttr(default=True)

    def accumulate(self, metadata: QueryResultMetadata) -> None:
        """Accumulate token usage and cost from an LLM call."""
        self.n_input_tokens += metadata.in_tokens
        self.n_output_tokens += metadata.out_tokens
        self._llm_call_count += 1
        self._reasoning_tokens_complete = self._reasoning_tokens_complete and metadata.reasoning_tokens is not None
        self._cache_read_tokens_complete = self._cache_read_tokens_complete and metadata.cache_read_tokens is not None
        self._cache_write_tokens_complete = self._cache_write_tokens_complete and metadata.cache_write_tokens is not None
        self._cost_complete = self._cost_complete and metadata.cost is not None
        self.n_reasoning_tokens += metadata.reasoning_tokens or 0
        self.n_cache_read_tokens += metadata.cache_read_tokens or 0
        self.n_cache_write_tokens += metadata.cache_write_tokens or 0
        if metadata.cost:
            self.cost_usd += metadata.cost.total
        if metadata.duration_seconds:
            self.latency_seconds += metadata.duration_seconds

    def optional_token_totals(self) -> dict[str, int | None]:
        """Return totals only when every model call reported the optional counter."""
        has_calls = self._llm_call_count > 0
        return {
            "reasoning_tokens": self.n_reasoning_tokens if has_calls and self._reasoning_tokens_complete else None,
            "cache_read_tokens": self.n_cache_read_tokens if has_calls and self._cache_read_tokens_complete else None,
            "cache_write_tokens": self.n_cache_write_tokens if has_calls and self._cache_write_tokens_complete else None,
        }

    def optional_cost_total(self) -> float | None:
        """Return aggregate cost only when every model call reported it."""
        return self.cost_usd if self._llm_call_count > 0 and self._cost_complete else None

    def is_empty(self) -> bool:
        return all(value is None for value in self.model_dump().values())
