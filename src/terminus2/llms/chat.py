import json
import time
from functools import reduce
from pathlib import Path

from model_library.base import LLM, QueryResultMetadata
from model_library.base.input import InputItem, TextInput
from model_library.base.output import QueryResult


def _is_sendable(message: object) -> bool:
    """Whether a history entry can be sent back to a provider.

    A reasoning model can return no content at all, having spent its whole
    output budget thinking. The assistant turn that describes it then carries
    neither content nor tool calls, and providers reject a request containing
    one -- DeepSeek with "Invalid assistant message: content or tool_calls must
    be set" -- so the run ends on the following turn rather than continuing.
    """

    if isinstance(message, dict):
        if message.get("role") != "assistant":
            return True
        return bool(message.get("content")) or bool(message.get("tool_calls"))

    if getattr(message, "role", None) != "assistant":
        return True
    return bool(getattr(message, "content", None)) or bool(getattr(message, "tool_calls", None))


def _without_unsendable_turns(history: list) -> list:
    """Drop assistant turns a provider will not accept back."""

    return [message for message in history if _is_sendable(message)]


class Chat:
    """Manages conversation history and LLM interactions."""

    def __init__(self, model: LLM, metrics_dir: Path | None = None):
        self._model = model
        self._messages: list[InputItem] = []
        self._metadata: list[QueryResultMetadata] = []
        self._metrics_dir = metrics_dir
        self._start_time = time.time()

    @property
    def model(self) -> LLM:
        return self._model

    @property
    def messages(self) -> list:
        return self._messages

    # unwinds the last user message and all responses after it
    # returns a boolean indicating whether a message was found
    # exhausts the list and return False if no user message found
    def unwind_last_user_message(self) -> bool:
        try:
            while not isinstance(self._messages.pop(), TextInput):
                continue
            return True
        except IndexError:
            return False

    async def chat(
        self,
        prompt: str,
        logging_path: Path | None = None,
        **kwargs,
    ) -> QueryResult:
        query_result: QueryResult = await self._model.query(
            input=prompt,
            history=self.messages,
            **kwargs,
        )

        # save message history
        self._messages = _without_unsendable_turns(query_result.history)

        # add the query metadata
        self._metadata.append(query_result.metadata)

        if self._metrics_dir is not None:
            self._write_metrics(query_result.metadata)

        return query_result

    def _write_metrics(self, turn_metadata: QueryResultMetadata) -> None:
        assert self._metrics_dir is not None
        # append per-turn metrics
        with open(self._metrics_dir / "metrics_per_turn.jsonl", "a") as f:
            f.write(turn_metadata.model_dump_json() + "\n")

        # write aggregated total metrics
        total = reduce(lambda a, b: a + b, self._metadata)
        total_dict = total.model_dump()
        total_dict["wall_clock_duration"] = round(time.time() - self._start_time, 3)
        with open(self._metrics_dir / "metrics_total.json", "w") as f:
            json.dump(total_dict, f, indent=2)
