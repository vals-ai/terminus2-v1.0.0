import json
import time
from functools import reduce
from pathlib import Path

from model_library.base import LLM, QueryResultMetadata
from model_library.base.input import InputItem, TextInput
from model_library.base.output import QueryResult


def _field(payload: object, name: str) -> object:
    """Read a field from a provider message, which may be a model or a dict."""

    if isinstance(payload, dict):
        return payload.get(name)
    return getattr(payload, name, None)


_TEXT_BLOCK_TYPES = frozenset({"text", "output_text"})


def _blocks(content: object) -> list | tuple:
    """The content blocks of a message, or nothing if it is not block-shaped."""

    return content if isinstance(content, (list, tuple)) else ()


def _has_blank_text_block(content: object) -> bool:
    """Whether any text block is empty.

    Anthropic rejects the block itself -- "messages: text content blocks must
    be non-empty" -- so one blank block spoils the turn however much else it
    carries. The turn is dropped whole rather than having the block stripped:
    model-library echoes `RawResponse.response` back to the provider verbatim
    because it is an opaque signed blob, so editing inside it is not an option.
    """

    for block in _blocks(content):
        if _field(block, "type") in _TEXT_BLOCK_TYPES:
            text = _field(block, "text")
            if not (isinstance(text, str) and text.strip()):
                return True
    return False


def _carries_content(content: object) -> bool:
    """Whether a provider would find anything in this content.

    Whitespace counts as nothing, so a turn whose content is `"   "` goes the
    same way as one whose content is `""`.
    """

    if isinstance(content, str):
        return bool(content.strip())
    return bool(content)


def _is_sendable(item: object) -> bool:
    """Whether a history entry can be sent back to a provider.

    A reasoning model can return no content at all, having spent its whole
    output budget thinking. model-library stores the turn as
    `RawResponse(response=<provider message>)`, and that message then carries
    neither content nor tool calls. Providers reject a request containing one --
    DeepSeek with "Invalid assistant message: content or tool_calls must be
    set" -- so the run ends on the following turn.

    The same turn reaches Anthropic as a content list holding an empty text
    block, which is a non-empty list and so looks like content to a plain
    truth test. That is judged per block instead.

    Only that wrapper is inspected. Prompts, system messages and tool results
    are other `InputItem` kinds and are always kept.
    """

    payload = _field(item, "response")
    if payload is None:
        return True

    # The responses API stores a list of output items rather than one message;
    # judging those is not attempted.
    if isinstance(payload, (list, tuple)):
        return True

    if _field(payload, "role") != "assistant":
        return True

    content = _field(payload, "content")
    if _has_blank_text_block(content):
        return False

    return _carries_content(content) or bool(_field(payload, "tool_calls"))


def _without_unsendable_turns(history: list[InputItem]) -> list[InputItem]:
    """Drop assistant turns a provider will not accept back."""

    return [item for item in history if _is_sendable(item)]


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
