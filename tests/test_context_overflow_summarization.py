"""Context overflow must fall through to the short summary, not "Technical difficulties".

`_query_llm` catches `MaxContextWindowExceededError`, unwinds the chat and asks
the model for a summary. When the full summary fails, the short summary calls
the LLM directly, so it must use the real `LLM.query` interface; a wrong keyword
raises `TypeError` inside the `except Exception` fallback chain, which then
silently degrades to a canned "Technical difficulties" reply on every turn.
"""

import logging
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from model_library.base.output import QueryResult, QueryResultMetadata
from model_library.exceptions import MaxContextWindowExceededError

from terminus2.terminus_2 import Terminus2


class FakeLLM:
    """Mirrors the `LLM.query(input, *, history=...)` signature."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def query(
        self, input: str, *, history: tuple[object, ...] = (), **kwargs: object
    ) -> QueryResult:
        self.calls.append(input)
        return _result("finish the build")


def _result(text: str) -> QueryResult:
    return QueryResult(
        output_text=text, metadata=QueryResultMetadata(duration_seconds=0)
    )


class ShortSummaryFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_short_summary_queries_the_llm_and_continues_the_chat(self) -> None:
        agent = object.__new__(Terminus2)
        agent._api_request_times = []
        agent._logger = logging.getLogger("test")
        agent._enable_summarize = True
        agent._llm = FakeLLM()
        agent.truncator = SimpleNamespace(unwind=AsyncMock())
        agent._summarize = AsyncMock(
            side_effect=RuntimeError("full summary unavailable")
        )

        chat = SimpleNamespace(
            chat=AsyncMock(
                side_effect=[MaxContextWindowExceededError(), _result("ls -la")]
            )
        )
        session = SimpleNamespace(
            capture_pane=AsyncMock(return_value="$ make\nerror: missing target")
        )

        result = await agent._query_llm(
            chat=chat,
            prompt="continue",
            logging_paths=(None, None, None),
            original_instruction="Build the project",
            session=session,
        )

        self.assertEqual(result.output_text, "ls -la")
        self.assertEqual(len(agent._llm.calls), 1)
        self.assertIn("Build the project", agent._llm.calls[0])
        summary_prompt = chat.chat.await_args_list[1].args[0]
        self.assertEqual(
            summary_prompt, "Build the project\n\nSummary: finish the build"
        )


if __name__ == "__main__":
    unittest.main()
