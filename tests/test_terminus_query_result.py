import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from model_library.base.output import QueryResult, QueryResultMetadata

from terminus2.terminus_2 import Terminus2


class TerminusQueryResultTests(unittest.IsolatedAsyncioTestCase):
    async def test_reasoning_only_result_is_normalized_to_empty_text(self) -> None:
        agent = object.__new__(Terminus2)
        agent._api_request_times = []
        chat = SimpleNamespace(
            chat=AsyncMock(
                return_value=QueryResult(
                    output_text=None,
                    metadata=QueryResultMetadata(duration_seconds=0),
                )
            )
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            response_path = Path(temp_dir) / "response.txt"
            result = await agent._query_llm(
                chat=chat,
                prompt="continue",
                logging_paths=(None, None, response_path),
            )

            self.assertEqual(result.output_text, "")
            self.assertEqual(response_path.read_text(), "")


if __name__ == "__main__":
    unittest.main()
