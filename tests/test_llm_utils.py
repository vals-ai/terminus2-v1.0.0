import unittest
from types import SimpleNamespace
from typing import cast

from model_library.base import LLM

from terminus2.llms.utils import get_model_context_limit


class ModelContextLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_reads_eagerly_loaded_context_window(self) -> None:
        llm = cast(
            LLM,
            cast(
                object,
                SimpleNamespace(
                    input_context_window=128_000,
                    model_name="test-model",
                ),
            ),
        )

        context_limit = await get_model_context_limit(llm)

        self.assertEqual(context_limit, 128_000)


if __name__ == "__main__":
    unittest.main()
