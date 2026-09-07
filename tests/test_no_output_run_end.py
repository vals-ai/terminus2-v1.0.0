"""A model that returns nothing must end the run, not abort the process.

Matching on `ModelNoOutputError` alone catches nothing in practice: it is an
`ImmediateRetryException`, so model-library rewraps it once its own retries are
spent, and through the gateway only `MaxContextWindowExceededError` keeps its
type. These tests pin all three transports so the handler cannot silently stop
matching the one that is actually in use.
"""

import logging
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from model_library.exceptions import (
    ContentFilterError,
    GatewayProviderError,
    ImmediateRetryExhaustedError,
    ModelNoOutputError,
)

from terminus2.terminus_2 import Terminus2, _no_output_error_name
from terminus2.trajectories.subagent_trajectory_ref import SubagentTrajectoryRef


def gateway_error(exception_type: str) -> GatewayProviderError:
    return GatewayProviderError(
        error_type="provider_error",
        code=None,
        message="no output",
        provider="test",
        raw_error={},
        exception_type=exception_type,
    )


class DiscriminatorTests(unittest.TestCase):
    def test_direct_error_is_recognised(self) -> None:
        self.assertEqual(_no_output_error_name(ModelNoOutputError()), "ModelNoOutputError")

    def test_rewrapped_by_the_immediate_retrier_is_recognised(self) -> None:
        wrapped = ImmediateRetryExhaustedError(10, 10, ModelNoOutputError())
        # The retrier's own type is unrelated to the cause it carries.
        self.assertNotIsInstance(wrapped, ModelNoOutputError)
        self.assertEqual(_no_output_error_name(wrapped), "ModelNoOutputError")

    def test_gateway_envelope_is_recognised(self) -> None:
        # The shape every run in gateway mode actually produces.
        self.assertEqual(_no_output_error_name(gateway_error("ModelNoOutputError")), "ModelNoOutputError")

    def test_content_filter_is_not_recognised(self) -> None:
        # CONTENT_FILTER and GUARDRAIL share this type, and a misconfigured
        # guardrail is infrastructure: it must not be graded as a model result.
        self.assertIsNone(_no_output_error_name(ContentFilterError("blocked")))
        self.assertIsNone(_no_output_error_name(gateway_error("ContentFilterError")))

    def test_unrelated_failures_are_not_recognised(self) -> None:
        self.assertIsNone(_no_output_error_name(RuntimeError("environment gone")))
        self.assertIsNone(_no_output_error_name(ImmediateRetryExhaustedError(10, 10, OSError("connect"))))


class LoopEndsRunTests(unittest.IsolatedAsyncioTestCase):
    def _agent(self, failure: BaseException) -> Terminus2:
        agent = object.__new__(Terminus2)
        agent._logger = logging.getLogger("test-no-output")
        agent._max_episodes = 5
        agent._context = SimpleNamespace()
        agent._session = SimpleNamespace(is_session_alive=AsyncMock(return_value=True))
        agent._setup_episode_logging = lambda logging_dir, episode: (None, None, None)
        agent._handle_llm_interaction = AsyncMock(side_effect=failure)
        agent._trajectory_steps = []
        agent._pending_subagent_refs = None
        agent._pending_handoff_prompt = None
        agent._linear_history = False
        agent._n_episodes = 0
        return agent

    async def test_gateway_no_output_ends_the_run(self) -> None:
        agent = self._agent(gateway_error("ModelNoOutputError"))

        episodes = await agent._run_agent_loop(initial_prompt="go", chat=SimpleNamespace())

        self.assertEqual(episodes, 0, "the failed episode must not be counted as completed")
        self.assertEqual(agent._n_episodes, 0)

    async def test_a_marker_step_records_why_the_score_stands(self) -> None:
        agent = self._agent(ImmediateRetryExhaustedError(10, 10, ModelNoOutputError()))

        await agent._run_agent_loop(initial_prompt="go", chat=SimpleNamespace())

        # Without this a zero from this path is indistinguishable from one the
        # model earned by trying and failing.
        self.assertEqual(len(agent._trajectory_steps), 1)
        step = agent._trajectory_steps[0]
        self.assertEqual(step.source, "system")
        self.assertIn("no usable output", step.message)

    async def test_pending_summarization_is_not_orphaned(self) -> None:
        agent = self._agent(gateway_error("ModelNoOutputError"))
        # Summarization runs inside the query, so a query that summarises and
        # then fails leaves these pending.
        agent._pending_subagent_refs = [SubagentTrajectoryRef(session_id="s-1")]
        agent._pending_handoff_prompt = "carry on"

        await agent._run_agent_loop(initial_prompt="go", chat=SimpleNamespace())

        sources = [s.source for s in agent._trajectory_steps]
        self.assertEqual(sources, ["system", "user", "system"])
        self.assertIsNone(agent._pending_subagent_refs)
        self.assertIsNone(agent._pending_handoff_prompt)

    async def test_content_filter_still_aborts(self) -> None:
        agent = self._agent(ContentFilterError("blocked"))

        with self.assertRaises(ContentFilterError):
            await agent._run_agent_loop(initial_prompt="go", chat=SimpleNamespace())

    async def test_other_failures_still_abort(self) -> None:
        agent = self._agent(RuntimeError("environment gone"))

        with self.assertRaises(RuntimeError):
            await agent._run_agent_loop(initial_prompt="go", chat=SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
