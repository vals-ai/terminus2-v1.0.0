"""A failure has to leave evidence where the run's artifacts are collected.

Without this the only trace is the process exit status. A harness reports that
as a non-zero exit, retries the task, and the retry overwrites the current copy
of the uploaded logs -- so the attempt that failed is not the one anybody reads.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from terminus2.trial.paths import TrialPaths


class Args:
    """Only the fields the CLI reads; anything else defaults to None."""

    def __init__(self, logs_dir: Path) -> None:
        self.logs_dir = logs_dir
        self.instruction = "do the thing"
        self.problem_path = None
        self.model = "test/model"
        self.no_summarize = True

    def __getattr__(self, name: str):
        return None


class AgentErrorIsRecordedTests(unittest.IsolatedAsyncioTestCase):
    async def _run(
        self,
        temp_dir: str,
        *,
        run_failure: BaseException | None = None,
        start_failure: BaseException | None = None,
    ) -> TrialPaths:
        from terminus2 import cli

        logs_dir = Path(temp_dir) / "logs"
        args = Args(logs_dir)

        agent = AsyncMock()
        agent.run = AsyncMock(side_effect=run_failure)
        agent._session = AsyncMock()
        environment = AsyncMock()
        environment.start = AsyncMock(side_effect=start_failure)

        # Each of these is imported inside `_run_agent`, so they must be
        # patched at their source module rather than as attributes of `cli`.
        with (
            patch("terminus2.terminus_2.Terminus2", return_value=agent),
            patch("terminus2.environment_local.LocalEnvironment", return_value=environment),
            patch("terminus2.model_patch.capture_model_patch_baseline", return_value=None),
            patch("terminus2.model_patch.cleanup_model_patch_baseline"),
        ):
            expected = run_failure or start_failure
            if expected is None:
                await cli._run_agent(args)
            else:
                with self.assertRaises(type(expected)):
                    await cli._run_agent(args)

        return TrialPaths(trial_dir=logs_dir)

    async def test_traceback_is_written_where_the_archive_collects_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = await self._run(temp_dir, run_failure=RuntimeError("boom in the agent"))

            recorded = (paths.agent_dir / "error.log").read_text()
            self.assertIn("boom in the agent", recorded)
            self.assertIn("Traceback", recorded)

    async def test_a_setup_failure_is_also_recorded(self) -> None:
        # The environment failing to start is precisely the case that used to
        # leave nothing behind, because it happened before the guarded block.
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = await self._run(temp_dir, start_failure=RuntimeError("sandbox never started"))

            self.assertIn("sandbox never started", (paths.agent_dir / "error.log").read_text())

    async def test_an_interrupt_is_recorded_and_still_propagates(self) -> None:
        # `except Exception` would miss this, and a cancelled or interrupted
        # run is one of the cases most worth having a trace of.
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = await self._run(temp_dir, run_failure=KeyboardInterrupt())

            self.assertTrue((paths.agent_dir / "error.log").exists())

    async def test_a_successful_run_leaves_no_error_log(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            logs_dir = Path(temp_dir) / "logs"
            stale = TrialPaths(trial_dir=logs_dir)
            stale.mkdir()
            # A previous attempt on the same mount left this behind; shipping it
            # beside a passing trial would mislabel the pass.
            (stale.agent_dir / "error.log").write_text("stale traceback from an earlier attempt")

            paths = await self._run(temp_dir)

            self.assertFalse((paths.agent_dir / "error.log").exists())


if __name__ == "__main__":
    unittest.main()
