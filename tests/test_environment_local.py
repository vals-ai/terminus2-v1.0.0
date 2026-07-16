import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from terminus2.environment_local import LocalEnvironment


class LocalEnvironmentExecTests(unittest.IsolatedAsyncioTestCase):
    async def test_exec_detaches_child_stdin_from_agent_pty(self) -> None:
        process = SimpleNamespace(
            communicate=AsyncMock(return_value=(b"ok\n", b"")),
            returncode=0,
        )

        with patch(
            "terminus2.environment_local.asyncio.create_subprocess_shell",
            new=AsyncMock(return_value=process),
        ) as create_process:
            environment = LocalEnvironment(trial_paths=SimpleNamespace())
            result = await environment.exec("printf ok", cwd="/workspace")

        self.assertEqual(result.stdout, "ok\n")
        self.assertEqual(result.return_code, 0)
        create_process.assert_awaited_once_with(
            "printf ok",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd="/workspace",
            env=create_process.await_args.kwargs["env"],
        )


if __name__ == "__main__":
    unittest.main()
