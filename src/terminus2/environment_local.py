import asyncio
import logging
import os
import shutil
from pathlib import Path

from terminus2.environment_base import BaseEnvironment, ExecResult
from terminus2.environment_type import EnvironmentType
from terminus2.trial.paths import TrialPaths


class LocalEnvironment(BaseEnvironment):
    """An environment that executes commands directly on the local machine."""

    def __init__(
        self,
        trial_paths: TrialPaths,
        logger: logging.Logger | None = None,
    ):
        self._trial_paths = trial_paths
        self._logger = logger or logging.getLogger(__name__)

    # --- Skip the parent __init__ validation since we don't need environment_dir etc. ---

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.DOCKER  # Closest match

    @property
    def is_mounted(self) -> bool:
        return True

    @property
    def trial_paths(self) -> TrialPaths:
        return self._trial_paths

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    def _validate_definition(self):
        pass

    async def start(self, force_build: bool) -> None:
        self._trial_paths.mkdir()

    async def stop(self, delete: bool) -> None:
        pass

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        shutil.copy2(str(source_path), target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        shutil.copytree(str(source_dir), target_dir, dirs_exist_ok=True)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        shutil.copy2(source_path, str(target_path))

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        shutil.copytree(source_dir, str(target_dir), dirs_exist_ok=True)

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> ExecResult:
        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=full_env,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout_sec,
            )
            return ExecResult(
                stdout=stdout.decode(errors="replace") if stdout else "",
                stderr=stderr.decode(errors="replace") if stderr else "",
                return_code=proc.returncode or 0,
            )
        except asyncio.TimeoutError:
            proc.kill()
            return ExecResult(
                stdout="",
                stderr="Command timed out",
                return_code=124,
            )
