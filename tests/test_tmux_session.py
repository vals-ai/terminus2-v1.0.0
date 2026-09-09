import shlex
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from terminus2.tmux_session import TmuxSession


def _session() -> TmuxSession:
    return TmuxSession(
        session_name="agent",
        environment=MagicMock(),
        logging_path=Path("/logs/agent.log"),
        local_asciinema_recording_path=Path("/tmp/a.cast"),
        remote_asciinema_recording_path=Path("/logs/a.cast"),
    )


class TmuxSessionCommandTests(unittest.TestCase):
    def test_start_session_does_not_need_the_script_utility(self) -> None:
        """fedora:42 ships util-linux-core without `script`; `tmux -d` needs no PTY."""
        command = _session()._tmux_start_session

        self.assertNotIn("script", shlex.split(command.split("&&")[-1]))
        self.assertIn("tmux new-session -x 160 -y 40 -d -s agent 'bash --login'", command)
        self.assertIn("pipe-pane -t agent 'cat > /logs/agent.log'", command)

    def test_send_keys_drops_nul_bytes_the_model_emitted(self) -> None:
        """A NUL in argv makes Popen raise ValueError and ends the whole run."""
        command = _session()._tmux_send_keys(["echo a\x00b", "Enter"])

        self.assertNotIn("\x00", command)
        self.assertEqual(shlex.split(command), ["tmux", "send-keys", "-t", "agent", "echo ab", "Enter"])


if __name__ == "__main__":
    unittest.main()
