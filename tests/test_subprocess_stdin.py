"""Subprocesses must never inherit the server's stdin, which is the MCP stdio pipe.

On Windows, a child that inherits that pipe blocks behind the server's pending
read on it, so the child (git, ssh) never starts and the tool call hangs.
"""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from unittest import mock

from slurm_mcp import local_files, ssh


def _completed(*_: object, **__: object) -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=b"", stderr=b"")


class NoInheritedStdinTests(unittest.TestCase):
    def test_local_git_calls_use_devnull(self) -> None:
        with mock.patch("slurm_mcp.local_files.subprocess.run", side_effect=_completed) as run:
            local_files.run_local_git(Path("."), ["status"])
            local_files.local_git_ok(Path("."), ["status"])
        for call in run.call_args_list:
            self.assertIs(call.kwargs.get("stdin"), subprocess.DEVNULL)

    def test_openssh_calls_use_devnull_unless_sending_input(self) -> None:
        backend = ssh.OpenSSHBackend(("ssh",))
        with mock.patch("slurm_mcp.ssh.subprocess.run", side_effect=_completed) as run:
            backend.run("narval", ["true"])
            backend.check("narval")
            backend.download("narval", "/x", Path(self.id() + ".tmp"))
        Path(self.id() + ".tmp").unlink(missing_ok=True)
        for call in run.call_args_list:
            self.assertIs(call.kwargs.get("stdin"), subprocess.DEVNULL)
            self.assertNotIn("input", call.kwargs)

        with mock.patch("slurm_mcp.ssh.subprocess.run", side_effect=_completed) as run:
            backend.run("narval", ["cat"], input_data=b"data")
        self.assertEqual(run.call_args.kwargs.get("input"), b"data")
        self.assertNotIn("stdin", run.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
