"""OpenSSH backend login: starting a ControlMaster and answering only the Duo prompt."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from slurm_mcp import ssh


def _result(returncode: int = 0, stdout: str = "", stderr: str = "") -> ssh.CommandResult:
    return ssh.CommandResult(returncode, stdout, stderr)


@unittest.skipIf(shutil.which("sh") is None, "needs a POSIX shell")
class AskpassHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.helper = Path(self.tmp.name) / "askpass.sh"
        self.helper.write_text(ssh._DUO_ASKPASS_SCRIPT, encoding="utf-8")
        self.helper.chmod(0o700)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def ask(self, prompt: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["sh", str(self.helper), prompt],
            capture_output=True,
            text=True,
            env={**os.environ, "SLURM_MCP_DUO_ANSWER": "1"},
            check=False,
        )

    def test_answers_duo_prompt(self) -> None:
        for prompt in ("Passcode or option (1-1): ", "Duo two-factor login\n\nEnter a passcode or select an option:"):
            result = self.ask(prompt)
            self.assertEqual(result.returncode, 0, prompt)
            self.assertEqual(result.stdout, "1\n")

    def test_refuses_other_prompts(self) -> None:
        for prompt in (
            "jzj@narval.alliancecan.ca's password: ",
            "Enter passphrase for key '/home/u/.ssh/id_ed25519': ",
            "Are you sure you want to continue connecting (yes/no/[fingerprint])? ",
            "Something unexpected: ",
        ):
            result = self.ask(prompt)
            self.assertNotEqual(result.returncode, 0, prompt)
            self.assertEqual(result.stdout, "", prompt)
            self.assertIn("refusing SSH prompt", result.stderr)


class OpenSSHLoginTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = ssh.OpenSSHBackend(("ssh",), duo_option="1")

    def _patch(self, checks: list[ssh.CommandResult], control_path: str = "/home/u/.ssh/cm-abc",
               ssh_returncode: int = 0, ssh_output: bytes = b"Success. Logging you in...\n",
               timeout: bool = False) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []

        def run_local(argv: list[str], timeout: int, input_data: bytes | None = None) -> ssh.CommandResult:
            if "-O" in argv:
                return checks.pop(0)
            if "-G" in argv:
                return _result(stdout=f"hostname narval.alliancecan.ca\ncontrolpath {control_path}\n")
            raise AssertionError(argv)

        def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
            helper = Path(kwargs["env"]["SSH_ASKPASS"])
            calls.append({"argv": argv, **kwargs, "helper_existed": helper.is_file(),
                          "helper_text": helper.read_text(encoding="utf-8")})
            if timeout:
                raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
            kwargs["stderr"].write(ssh_output)
            return subprocess.CompletedProcess(argv, ssh_returncode)

        for patch in (mock.patch.object(ssh, "_run_local", side_effect=run_local),
                      mock.patch("slurm_mcp.ssh.subprocess.run", side_effect=run)):
            patch.start()
            self.addCleanup(patch.stop)
        return calls

    def test_reuses_running_master(self) -> None:
        calls = self._patch([_result(0, stderr="Master running (pid=1)")])
        result = self.backend.login("narval")
        self.assertTrue(result["connected"])
        self.assertTrue(result["reused"])
        self.assertEqual(calls, [])

    def test_starts_master_answering_duo(self) -> None:
        calls = self._patch([_result(255, stderr="Control socket connect: No such file"), _result(0)])
        result = self.backend.login("narval")
        self.assertEqual(result, {"connected": True, "message": "Success. Logging you in..."})
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(call["argv"][:2], ["ssh", "-MNf"])
        self.assertEqual(call["argv"][-1], "narval")
        self.assertIs(call["stdin"], subprocess.DEVNULL)
        self.assertEqual(call["env"]["SSH_ASKPASS_REQUIRE"], "force")
        self.assertEqual(call["env"]["SLURM_MCP_DUO_ANSWER"], "1")
        self.assertTrue(call["helper_existed"])
        self.assertEqual(call["helper_text"], ssh._DUO_ASKPASS_SCRIPT)
        # The helper (and its directory) are removed afterwards.
        self.assertFalse(Path(call["env"]["SSH_ASKPASS"]).exists())

    def test_passcode_replaces_push_option(self) -> None:
        calls = self._patch([_result(255), _result(0)])
        self.backend.login("narval", passcode="123456")
        self.assertEqual(calls[0]["env"]["SLURM_MCP_DUO_ANSWER"], "123456")
        self.assertNotIn("123456", calls[0]["argv"])

    def test_missing_control_path_raises_with_hint(self) -> None:
        calls = self._patch([_result(255, stderr='No ControlPath specified for "-O" command')], control_path="none")
        with self.assertRaisesRegex(RuntimeError, "ControlPath ~/.ssh/cm-%C"):
            self.backend.login("narval")
        self.assertEqual(calls, [])

    def test_failed_login_raises_with_useful_line(self) -> None:
        banner = b"Multifactor authentication is now mandatory\n*****\njzj@narval: Permission denied (publickey).\n"
        self._patch([_result(255)], ssh_returncode=255, ssh_output=banner)
        with self.assertRaisesRegex(RuntimeError, "Permission denied"):
            self.backend.login("narval")

    def test_refused_prompt_is_reported(self) -> None:
        output = b"slurm-mcp: refusing SSH prompt: jzj@narval's password: \nPermission denied\n"
        self._patch([_result(255)], ssh_returncode=255, ssh_output=output)
        with self.assertRaisesRegex(RuntimeError, "refusing SSH prompt.*only the Duo prompt"):
            self.backend.login("narval")

    def test_timeout_raises(self) -> None:
        self._patch([_result(255)], timeout=True)
        with self.assertRaisesRegex(RuntimeError, "not approved within 5 seconds"):
            self.backend.login("narval", timeout=5)

    def test_check_explains_missing_control_path(self) -> None:
        self._patch([_result(255, stderr='No ControlPath specified for "-O" command\n')])
        self.assertIn("~/.ssh/config", self.backend.check("narval").stderr)


if __name__ == "__main__":
    unittest.main()
