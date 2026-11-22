from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import paramiko

from slurm_mcp.ssh_paramiko import HostSettings, ParamikoBackend


DUO_PROMPT = (
    "Duo two-factor login for alice\n\nEnter a passcode or select one of the following options:\n\n"
    " 1. Duo Push to XXX-XXX-1234\n\nPasscode or option (1-1): "
)


class FakeTransport:
    def __init__(self, *, remaining: list[str], prompts: list[tuple[str, bool]]) -> None:
        self.remaining = remaining
        self.prompts = prompts
        self.authenticated = False
        self.answers: list[str] = []
        self.auth_timeout = None

    def auth_publickey(self, user: str, key: object) -> list[str]:
        if not self.remaining:
            self.authenticated = True
        return self.remaining

    def is_authenticated(self) -> bool:
        return self.authenticated

    def auth_interactive(self, user: str, handler):  # noqa: ANN001
        self.answers = handler("", "", self.prompts)
        self.authenticated = True
        return []


class ParamikoBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.ssh_dir = Path(self._tmp.name)
        key_path = self.ssh_dir / "id_rsa"
        paramiko.RSAKey.generate(1024).write_private_key_file(str(key_path))
        (self.ssh_dir / "config").write_text(
            "Host narval\n  HostName narval.alliancecan.ca\n  User alice\n  Port 2222\n", encoding="utf-8"
        )
        self.backend = ParamikoBackend(ssh_config=self.ssh_dir / "config", known_hosts=self.ssh_dir / "known_hosts")
        self.settings = self.backend.host_settings("narval")
        self.agent = mock.patch("slurm_mcp.ssh_paramiko.paramiko.Agent")
        self.agent.start().return_value.get_keys.return_value = ()

    def tearDown(self) -> None:
        mock.patch.stopall()
        self._tmp.cleanup()

    def test_host_settings_from_ssh_config(self) -> None:
        self.assertEqual(self.settings.hostname, "narval.alliancecan.ca")
        self.assertEqual(self.settings.user, "alice")
        self.assertEqual(self.settings.port, 2222)
        self.assertEqual(self.settings.identity_files, (self.ssh_dir / "id_rsa",))

    def test_missing_user_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "No SSH user"):
            self.backend.host_settings("unknown-host")

    def test_check_before_login(self) -> None:
        result = self.backend.check("narval")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cluster_login", result.stderr)
        with self.assertRaisesRegex(RuntimeError, "cluster_login"):
            self.backend.run("narval", ["whoami"])

    def test_duo_prompt_answered_with_push_option(self) -> None:
        transport = FakeTransport(remaining=["keyboard-interactive"], prompts=[(DUO_PROMPT, True)])
        prompts: list[str] = []
        self.backend._authenticate(transport, self.settings, passcode=None, duo_option="1", timeout=5, prompts=prompts)
        self.assertEqual(transport.answers, ["1"])
        self.assertIn("Duo Push", prompts[0])

    def test_duo_prompt_answered_with_passcode(self) -> None:
        transport = FakeTransport(remaining=["keyboard-interactive"], prompts=[(DUO_PROMPT, True)])
        self.backend._authenticate(transport, self.settings, passcode="123456", duo_option="1", timeout=5, prompts=[])
        self.assertEqual(transport.answers, ["123456"])

    def test_password_prompt_is_refused(self) -> None:
        transport = FakeTransport(remaining=["keyboard-interactive"], prompts=[("Password: ", False)])
        with self.assertRaises(paramiko.AuthenticationException):
            self.backend._authenticate(transport, self.settings, passcode=None, duo_option="1", timeout=5, prompts=[])

    def test_key_only_login_skips_mfa(self) -> None:
        transport = FakeTransport(remaining=[], prompts=[])
        self.backend._authenticate(transport, self.settings, passcode=None, duo_option="1", timeout=5, prompts=[])
        self.assertTrue(transport.is_authenticated())
        self.assertEqual(transport.answers, [])

    def test_no_keys_explains_ccdb(self) -> None:
        settings = HostSettings(alias="narval", hostname="h", user="alice", port=22, identity_files=())
        transport = FakeTransport(remaining=["keyboard-interactive"], prompts=[])
        with self.assertRaisesRegex(RuntimeError, "CCDB"):
            self.backend._authenticate(transport, settings, passcode=None, duo_option="1", timeout=5, prompts=[])

    def test_known_hosts_skips_malformed_lines(self) -> None:
        from slurm_mcp.ssh_paramiko import load_known_hosts

        key = paramiko.RSAKey.generate(1024)
        known = self.ssh_dir / "known_hosts"
        known.write_text(
            "# a real comment\n"
            "; vulcan.alliancecan.ca ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMuIj6T45HqVeJgRotH9Qq46FzidekS2lXkD7FOTltnC\n"
            "@revoked * ssh-rsa AAAA\n"
            f"narval.alliancecan.ca ssh-rsa {key.get_base64()}\n"
            "garbage\n",
            encoding="utf-8",
        )
        host_keys, skipped = load_known_hosts(known)
        self.assertEqual(skipped, [2, 3, 5])
        self.assertEqual(host_keys.lookup("narval.alliancecan.ca")["ssh-rsa"], key)
        self.assertIsNone(host_keys.lookup("vulcan.alliancecan.ca"))
        missing_keys, missing_skipped = load_known_hosts(self.ssh_dir / "missing")
        self.assertEqual((len(missing_keys), missing_skipped), (0, []))

    def test_unknown_host_key_is_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "known_hosts"):
            self.backend._open_transport(self.settings)


if __name__ == "__main__":
    unittest.main()
