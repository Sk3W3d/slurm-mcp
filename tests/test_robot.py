"""Automation-node ("robot") routing, with a fake backend.

The fake login node runs commands with local bash. The fake automation node
behaves like Alliance's allowed_commands.sh: it accepts commands by prefix,
runs them with plain word splitting (no shell), and on rejection prints
"Command rejected by ..." with exit code 0. Its SFTP is the local filesystem.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any

import anyio
import yaml

from slurm_mcp import robot, server
from slurm_mcp.config import load_settings
from slurm_mcp.ssh import CommandResult, shell_join

from test_server import BASH, _posix

_WRAPPER_PREFIXES = (
    "ls", "cat", "cd", "echo", "uname", "id", "groups", "mv", "cp", "rm", "mkdir", "python", "git", "tar", "dar",
    "gzip", "zip", "bzip2", "rsync", "/usr/libexec/openssh/sftp-server", "scp", "squeue", "scancel", "sbatch",
    "scontrol", "sq", "sinfo", "sacct",
)


def _local(path: str) -> str:
    """Posix path as Git Bash sees it (/c/..., /tmp/...) to a path Python can open."""
    if sys.platform != "win32":
        return path
    return subprocess.run(["cygpath", "-w", path], capture_output=True, text=True, check=True).stdout.strip()


class FakeSFTP:
    def __init__(self) -> None:
        self.closed = False

    def stat(self, path: str) -> Any:
        return os.stat(_local(path))

    def listdir_attr(self, path: str) -> list[Any]:
        entries = []
        for name in os.listdir(_local(path)):
            info = os.stat(os.path.join(_local(path), name))
            entries.append(types.SimpleNamespace(filename=name, st_mode=info.st_mode, st_size=info.st_size,
                                                 st_mtime=info.st_mtime))
        return entries

    def open(self, path: str, mode: str = "rb") -> Any:
        return open(_local(path), mode)

    def mkdir(self, path: str) -> None:
        os.mkdir(_local(path))

    def chmod(self, path: str, mode: int) -> None:
        os.chmod(_local(path), mode)

    def posix_rename(self, old: str, new: str) -> None:
        os.replace(_local(old), _local(new))

    def close(self) -> None:
        self.closed = True


class DualBackend:
    """A login node that may or may not be connected, plus an automation node."""

    name = "paramiko"

    def __init__(self, shim_dir: Path) -> None:
        self.env = {**os.environ, "PATH": str(shim_dir) + os.pathsep + os.environ.get("PATH", "")}
        self.connected: set[str] = set()
        self.duo_result: str = "ok"  # or an error message
        self.robot_error: str | None = None
        self.robot_logins = 0
        self.login_calls: list[str] = []
        self.robot_commands: list[str] = []

    def check(self, host: str) -> CommandResult:
        return CommandResult(0, "ok", "") if host in self.connected else CommandResult(255, "", f"Not connected to {host}.")

    def login(self, host: str, **_: Any) -> dict[str, Any]:
        self.login_calls.append(host)
        if self.duo_result != "ok":
            raise RuntimeError(self.duo_result)
        self.connected.add(host)
        return {"connected": True}

    def logout(self, host: str) -> dict[str, Any]:
        self.connected.discard(host)
        return {"disconnected": True}

    def login_robot(self, key: str, **kwargs: Any) -> dict[str, Any]:
        self.robot_logins += 1
        if self.robot_error:
            raise RuntimeError(self.robot_error)
        self.connected.add(key)
        return {"connected": True, "reused": False, "peer_address": "203.0.113.5", **{"hostname": kwargs["hostname"]}}

    def _bash(self, script: str, env: dict[str, str] | None = None, input_data: bytes | None = None) -> CommandResult:
        completed = subprocess.run([BASH, "-c", script], input=input_data, capture_output=True, env=env or self.env,
                                   check=False, timeout=60)
        return CommandResult(completed.returncode, completed.stdout.decode(), completed.stderr.decode())

    def run(self, host: str, argv: list[str], *, timeout: int = 60, input_data: bytes | None = None) -> CommandResult:
        if host not in self.connected:
            raise RuntimeError(f"Not connected to {host}. Call cluster_login(cluster=...) and approve the Duo prompt first.")
        return self._bash(shell_join(argv), input_data=input_data)

    def run_raw(self, host: str, command: str, *, timeout: int = 60) -> CommandResult:
        assert host.startswith("robot:") and host in self.connected
        self.robot_commands.append(command)
        if not command.startswith(_WRAPPER_PREFIXES):
            return CommandResult(0, f"Command rejected by allowed_commands.sh: {command}\n", "")
        # Plain word splitting, like `$SSH_ORIGINAL_COMMAND` in the wrapper.
        return self._bash('set -f; $SSH_ORIGINAL_COMMAND', env={**self.env, "SSH_ORIGINAL_COMMAND": command})

    def sftp(self, host: str) -> FakeSFTP:
        assert host in self.connected
        return FakeSFTP()

    def upload(self, host: str, local_path: Path, remote_path: str, *, timeout: int = 600) -> None:
        assert host in self.connected
        shutil.copyfile(local_path, _local(remote_path))

    def download(self, host: str, remote_path: str, local_path: Path, *, timeout: int = 600) -> None:
        assert host in self.connected
        shutil.copyfile(_local(remote_path), local_path)


@unittest.skipIf(BASH is None, "needs a POSIX bash")
class RobotRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()
        self.remote = self.tmp / "remote"
        self.shims = self.tmp / "shims"
        (self.remote / "repo").mkdir(parents=True)
        (self.remote / "runs" / "logs").mkdir(parents=True)
        self.shims.mkdir()
        tmp = _posix(self.tmp)
        for name, body in {
            "sbatch": f'echo "$@" > "{tmp}/sbatch-args"; echo "Submitted batch job 777"',
            "squeue": 'echo "777|gpu|train|tester|RUNNING|1:00|30:00|1|4|32G|node1"',
            "sacct": 'echo "777|777|train|gpu|def-pi|4|RUNNING|0:0|00:01:00"',
        }.items():
            shim = self.shims / name
            shim.write_text(f"#!/bin/bash\n{body}\n", encoding="utf-8", newline="\n")
            shim.chmod(0o755)
        (self.remote / "repo" / "train.sh").write_text("#!/bin/bash\necho hi\n", encoding="utf-8", newline="\n")
        log = self.remote / "runs" / "logs" / "train-777.out"
        log.write_text("".join(f"line {i}\n" for i in range(1, 51)), encoding="utf-8", newline="\n")

        base = _posix(self.remote)
        config = {
            "default_cluster": "c",
            "audit_log": str(self.tmp / "audit.jsonl"),
            "local_roots": [str(self.tmp)],
            "robot_enabled": True,
            "clusters": {
                "c": {
                    "host": "login-c",
                    "user": "tester",
                    "robot_host": "robot.c.example",
                    "storage": {"remote_repo": base + "/repo", "run_root": base + "/runs"},
                },
                "plain": {"host": "login-plain", "user": "tester", "storage": {"run_root": base + "/runs"}},
            },
            "profiles": {"cpu": {"time": "00:10:00", "cpus_per_task": 2}},
        }
        config_path = self.tmp / "config.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        self.backend = DualBackend(self.shims)
        server.settings = load_settings(config_path)
        server.backend = self.backend
        server.active_project = server._UNSET
        server._ledger_cache = None
        server._robot_failures.clear()
        server._anchors_ok.clear()

    def tearDown(self) -> None:
        server.settings = None
        server.backend = None
        server._ledger_cache = None
        server._robot_failures.clear()
        self._tmp.cleanup()

    def _login(self, **kwargs: Any) -> dict[str, Any]:
        return anyio.run(lambda: server.cluster_login(cluster="c", **kwargs))

    def test_tools_use_the_automation_node_without_a_login(self) -> None:
        jobs = server.list_jobs()
        self.assertEqual(jobs["jobs"][0]["job_id"], "777")
        self.assertIn("robot:c", self.backend.connected)
        self.assertEqual(self.backend.login_calls, [])  # no Duo push was needed
        self.assertTrue(self.backend.robot_commands[0].startswith("squeue -u tester --noheader --format="))

        self.assertEqual(server.job_status("777")["summary"]["state"], "RUNNING")
        self.assertEqual(server.tail_file("logs/train-777.out", lines=3)["content"], "line 48\nline 49\nline 50\n")
        self.assertIn("line 1\n", server.read_file("logs/train-777.out", max_bytes=20)["content"])
        self.assertEqual(server.tail_job_log("777", lines=1)["content"], "line 50\n")
        listed = server.list_run_files()["files"]
        self.assertEqual([f["relative_path"] for f in listed], ["logs/train-777.out"])

        server.write_file("scripts/new.sh", "echo new\n", executable=True)
        self.assertEqual((self.remote / "repo" / "scripts" / "new.sh").read_text(encoding="utf-8"), "echo new\n")
        with self.assertRaisesRegex(RuntimeError, "Refusing to overwrite"):
            server.write_file("scripts/new.sh", "x")

        submitted = server.submit_job("cpu", "train.sh", purpose="robot submit")
        self.assertEqual(submitted["job_id"], "777")
        args = (self.tmp / "sbatch-args").read_text(encoding="utf-8")
        self.assertIn(f"--chdir={_posix(self.remote)}/repo", args)
        self.assertTrue(args.strip().endswith(f"{_posix(self.remote)}/repo/train.sh"))

        audit = (self.tmp / "audit.jsonl").read_text(encoding="utf-8")
        self.assertIn('"via": "robot"', audit)
        self.assertIn('"via": "robot-sftp"', audit)

    def test_shell_work_still_needs_the_login(self) -> None:
        server.settings = dataclasses.replace(server.get_settings(), allow_remote_shell=True)
        with self.assertRaisesRegex(RuntimeError, "needs the interactive login"):
            server.grep_files("x")  # grep is not accepted by the wrapper
        with self.assertRaisesRegex(RuntimeError, "needs the interactive login"):
            server.slurm_accounts()  # sshare is not accepted either

    def test_login_connection_is_preferred_when_present(self) -> None:
        self._login(method="login")
        server.list_jobs()
        self.assertEqual(self.backend.robot_logins, 0)
        self.assertEqual(self.backend.robot_commands, [])

    def test_robot_is_off_by_default_and_per_cluster(self) -> None:
        server.settings = dataclasses.replace(server.get_settings(), robot_enabled=False)
        with self.assertRaisesRegex(RuntimeError, "Not connected"):
            server.list_jobs()
        server.settings = dataclasses.replace(server.get_settings(), robot_enabled=True)
        with self.assertRaisesRegex(RuntimeError, "Not connected"):
            server.list_jobs(cluster="plain")  # no robot_host for this cluster
        self.assertEqual(self.backend.robot_logins, 0)

    def test_fallback_when_duo_is_not_approved(self) -> None:
        self.backend.duo_result = "Authentication timeout."
        with self.assertRaisesRegex(RuntimeError, "Authentication timeout"):
            self._login()  # fallback is off by default
        server.settings = dataclasses.replace(server.get_settings(), robot_fallback_on_mfa_timeout=True)
        result = self._login()
        self.assertEqual(result["via"], "robot")
        self.assertIn("Authentication timeout", result["login_error"])
        self.assertIn("submit_job", result["capable_tools"])
        with self.assertRaisesRegex(RuntimeError, "Authentication timeout"):
            self._login(method="login")  # an explicit login never falls back

        status = server.cluster_auth_status("c")
        self.assertTrue(status["robot"]["connected"])
        self.assertEqual(status["recent_logins"]["login"]["attempts"], 3)
        self.assertEqual(status["recent_logins"]["login"]["succeeded"], 0)
        self.assertEqual(status["recent_logins"]["robot"]["succeeded"], 1)

    def test_failed_robot_login_is_not_retried_right_away(self) -> None:
        self.backend.robot_error = "key refused"
        for _ in range(2):
            with self.assertRaises(RuntimeError):
                server.list_jobs()
        self.assertEqual(self.backend.robot_logins, 1)
        with self.assertRaisesRegex(RuntimeError, "not retrying"):
            server.list_jobs()


class RobotCommandTests(unittest.TestCase):
    def test_command_string_rules(self) -> None:
        self.assertEqual(robot.command_string(["squeue", "-u", "jzj", "--format=%i|%T"]), "squeue -u jzj --format=%i|%T")
        self.assertEqual(robot.command_string(["tar", "-cf", "/a/b.tar", "--exclude=*.ckpt", "."]),
                         "tar -cf /a/b.tar --exclude=*.ckpt .")
        for argv, why in (
            (["bash", "-c", "ls"], "not accepted"),
            (["find", "/x"], "not accepted"),
            (["git", "log", "--format=%h %s"], "spaces"),
            (["ls", "/x/*"], "wildcards"),
            (["echo", "'quoted'"], "quotes"),
            ([], "empty"),
        ):
            with self.subTest(argv=argv), self.assertRaisesRegex(robot.RobotCommandError, why):
                robot.command_string(argv)

    def test_rejection_is_detected_even_with_exit_code_zero(self) -> None:
        self.assertTrue(robot.was_rejected("Command rejected by allowed_commands.sh: find /x\n"))
        self.assertFalse(robot.was_rejected("777|gpu|train\n"))


if __name__ == "__main__":
    unittest.main()
