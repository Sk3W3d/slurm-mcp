from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class SSHBackend(Protocol):
    name: str

    def check(self, host: str) -> CommandResult: ...

    def run(
        self,
        host: str,
        argv: list[str],
        *,
        timeout: int = 60,
        input_data: bytes | None = None,
    ) -> CommandResult: ...

    def login(
        self,
        host: str,
        *,
        passcode: str | None = None,
        duo_option: str | None = None,
        timeout: int = 120,
    ) -> dict[str, Any]: ...

    def logout(self, host: str) -> dict[str, Any]: ...

    def upload(self, host: str, local_path: Path, remote_path: str, *, timeout: int = 600) -> None: ...

    def download(self, host: str, remote_path: str, local_path: Path, *, timeout: int = 600) -> None: ...


def shell_join(argv: list[str]) -> str:
    if not argv:
        raise ValueError("Remote command cannot be empty.")
    return " ".join(shlex.quote(part) for part in argv)


def argv_in_dir(cwd: str, argv: list[str]) -> list[str]:
    if not cwd:
        raise ValueError("Remote working directory cannot be empty.")
    return ["bash", "-lc", f"cd {shlex.quote(cwd)} && {shell_join(argv)}"]


# ---------------------------------------------------------------------------
# OpenSSH backend: reuses a 2FA-backed ControlMaster session (Linux/macOS/WSL).
# ---------------------------------------------------------------------------


# OpenSSH asks keyboard-interactive (Duo) questions on a TTY, which the MCP
# server has none of. SSH_ASKPASS_REQUIRE=force (OpenSSH 8.4+) sends them to
# this helper instead. It answers only the Duo prompt ("Passcode or option
# (1-N):") and refuses everything else, so it can never type the answer into a
# password, key passphrase, or unknown-host-key prompt.
_DUO_ASKPASS_SCRIPT = """#!/bin/sh
prompt=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
case "$prompt" in
  *password*|*passphrase*|*yes/no*|*fingerprint*)
    echo "slurm-mcp: refusing SSH prompt: $1" >&2; exit 1 ;;
  *passcode*|*option*)
    printf '%s\\n' "$SLURM_MCP_DUO_ANSWER" ;;
  *)
    echo "slurm-mcp: refusing SSH prompt: $1" >&2; exit 1 ;;
esac
"""

_CONTROL_CONFIG_HINT = (
    "Add to ~/.ssh/config for this host:\n"
    "    ControlMaster auto\n"
    "    ControlPath ~/.ssh/cm-%C\n"
    "    ControlPersist 8h"
)


class OpenSSHBackend:
    name = "openssh"

    def __init__(self, ssh_command: tuple[str, ...] = ("ssh",), duo_option: str = "1") -> None:
        self.ssh_command = ssh_command
        self.duo_option = duo_option

    def check(self, host: str) -> CommandResult:
        result = check_master(self.ssh_command, host)
        if result.returncode != 0 and "No ControlPath specified" in result.stderr:
            return CommandResult(
                result.returncode,
                result.stdout,
                f"No ControlPath is configured for {host}, so no connection can be shared. {_CONTROL_CONFIG_HINT}",
            )
        return result

    def run(
        self,
        host: str,
        argv: list[str],
        *,
        timeout: int = 60,
        input_data: bytes | None = None,
    ) -> CommandResult:
        # BatchMode prevents the MCP server from hanging on password or 2FA prompts.
        # ControlMaster=no makes sure this process never becomes a long-lived master.
        return _run_local(
            [*self.ssh_command, "-o", "BatchMode=yes", "-o", "ControlMaster=no", host, shell_join(argv)],
            timeout=timeout,
            input_data=input_data,
        )

    def login(
        self,
        host: str,
        *,
        passcode: str | None = None,
        duo_option: str | None = None,
        timeout: int = 120,
    ) -> dict[str, Any]:
        """Start a ControlMaster for host, answering Duo through an askpass helper.

        Raises RuntimeError when the connection cannot be made, so callers
        record the attempt as failed.
        """
        if self.check(host).returncode == 0:
            return {"connected": True, "reused": True, "message": "Master connection already running."}
        if not _has_control_path(self.ssh_command, host):
            raise RuntimeError(f"No ControlPath is configured for {host}, so the login could not be shared. {_CONTROL_CONFIG_HINT}")

        answer = passcode or duo_option or self.duo_option
        with tempfile.TemporaryDirectory(prefix="slurm-mcp-askpass-") as tmp:
            helper = Path(tmp) / "duo-askpass.sh"
            helper.write_text(_DUO_ASKPASS_SCRIPT, encoding="utf-8")
            helper.chmod(0o700)
            env = {
                **os.environ,
                "SSH_ASKPASS": str(helper),
                "SSH_ASKPASS_REQUIRE": "force",
                "SLURM_MCP_DUO_ANSWER": answer,
            }
            # OpenSSH before 8.4 ignores SSH_ASKPASS_REQUIRE and uses askpass only with DISPLAY set.
            env.setdefault("DISPLAY", ":0")
            # -f backgrounds the master after authentication. Its output goes to a file rather
            # than a pipe, so a master that kept the pipe open could not make this call hang.
            log_path = Path(tmp) / "ssh.log"
            with log_path.open("wb") as log:
                try:
                    completed = subprocess.run(
                        [*self.ssh_command, "-MNf", "-o", "ControlMaster=yes", "-o", "ConnectTimeout=20", host],
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=log,
                        env=env,
                        timeout=timeout,
                        check=False,
                    )
                    returncode: int | None = completed.returncode
                except subprocess.TimeoutExpired:
                    returncode = None
            output = log_path.read_text(encoding="utf-8", errors="replace")

        if returncode is None:
            raise RuntimeError(f"The Duo prompt was not approved within {timeout} seconds.")
        if returncode != 0 or self.check(host).returncode != 0:
            raise RuntimeError(f"SSH login to {host} failed: {_login_failure_detail(output)}")
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        return {"connected": True, "message": lines[-1] if lines else "Master connection started."}

    def logout(self, host: str) -> dict[str, Any]:
        result = _run_local([*self.ssh_command, "-O", "exit", host], timeout=15)
        return {"disconnected": result.returncode == 0, "message": (result.stderr or result.stdout).strip()}

    def upload(self, host: str, local_path: Path, remote_path: str, *, timeout: int = 600) -> None:
        result = self.run(
            host,
            ["bash", "-c", 'cat > "$1"', "_", remote_path],
            timeout=timeout,
            input_data=local_path.read_bytes(),
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "remote upload failed")

    def download(self, host: str, remote_path: str, local_path: Path, *, timeout: int = 600) -> None:
        completed = subprocess.run(
            [*self.ssh_command, "-o", "BatchMode=yes", "-o", "ControlMaster=no", host, shell_join(["cat", "--", remote_path])],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.decode("utf-8", "replace").strip() or "remote download failed")
        local_path.write_bytes(completed.stdout)


def check_master(ssh_command: tuple[str, ...], host: str, timeout: int = 10) -> CommandResult:
    return _run_local([*ssh_command, "-O", "check", host], timeout=timeout)


def _has_control_path(ssh_command: tuple[str, ...], host: str) -> bool:
    """Whether ssh's resolved config for host names a ControlPath to share."""
    result = _run_local([*ssh_command, "-G", host], timeout=10)
    for line in result.stdout.splitlines():
        key, _, value = line.partition(" ")
        if key.lower() == "controlpath":
            return value.strip().lower() not in ("", "none")
    return False


def _login_failure_detail(output: str) -> str:
    """The useful part of a failed ssh login's output, skipping cluster banners."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    refused = [line for line in lines if line.startswith("slurm-mcp: refusing")]
    if refused:
        return refused[0] + " (only the Duo prompt is answered automatically)"
    useful = [
        line for line in lines
        if any(word in line.lower() for word in ("denied", "fail", "error", "refused", "timed out", "could not", "closed"))
    ]
    return "; ".join(useful[-3:]) or (lines[-1] if lines else "no output from ssh")


def remote_command_result(
    ssh_command: tuple[str, ...],
    host: str,
    argv: list[str],
    timeout: int = 60,
) -> CommandResult:
    return OpenSSHBackend(ssh_command).run(host, argv, timeout=timeout)


def remote_command_result_in_dir(
    ssh_command: tuple[str, ...],
    host: str,
    cwd: str,
    argv: list[str],
    timeout: int = 60,
) -> CommandResult:
    return remote_command_result(ssh_command, host, argv_in_dir(cwd, argv), timeout=timeout)


def _run_local(argv: list[str], timeout: int, input_data: bytes | None = None) -> CommandResult:
    # Never let the child inherit stdin: it is the MCP stdio pipe, and on Windows a
    # child using that handle while the server is reading from it can hang forever.
    stdin_args: dict[str, Any] = {"input": input_data} if input_data is not None else {"stdin": subprocess.DEVNULL}
    completed = subprocess.run(
        argv,
        capture_output=True,
        timeout=timeout,
        check=False,
        **stdin_args,
    )
    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout.decode("utf-8", "replace"),
        stderr=completed.stderr.decode("utf-8", "replace"),
    )
