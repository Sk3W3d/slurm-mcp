from __future__ import annotations

import shlex
import subprocess
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


class OpenSSHBackend:
    name = "openssh"

    def __init__(self, ssh_command: tuple[str, ...] = ("ssh",)) -> None:
        self.ssh_command = ssh_command

    def check(self, host: str) -> CommandResult:
        return check_master(self.ssh_command, host)

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
        result = self.check(host)
        return {
            "connected": result.returncode == 0,
            "message": (result.stderr or result.stdout).strip(),
            "instructions": (
                "The openssh backend cannot answer 2FA prompts. Start a master connection "
                f"manually in a terminal with: ssh -MNf {host}"
            ),
        }

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
