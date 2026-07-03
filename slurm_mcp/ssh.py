from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


def check_master(ssh_command: tuple[str, ...], host: str, timeout: int = 10) -> CommandResult:
    return _run_local([*ssh_command, "-O", "check", host], timeout=timeout)


def remote_command(
    ssh_command: tuple[str, ...],
    host: str,
    argv: list[str],
    timeout: int = 60,
) -> str:
    result = remote_command_result(ssh_command, host, argv, timeout=timeout)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "remote command failed"
        raise RuntimeError(stderr)
    return result.stdout.strip()


def remote_command_in_dir(
    ssh_command: tuple[str, ...],
    host: str,
    cwd: str,
    argv: list[str],
    timeout: int = 60,
) -> str:
    result = remote_command_result_in_dir(ssh_command, host, cwd, argv, timeout=timeout)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "remote command failed"
        raise RuntimeError(stderr)
    return result.stdout.strip()


def remote_command_result_in_dir(
    ssh_command: tuple[str, ...],
    host: str,
    cwd: str,
    argv: list[str],
    timeout: int = 60,
) -> CommandResult:
    if not cwd:
        raise ValueError("Remote working directory cannot be empty.")
    command = " ".join(shlex.quote(part) for part in argv)
    return remote_command_result(
        ssh_command,
        host,
        ["bash", "-lc", f"cd {shlex.quote(cwd)} && {command}"],
        timeout=timeout,
    )


def remote_command_result(
    ssh_command: tuple[str, ...],
    host: str,
    argv: list[str],
    timeout: int = 60,
) -> CommandResult:
    if not argv:
        raise ValueError("Remote command cannot be empty.")

    # BatchMode prevents the MCP server from hanging on password or 2FA prompts.
    command = " ".join(shlex.quote(part) for part in argv)
    return _run_local(
        [*ssh_command, "-o", "BatchMode=yes", host, command],
        timeout=timeout,
    )


def _run_local(argv: list[str], timeout: int) -> CommandResult:
    completed = subprocess.run(
        argv,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
