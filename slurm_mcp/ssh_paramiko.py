"""In-process SSH backend built on Paramiko.

Windows OpenSSH cannot multiplex sessions over a ControlMaster socket (the
native build has no ControlMaster, and the MSYS/Git build cannot pass file
descriptors over its emulated Unix sockets). This backend keeps one
authenticated SSH transport per cluster inside the MCP server process instead,
so a single Duo approval covers every later command, upload, and download.
"""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import paramiko

from .ssh import CommandResult, shell_join


_DEFAULT_IDENTITIES = ("id_ed25519", "id_ecdsa", "id_rsa")
_READ_CHUNK = 65536


@dataclass
class _Connection:
    transport: paramiko.Transport
    hostname: str
    user: str
    port: int
    connected_at: float = field(default_factory=time.time)

    @property
    def alive(self) -> bool:
        return self.transport.is_active() and self.transport.is_authenticated()


@dataclass(frozen=True)
class HostSettings:
    alias: str
    hostname: str
    user: str
    port: int
    identity_files: tuple[Path, ...]


class ParamikoBackend:
    name = "paramiko"

    def __init__(
        self,
        *,
        ssh_config: Path | None = None,
        known_hosts: Path | None = None,
        default_users: dict[str, str] | None = None,
        duo_option: str = "1",
        keepalive_seconds: int = 60,
        connect_timeout: int = 20,
    ) -> None:
        home = Path.home()
        self.ssh_config_path = ssh_config or home / ".ssh" / "config"
        self.known_hosts_path = known_hosts or home / ".ssh" / "known_hosts"
        self.default_users = default_users or {}
        self.duo_option = duo_option
        self.keepalive_seconds = keepalive_seconds
        self.connect_timeout = connect_timeout
        self._connections: dict[str, _Connection] = {}
        self._lock = threading.Lock()
        self._host_locks: dict[str, threading.Lock] = {}

    # -- connection management -------------------------------------------

    def host_settings(self, host: str) -> HostSettings:
        options: dict[str, Any] = {}
        if self.ssh_config_path.is_file():
            config = paramiko.SSHConfig.from_path(str(self.ssh_config_path))
            options = config.lookup(host)
        hostname = str(options.get("hostname") or host)
        user = str(options.get("user") or self.default_users.get(host) or "")
        if not user:
            raise ValueError(f"No SSH user configured for '{host}'. Set 'user' in config.yaml or ~/.ssh/config.")
        port = int(options.get("port") or 22)
        identities = [Path(item).expanduser() for item in options.get("identityfile", [])]
        identities.extend(self.ssh_config_path.parent / name for name in _DEFAULT_IDENTITIES)
        unique: list[Path] = []
        for path in identities:
            if path not in unique and path.is_file():
                unique.append(path)
        return HostSettings(alias=host, hostname=hostname, user=user, port=port, identity_files=tuple(unique))

    def check(self, host: str) -> CommandResult:
        conn = self._connections.get(host)
        if conn is None:
            return CommandResult(255, "", f"Not connected to {host}. Call cluster_login first.")
        if not conn.alive:
            return CommandResult(255, "", f"Connection to {host} was lost. Call cluster_login again.")
        age = int(time.time() - conn.connected_at)
        return CommandResult(0, f"Connected to {conn.user}@{conn.hostname} for {age}s.", "")

    def login(
        self,
        host: str,
        *,
        passcode: str | None = None,
        duo_option: str | None = None,
        timeout: int = 120,
    ) -> dict[str, Any]:
        with self._host_lock(host):
            existing = self._connections.get(host)
            if existing is not None and existing.alive:
                return {"connected": True, "reused": True, "message": self.check(host).stdout}
            if existing is not None:
                self._close(host)

            settings = self.host_settings(host)
            transport = self._open_transport(settings)
            prompts: list[str] = []
            try:
                self._authenticate(
                    transport,
                    settings,
                    passcode=passcode,
                    duo_option=duo_option or self.duo_option,
                    timeout=timeout,
                    prompts=prompts,
                )
            except Exception:
                transport.close()
                raise
            transport.set_keepalive(self.keepalive_seconds)
            self._connections[host] = _Connection(
                transport=transport,
                hostname=settings.hostname,
                user=settings.user,
                port=settings.port,
            )
            return {
                "connected": True,
                "reused": False,
                "message": f"Authenticated as {settings.user}@{settings.hostname}.",
                "mfa_prompts": prompts,
            }

    def logout(self, host: str) -> dict[str, Any]:
        with self._host_lock(host):
            existed = host in self._connections
            self._close(host)
        return {"disconnected": existed, "message": "Closed connection." if existed else "Was not connected."}

    def close_all(self) -> None:
        for host in list(self._connections):
            self._close(host)

    # -- remote operations -------------------------------------------------

    def run(
        self,
        host: str,
        argv: list[str],
        *,
        timeout: int = 60,
        input_data: bytes | None = None,
    ) -> CommandResult:
        command = shell_join(argv)
        transport = self._transport(host)
        channel = transport.open_session(timeout=self.connect_timeout)
        try:
            channel.exec_command(command)
            if input_data:
                channel.sendall(input_data)
            channel.shutdown_write()
            return _collect(channel, timeout)
        finally:
            channel.close()

    def upload(self, host: str, local_path: Path, remote_path: str, *, timeout: int = 600) -> None:
        sftp = paramiko.SFTPClient.from_transport(self._transport(host))
        if sftp is None:
            raise RuntimeError("Could not open SFTP session.")
        try:
            sftp.get_channel().settimeout(timeout)
            sftp.put(str(local_path), remote_path)
        finally:
            sftp.close()

    def download(self, host: str, remote_path: str, local_path: Path, *, timeout: int = 600) -> None:
        sftp = paramiko.SFTPClient.from_transport(self._transport(host))
        if sftp is None:
            raise RuntimeError("Could not open SFTP session.")
        try:
            sftp.get_channel().settimeout(timeout)
            sftp.get(remote_path, str(local_path))
        finally:
            sftp.close()

    # -- internals ---------------------------------------------------------

    def _host_lock(self, host: str) -> threading.Lock:
        with self._lock:
            return self._host_locks.setdefault(host, threading.Lock())

    def _transport(self, host: str) -> paramiko.Transport:
        conn = self._connections.get(host)
        if conn is None or not conn.alive:
            raise RuntimeError(
                f"Not connected to {host}. Call cluster_login(cluster=...) and approve the Duo prompt first."
            )
        return conn.transport

    def _close(self, host: str) -> None:
        conn = self._connections.pop(host, None)
        if conn is not None:
            conn.transport.close()

    def _open_transport(self, settings: HostSettings) -> paramiko.Transport:
        host_keys = paramiko.HostKeys()
        if self.known_hosts_path.is_file():
            host_keys.load(str(self.known_hosts_path))
        lookup_name = settings.hostname if settings.port == 22 else f"[{settings.hostname}]:{settings.port}"
        known = host_keys.lookup(lookup_name)
        if not known:
            raise RuntimeError(
                f"No known_hosts entry for {lookup_name}. Run 'ssh {settings.alias}' once in a terminal "
                "to verify and record the host key, then retry."
            )

        sock = socket.create_connection((settings.hostname, settings.port), timeout=self.connect_timeout)
        transport = paramiko.Transport(sock)
        options = transport.get_security_options()
        preferred = [name for name in options.key_types if name in known]
        if preferred:
            options.key_types = preferred + [name for name in options.key_types if name not in preferred]
        try:
            transport.start_client(timeout=self.connect_timeout)
            server_key = transport.get_remote_server_key()
            expected = known.get(server_key.get_name())
            if expected is None or expected != server_key:
                raise RuntimeError(
                    f"Host key for {lookup_name} does not match ~/.ssh/known_hosts. Refusing to connect."
                )
        except Exception:
            transport.close()
            raise
        return transport

    def _authenticate(
        self,
        transport: paramiko.Transport,
        settings: HostSettings,
        *,
        passcode: str | None,
        duo_option: str,
        timeout: int,
        prompts: list[str],
    ) -> None:
        transport.auth_timeout = timeout
        remaining: list[str] = []
        errors: list[str] = []

        for key in _candidate_keys(settings.identity_files, errors):
            try:
                remaining = list(transport.auth_publickey(settings.user, key))
            except paramiko.AuthenticationException as exc:
                errors.append(f"{key.get_name()}: {exc}")
                continue
            break

        if transport.is_authenticated():
            return
        if "keyboard-interactive" not in remaining:
            detail = "; ".join(errors) or "no usable private key found"
            raise RuntimeError(
                "Public key authentication failed. Make sure your public key is registered in CCDB "
                f"(https://ccdb.alliancecan.ca/ssh_authorized_keys). Details: {detail}"
            )

        def handler(title: str, instructions: str, prompt_list: list[tuple[str, bool]]) -> list[str]:
            answers: list[str] = []
            for prompt, _echo in prompt_list:
                prompts.append(" ".join(part for part in (title, instructions, prompt) if part).strip())
                lowered = prompt.lower()
                if "password" in lowered:
                    raise paramiko.AuthenticationException("Server asked for a password; only key + MFA is supported.")
                answers.append(passcode or duo_option)
            return answers

        transport.auth_interactive(settings.user, handler)
        if not transport.is_authenticated():
            raise RuntimeError("Multifactor authentication was not completed.")


def _candidate_keys(identity_files: tuple[Path, ...], errors: list[str]):
    try:
        agent_keys = paramiko.Agent().get_keys()
    except Exception:  # pragma: no cover - depends on local agent availability
        agent_keys = ()
    yield from agent_keys
    for path in identity_files:
        try:
            yield paramiko.PKey.from_path(str(path))
        except paramiko.PasswordRequiredException:
            errors.append(f"{path}: encrypted key (load it into ssh-agent instead)")
        except Exception as exc:
            errors.append(f"{path}: {exc}")


def _collect(channel: paramiko.Channel, timeout: int) -> CommandResult:
    deadline = time.monotonic() + timeout
    stdout = bytearray()
    stderr = bytearray()
    while True:
        progressed = False
        while channel.recv_ready():
            stdout.extend(channel.recv(_READ_CHUNK))
            progressed = True
        while channel.recv_stderr_ready():
            stderr.extend(channel.recv_stderr(_READ_CHUNK))
            progressed = True
        if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
            break
        if time.monotonic() > deadline:
            raise TimeoutError(f"Remote command timed out after {timeout}s.")
        if not progressed:
            time.sleep(0.02)
    return CommandResult(
        returncode=channel.recv_exit_status(),
        stdout=stdout.decode("utf-8", "replace"),
        stderr=stderr.decode("utf-8", "replace"),
    )
