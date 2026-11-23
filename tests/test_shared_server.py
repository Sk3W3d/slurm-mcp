"""The shared HTTP server (--serve): token check, per-session state, and no blocking between clients.

Runs the real streamable-HTTP app with uvicorn in a background thread and
talks to it with the MCP client library, the way Claude Code or Codex would.
"""

from __future__ import annotations

import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

import anyio
import httpx
import uvicorn
import yaml
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from slurm_mcp import server
from slurm_mcp.config import load_settings
from slurm_mcp.ssh import CommandResult


class SlowCheckBackend:
    """check() on host 'slow' takes two seconds, like a login waiting for Duo."""

    name = "fake"

    def check(self, host: str) -> CommandResult:
        if host == "slow":
            time.sleep(2)
        return CommandResult(0, f"Connected to {host}.", "")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class SharedServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        config = {
            "audit_log": None,
            "state_db": str(root / "state.sqlite3"),
            "default_cluster": "fast",
            "cluster_defaults": {"storage": {"remote_repo": "/r/{project}", "run_root": "/runs/{project}"}},
            "clusters": {"fast": {"host": "fast", "user": "u"}, "slow": {"host": "slow", "user": "u"}},
        }
        (root / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        server.settings = load_settings(root / "config.yaml")
        server.backend = SlowCheckBackend()
        server.active_project = server._UNSET
        cls.token = server.load_or_create_token(root / "token")
        cls.port = _free_port()
        cls.url = f"http://127.0.0.1:{cls.port}/mcp"
        cls.uvicorn = uvicorn.Server(
            uvicorn.Config(server.http_app(cls.token), host="127.0.0.1", port=cls.port, log_level="warning")
        )
        cls.thread = threading.Thread(target=cls.uvicorn.run, daemon=True)
        cls.thread.start()
        for _ in range(100):
            if cls.uvicorn.started:
                break
            time.sleep(0.05)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.uvicorn.should_exit = True
        cls.thread.join(timeout=10)
        server.settings = None
        server.backend = None
        server._ledger_cache = None
        cls._tmp.cleanup()

    def _with_clients(self, count: int, body: Any) -> Any:
        async def run() -> Any:
            headers = {"Authorization": f"Bearer {self.token}"}
            async with httpx.AsyncClient(headers=headers, timeout=30) as http1,                     streamable_http_client(self.url, http_client=http1) as (r1, w1, _):
                async with ClientSession(r1, w1) as first:
                    await first.initialize()
                    if count == 1:
                        return await body(first)
                    async with httpx.AsyncClient(headers=headers, timeout=30) as http2,                             streamable_http_client(self.url, http_client=http2) as (r2, w2, _):
                        async with ClientSession(r2, w2) as second:
                            await second.initialize()
                            return await body(first, second)

        return anyio.run(run)

    def test_requests_without_the_token_are_refused(self) -> None:
        for headers in ({}, {"Authorization": "Bearer wrong"}):
            with self.subTest(headers=headers):
                response = httpx.post(self.url, headers=headers, json={}, timeout=10)
                self.assertEqual(response.status_code, 401)

    def test_tools_work_over_http(self) -> None:
        async def body(client: ClientSession) -> Any:
            tools = await client.list_tools()
            result = await client.call_tool("list_clusters", {})
            return len(tools.tools), result.structuredContent

        count, clusters = self._with_clients(1, body)
        self.assertGreater(count, 50)
        self.assertEqual(clusters, {"result": ["fast", "slow"]})

    def test_each_client_session_has_its_own_project(self) -> None:
        async def body(first: ClientSession, second: ClientSession) -> Any:
            await first.call_tool("project_use", {"project": "alpha"})
            await second.call_tool("project_use", {"project": "beta"})
            seen_first = (await first.call_tool("project_list", {})).structuredContent["active_project"]
            seen_second = (await second.call_tool("project_list", {})).structuredContent["active_project"]
            return seen_first, seen_second

        self.assertEqual(self._with_clients(2, body), ("alpha", "beta"))

    def test_a_slow_call_does_not_block_other_clients(self) -> None:
        async def body(first: ClientSession, second: ClientSession) -> Any:
            timings: dict[str, float] = {}

            async def slow() -> None:
                await first.call_tool("cluster_auth_status", {"cluster": "slow"})
                timings["slow_done"] = time.monotonic()

            async def fast() -> None:
                await anyio.sleep(0.3)  # start while the slow call is running
                start = time.monotonic()
                await second.call_tool("list_clusters", {})
                timings["fast"] = time.monotonic() - start
                timings["fast_done"] = time.monotonic()

            async with anyio.create_task_group() as group:
                group.start_soon(slow)
                group.start_soon(fast)
            return timings

        timings = self._with_clients(2, body)
        self.assertLess(timings["fast"], 1.0)
        self.assertLess(timings["fast_done"], timings["slow_done"])

    def test_token_file_is_created_once(self) -> None:
        path = Path(self._tmp.name) / "fresh" / "token"
        first = server.load_or_create_token(path)
        self.assertGreaterEqual(len(first), 32)
        self.assertEqual(server.load_or_create_token(path), first)


if __name__ == "__main__":
    unittest.main()
