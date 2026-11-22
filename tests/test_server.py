"""End-to-end tool tests that use local bash in place of SSH.

LocalBackend runs each "remote" command through bash on this machine, so the
tools' real argv, quoting, stdin handling, and git flows are exercised. On
Windows this needs Git Bash; the tests are skipped when no usable bash exists.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

import yaml

from slurm_mcp import server
from slurm_mcp.config import load_settings
from slurm_mcp.ssh import CommandResult, shell_join


def _find_bash() -> str | None:
    bash = shutil.which("bash")
    if bash is None:
        return None
    # C:\Windows\System32\bash.exe is the WSL launcher, which sees a different filesystem.
    if sys.platform == "win32" and "system32" in bash.lower():
        return None
    return bash


BASH = _find_bash()
GIT_ENV = {
    "GIT_AUTHOR_NAME": "Test",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_COMMITTER_NAME": "Test",
    "GIT_COMMITTER_EMAIL": "test@example.com",
}


def _posix(path: Path) -> str:
    if sys.platform != "win32":
        return str(path)
    return subprocess.run(["cygpath", "-u", str(path)], capture_output=True, text=True, check=True).stdout.strip()


class LocalBackend:
    name = "local"

    def __init__(self, shim_dir: Path) -> None:
        self.env = {**os.environ, **GIT_ENV, "PATH": _posix(shim_dir) + os.pathsep + os.environ.get("PATH", "")}
        if sys.platform == "win32":
            self.env["PATH"] = str(shim_dir) + os.pathsep + os.environ.get("PATH", "")
        self.calls: list[list[str]] = []

    def check(self, host: str) -> CommandResult:
        return CommandResult(0, "local", "")

    def run(self, host: str, argv: list[str], *, timeout: int = 60, input_data: bytes | None = None) -> CommandResult:
        self.calls.append(argv)
        completed = subprocess.run(
            [BASH, "-c", shell_join(argv)],
            input=input_data,
            capture_output=True,
            timeout=timeout,
            env=self.env,
            check=False,
        )
        return CommandResult(
            completed.returncode,
            completed.stdout.decode("utf-8", "replace"),
            completed.stderr.decode("utf-8", "replace"),
        )

    def login(self, host: str, **_: Any) -> dict[str, Any]:
        return {"connected": True}

    def logout(self, host: str) -> dict[str, Any]:
        return {"disconnected": True}

    def upload(self, host: str, local_path: Path, remote_path: str, *, timeout: int = 600) -> None:
        result = self.run(host, ["bash", "-c", 'cat > "$1"', "_", remote_path], input_data=local_path.read_bytes())
        if result.returncode != 0:
            raise RuntimeError(result.stderr)

    def download(self, host: str, remote_path: str, local_path: Path, *, timeout: int = 600) -> None:
        completed = subprocess.run(
            [BASH, "-c", shell_join(["cat", "--", remote_path])], capture_output=True, env=self.env, check=False
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.decode())
        local_path.write_bytes(completed.stdout)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, **GIT_ENV},
    ).stdout.strip()


@unittest.skipIf(BASH is None, "needs a POSIX bash")
class LocalToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()
        self.local = self.tmp / "local"
        self.remote = self.tmp / "remote"
        self.shims = self.tmp / "shims"
        for path in (self.local, self.remote / "runs", self.shims):
            path.mkdir(parents=True)
        # Alliance login nodes have util-linux setsid; Git Bash does not.
        setsid = self.shims / "setsid"
        setsid.write_text('#!/bin/bash\nexec "$@"\n', encoding="utf-8", newline="\n")
        setsid.chmod(0o755)

        config = {
            "default_cluster": "local",
            "audit_log": str(self.tmp / "audit.jsonl"),
            "local_roots": [str(self.local)],
            "cluster_defaults": {"policies": {"compute_node_internet": False}},
            "clusters": {
                "local": {
                    "host": "localhost",
                    "user": "tester",
                    "slurm": {"default_account": "def-pi", "job_env": {"OMP_NUM_THREADS": "4"}},
                    "storage": {
                        "remote_repo": _posix(self.remote / "repo"),
                        "run_root": _posix(self.remote / "runs"),
                        "venv_root": _posix(self.remote / "venvs"),
                        "hf_home": _posix(self.remote / "hf"),
                    },
                }
            },
            "profiles": {"cpu": {"time": "00:10:00", "cpus_per_task": 2}},
        }
        config_path = self.tmp / "config.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        self.backend = LocalBackend(self.shims)
        server.settings = load_settings(config_path)
        server.backend = self.backend

    def tearDown(self) -> None:
        server.settings = None
        server.backend = None
        self._tmp.cleanup()

    def _init_local_repo(self) -> Path:
        repo = self.local / "proj"
        repo.mkdir()
        _git(repo, "init", "-q", "-b", "main")
        (repo / "train.py").write_text("print('v1')\n", encoding="utf-8")
        _git(repo, "add", ".")
        _git(repo, "commit", "-q", "-m", "v1")
        return repo

    def test_write_then_read_file_and_refuse_overwrite(self) -> None:
        (self.remote / "repo").mkdir()
        written = server.write_file("scripts/job.sh", "#!/bin/bash\necho 'it''s ok' $HOME\n", executable=True)
        self.assertEqual(written["relative_path"], "scripts/job.sh")
        read = server.read_file("scripts/job.sh", base="remote_repo")
        self.assertEqual(read["content"], "#!/bin/bash\necho 'it''s ok' $HOME\n")
        with self.assertRaisesRegex(RuntimeError, "Refusing to overwrite"):
            server.write_file("scripts/job.sh", "x")
        server.write_file("scripts/job.sh", "y\n", overwrite=True)
        self.assertEqual(server.read_file("scripts/job.sh", base="remote_repo")["content"], "y\n")

    def test_write_file_rejects_escape(self) -> None:
        with self.assertRaises(ValueError):
            server.write_file("../outside.sh", "x")
        with self.assertRaises(ValueError):
            server.write_file("/etc/passwd", "x")

    def test_upload_and_download_round_trip(self) -> None:
        source = self.local / "data.bin"
        source.write_bytes(b"hello\x00world\n")
        server.upload_file("data.bin", "inputs/data.bin", base="run_root")
        result = server.download_file("inputs/data.bin")
        self.assertEqual(Path(result["local_path"]).read_bytes(), b"hello\x00world\n")
        self.assertIn("slurm-mcp-downloads", result["local_path"])
        with self.assertRaises(ValueError):
            server.download_file("inputs/data.bin")
        with self.assertRaises(ValueError):
            server.upload_file(str(self.tmp / "config.yaml"), "x.yaml", base="run_root")

    def test_submit_job_dry_run_merges_env_overrides_and_account(self) -> None:
        preview = server.submit_job("cpu", "scripts/job.sh", overrides={"time": "01:00:00"}, env={"SEED": "1"}, dry_run=True)
        argv = preview["argv"]
        self.assertIn("--time=01:00:00", argv)
        self.assertIn("--account=def-pi", argv)
        export = next(item for item in argv if item.startswith("--export="))
        for part in ("ALL", "HF_HOME=", "HF_HUB_OFFLINE=1", "OMP_NUM_THREADS=4", "SEED=1"):
            self.assertIn(part, export)
        self.assertEqual(self.backend.calls, [])

    def test_background_task_lifecycle(self) -> None:
        target, roots, _ = server._env_context(server.get_settings(), None)
        task = server._launch_task(server.get_settings(), target, roots, "demo", "echo hello from task\nexit 3", "demo task")
        status: dict[str, Any] = {}
        for _ in range(100):
            status = server.remote_task_status(task["task_id"])
            if status["state"] in {"succeeded", "failed"}:
                break
            time.sleep(0.1)
        self.assertEqual(status["state"], "failed")
        self.assertEqual(status["exit_code"], 3)
        self.assertIn("hello from task", status["log_tail"])
        self.assertEqual(status["description"], "demo task")
        listed = server.list_remote_tasks()["tasks"]
        self.assertEqual(listed[0]["task_id"], task["task_id"])
        self.assertEqual(listed[0]["state"], "failed")

    def test_hf_store_token_is_private_and_not_audited(self) -> None:
        os.environ["HF_TOKEN"] = "hf_secret_value"
        try:
            server.hf_store_token()
        finally:
            del os.environ["HF_TOKEN"]
        self.assertEqual((self.remote / "hf" / "token").read_text(), "hf_secret_value")
        self.assertNotIn("hf_secret_value", (self.tmp / "audit.jsonl").read_text())

    def test_sync_local_commits_fast_forwards_and_refuses_divergence(self) -> None:
        local = self._init_local_repo()
        _git(self.remote, "clone", "-q", str(local), "repo")
        remote_repo = self.remote / "repo"

        (local / "train.py").write_text("print('v2')\n", encoding="utf-8")
        _git(local, "commit", "-q", "-am", "v2")
        result = server.sync_local_commits("proj")
        self.assertEqual(result["remote_commit_after"], _git(local, "rev-parse", "HEAD"))
        self.assertEqual((remote_repo / "train.py").read_text(), "print('v2')\n")
        self.assertFalse((remote_repo / ".git" / "slurm-mcp-sync.bundle").exists())

        again = server.sync_local_commits("proj")
        self.assertTrue(again["already_up_to_date"])

        _git(local, "checkout", "-q", "-b", "feature")
        (local / "new.py").write_text("x = 1\n", encoding="utf-8")
        _git(local, "add", "new.py")
        _git(local, "commit", "-q", "-m", "feature")
        feature = server.sync_local_commits("proj")
        self.assertTrue(feature["checked_out"])
        self.assertEqual(_git(remote_repo, "branch", "--show-current"), "feature")
        self.assertTrue((remote_repo / "new.py").exists())

        (remote_repo / "remote_only.txt").write_text("r\n", encoding="utf-8")
        _git(remote_repo, "add", ".")
        _git(remote_repo, "commit", "-q", "-m", "remote only")
        (local / "local_only.txt").write_text("l\n", encoding="utf-8")
        _git(local, "add", ".")
        _git(local, "commit", "-q", "-m", "local only")
        with self.assertRaisesRegex(ValueError, "diverged"):
            server.sync_local_commits("proj")

    def test_sync_local_commits_bootstraps_missing_remote_repo(self) -> None:
        local = self._init_local_repo()
        result = server.sync_local_commits("proj")
        remote_repo = self.remote / "repo"
        self.assertTrue(result["initialized_remote_repo"])
        self.assertTrue(result["checked_out"])
        self.assertEqual(_git(remote_repo, "branch", "--show-current"), "main")
        self.assertEqual(_git(remote_repo, "rev-parse", "HEAD"), _git(local, "rev-parse", "HEAD"))
        self.assertEqual((remote_repo / "train.py").read_text(), "print('v1')\n")

    def test_sync_refuses_non_empty_non_repo_directory(self) -> None:
        self._init_local_repo()
        (self.remote / "repo").mkdir()
        (self.remote / "repo" / "stray.txt").write_text("x", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "not empty"):
            server.sync_local_commits("proj")

    def test_remote_git_checkout_refuses_dirty_tree(self) -> None:
        local = self._init_local_repo()
        _git(self.remote, "clone", "-q", str(local), "repo")
        (self.remote / "repo" / "train.py").write_text("dirty\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "uncommitted"):
            server.remote_git_checkout("main")

    def test_generated_task_scripts_are_valid_bash(self) -> None:
        from slurm_mcp import huggingface as hf
        from slurm_mcp import python_env, remote_tasks

        bodies = [
            python_env.build_venv_script(
                venv="/v/default",
                modules=["python/3.11", "arrow"],
                remote_repo="/r",
                requirements="requirements.txt",
                packages=["numpy>=1.26", "accelerate[torch]"],
                no_index=True,
                recreate=True,
            ),
            hf.build_download_script(
                hf_home="/p/hf",
                modules=["python/3.11"],
                venv="/v/_slurm-mcp-hf",
                create_venv=True,
                repo_id="org/model-name",
                repo_type="model",
                revision=None,
                allow_patterns=["*.json"],
                ignore_patterns=["*.bin"],
            ),
        ]
        for body in bodies:
            script = remote_tasks.wrap_task_script("/t/mcp-tasks/x", body, description="check")
            result = subprocess.run([BASH, "-n"], input=script.encode(), capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_run_remote_command_disabled_by_default(self) -> None:
        with self.assertRaisesRegex(ValueError, "disabled"):
            server.run_remote_command("echo hi")


@unittest.skipIf(BASH is None, "needs a POSIX bash")
class ProjectToolTests(unittest.TestCase):
    """Several projects on one cluster through {project} storage templates."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()
        self.local = self.tmp / "local"
        self.remote = self.tmp / "remote"
        self.shims = self.tmp / "shims"
        for path in (self.local, self.remote, self.shims):
            path.mkdir(parents=True)
        # Fake Slurm: sbatch hands out increasing job ids; sacct and squeue print fixture files.
        shims = {
            "setsid": 'exec "$@"',
            "sbatch": (
                f'n=$(cat "{_posix(self.tmp)}/jobid" 2>/dev/null || echo 1000); n=$((n+1)); '
                f'echo $n > "{_posix(self.tmp)}/jobid"; echo "Submitted batch job $n"'
            ),
            "sacct": f'cat "{_posix(self.tmp)}/sacct.out" 2>/dev/null; exit 0',
            "squeue": f'cat "{_posix(self.tmp)}/squeue.out" 2>/dev/null; exit 0',
        }
        for name, body in shims.items():
            shim = self.shims / name
            shim.write_text(f"#!/bin/bash\n{body}\n", encoding="utf-8", newline="\n")
            shim.chmod(0o755)

        base = _posix(self.remote)
        config = {
            "default_cluster": "local",
            "audit_log": None,
            "local_roots": [str(self.local)],
            "cluster_defaults": {
                "python": {"venv_name": "{project}"},
                "storage": {
                    "workspace_root": base + "/projects",
                    "remote_repo": "{workspace_root}/{project}",
                    "run_root": base + "/runs/{project}",
                    "venv_root": base + "/venvs",
                    "hf_home": base + "/hf",
                    "tasks_root": base + "/tasks",
                },
            },
            "clusters": {
                "local": {"host": "localhost", "user": "tester"},
                "fixed": {
                    "host": "localhost",
                    "user": "tester",
                    "storage": {"remote_repo": base + "/fixed", "run_root": base + "/fixed-runs"},
                },
            },
            "profiles": {"cpu": {"time": "00:10:00"}},
        }
        config_path = self.tmp / "config.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        self.backend = LocalBackend(self.shims)
        server.settings = load_settings(config_path)
        server.backend = self.backend
        server.active_project = server._UNSET

    def tearDown(self) -> None:
        server.settings = None
        server.backend = None
        server.active_project = server._UNSET
        server._ledger_cache = None
        self._tmp.cleanup()

    def _fixture(self, name: str, text: str) -> None:
        (self.tmp / name).write_text(text, encoding="utf-8", newline="\n")

    def _sync_project_repo(self) -> str:
        repo = self.local / "proj"
        repo.mkdir()
        _git(repo, "init", "-q", "-b", "main")
        (repo / "job.sh").write_text("#!/bin/bash\necho hi\n", encoding="utf-8", newline="\n")
        _git(repo, "add", ".")
        _git(repo, "commit", "-q", "-m", "job")
        server.sync_local_commits("proj")
        return _git(repo, "rev-parse", "HEAD")

    def test_submitted_jobs_are_recorded_and_refreshed(self) -> None:
        server.project_use("alpha")
        head = self._sync_project_repo()
        submitted = server.submit_job("cpu", "job.sh", purpose="baseline run")
        self.assertEqual(submitted["job_id"], "1001")
        self.assertEqual(submitted["git_commit"], head)
        self.assertFalse(submitted["git_dirty"])
        self.assertNotIn("ledger_error", submitted)

        self._fixture("sacct.out", "1001|1001|job.sh|cpu|def-pi|2|RUNNING|0:0|00:00:30\n")
        status = server.project_status()
        local = status["clusters"]["local"]
        self.assertTrue(local["refreshed"])
        self.assertEqual(local["unfinished_jobs"][0]["state"], "RUNNING")
        self.assertEqual(local["unfinished_jobs"][0]["purpose"], "baseline run")
        self.assertIsNotNone(local["unfinished_jobs"][0]["state_checked_at"])
        self.assertEqual(local["location"]["commit"], head)
        self.assertEqual(local["location"]["remote_repo"], _posix(self.remote) + "/projects/alpha")
        self.assertEqual(local["location"]["run_root"], _posix(self.remote) + "/runs/alpha")

        # squeue rows are tagged with the project that submitted them.
        self._fixture(
            "squeue.out",
            "1001|cpu|job.sh|tester|RUNNING|0:10|10:00|1|2|1G|node1\n555|cpu|other|tester|PENDING|0:00|10:00|1|2|1G|(None)\n",
        )
        rows = {row["job_id"]: row for row in server.list_jobs()["jobs"]}
        self.assertEqual((rows["1001"]["project"], rows["1001"]["tracked"]), ("alpha", True))
        self.assertFalse(rows["555"]["tracked"])

        self._fixture("sacct.out", "1001|1001|job.sh|cpu|def-pi|2|COMPLETED|0:0|00:05:00\n")
        server.project_status()
        again = server.project_status()["clusters"]["local"]
        self.assertEqual(again["unfinished_jobs"], [])
        self.assertEqual(again["job_counts"], {"COMPLETED": 1})
        self.assertEqual(server.project_history(kind="job_finished")["count"], 1)
        kinds = [event["kind"] for event in server.project_history()["events"]]
        self.assertEqual(kinds, ["job_finished", "job_submitted", "repo_synced"])

        # Logs are found through the job's recorded project, even with no active project.
        server.project_use(None)
        log = self.remote / "runs" / "alpha" / "logs" / "job.sh-1001.out"
        log.write_text("hello\n", encoding="utf-8", newline="\n")
        self.assertEqual(server.tail_job_log("1001")["content"], "hello\n")
        with self.assertRaisesRegex(ValueError, "Recorded projects: alpha"):
            server.project_status()
        self.assertEqual(server.project_list()["recorded"][0]["project"], "alpha")

    def test_project_record_failure_does_not_fail_the_tool(self) -> None:
        server.project_use("alpha")
        self._sync_project_repo()
        # A directory is not a usable SQLite file.
        server._ledger_cache = (server.get_settings(), server.Ledger(self.tmp))
        submitted = server.submit_job("cpu", "job.sh")
        self.assertTrue(submitted["submitted"])
        self.assertIn("Could not update the project record", submitted["ledger_error"])

    def test_tools_need_a_project_when_paths_use_it(self) -> None:
        with self.assertRaisesRegex(ValueError, "No project selected"):
            server.remote_git_status()
        with self.assertRaisesRegex(ValueError, "No project selected"):
            server.submit_job("cpu", "job.sh", dry_run=True)

    def test_projects_are_isolated_and_active_project_is_the_default(self) -> None:
        server.write_file("notes.txt", "alpha\n", project="alpha")
        server.project_use("beta")
        server.write_file("notes.txt", "beta\n")
        self.assertEqual((self.remote / "projects" / "alpha" / "notes.txt").read_text(), "alpha\n")
        self.assertEqual((self.remote / "projects" / "beta" / "notes.txt").read_text(), "beta\n")
        self.assertEqual(server.read_file("notes.txt", base="remote_repo")["content"], "beta\n")
        self.assertEqual(server.read_file("notes.txt", base="remote_repo", project="alpha")["content"], "alpha\n")
        with self.assertRaises(ValueError):
            server.write_file("x.txt", "x", project="../alpha")

    def test_submit_job_exports_project_paths(self) -> None:
        preview = server.submit_job("cpu", "train.sh", project="alpha", dry_run=True)
        self.assertEqual(preview["project"], "alpha")
        self.assertEqual(preview["log_dir"], _posix(self.remote) + "/runs/alpha/logs")
        self.assertEqual(preview["env"]["MCP_PROJECT"], "alpha")
        self.assertEqual(preview["env"]["MCP_REPO_DIR"], _posix(self.remote) + "/projects/alpha")
        self.assertEqual(preview["env"]["MCP_RUN_DIR"], _posix(self.remote) + "/runs/alpha")

    def test_project_on_cluster_without_templates(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not change any path"):
            server.remote_git_status(cluster="fixed", project="alpha")
        # The active project is ignored there instead of failing.
        server.project_use("alpha")
        (self.remote / "fixed").mkdir()
        written = server.write_file("a.txt", "x", cluster="fixed")
        self.assertEqual(written["path"], _posix(self.remote) + "/fixed/a.txt")

    def test_project_use_and_list(self) -> None:
        used = server.project_use("alpha")
        self.assertEqual(used["active_project"], "alpha")
        self.assertEqual(used["clusters"]["local"]["paths"]["remote_repo"], _posix(self.remote) + "/projects/alpha")
        self.assertFalse(used["clusters"]["fixed"]["project_applies"])

        (self.remote / "projects" / "beta").mkdir(parents=True)
        _git(self.remote / "projects", "init", "-q", "alpha")
        listed = server.project_list(cluster="local")
        self.assertEqual(listed["active_project"], "alpha")
        self.assertEqual(
            sorted((p["name"], p["git"]) for p in listed["found"]),
            [("alpha", True), ("beta", False)],
        )
        self.assertIsNone(server.project_list(cluster="fixed")["workspace"])

        self.assertIsNone(server.project_use(None)["active_project"])

    def test_background_tasks_are_shared_across_projects(self) -> None:
        target, roots, _ = server._env_context(server.get_settings(), None, "alpha")
        task = server._launch_task(server.get_settings(), target, roots, "demo", "echo done", "demo")
        self.assertTrue(task["task_dir"].startswith(_posix(self.remote) + "/tasks/"))
        status: dict[str, Any] = {}
        for _ in range(100):
            # No project needed: tasks_root is not project-scoped.
            status = server.remote_task_status(task["task_id"])
            if status["state"] in {"succeeded", "failed"}:
                break
            time.sleep(0.1)
        self.assertEqual(status["state"], "succeeded")

    def test_default_venv_name_follows_project(self) -> None:
        info = server.venv_info(project="alpha")
        self.assertEqual(info["venv"], _posix(self.remote) + "/venvs/alpha")
        self.assertEqual(server.venv_info()["venv"], _posix(self.remote) + "/venvs/default")

    def test_download_path_includes_project(self) -> None:
        (self.remote / "runs" / "alpha").mkdir(parents=True)
        (self.remote / "runs" / "alpha" / "out.txt").write_text("ok\n", encoding="utf-8")
        result = server.download_file("out.txt", project="alpha")
        self.assertEqual(
            Path(result["local_path"]),
            self.local / "slurm-mcp-downloads" / "local" / "alpha" / "run_root" / "out.txt",
        )


class RecordingBackend:
    name = "recording"

    def __init__(self, responses: dict[str, CommandResult]) -> None:
        self.responses = responses
        self.calls: list[list[str]] = []

    def check(self, host: str) -> CommandResult:
        return CommandResult(0, "", "")

    def run(self, host: str, argv: list[str], *, timeout: int = 60, input_data: bytes | None = None) -> CommandResult:
        self.calls.append(argv)
        return self.responses.get(argv[0], CommandResult(0, "", ""))


class SlurmToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        config_path = Path(self._tmp.name) / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "audit_log": None,
                    "clusters": {"narval": {"host": "narval", "user": "alice", "storage": {"run_root": "/scratch/alice/runs"}}},
                }
            ),
            encoding="utf-8",
        )
        server.settings = load_settings(config_path)

    def tearDown(self) -> None:
        server.settings = None
        server.backend = None
        self._tmp.cleanup()

    def test_cancel_job_is_user_scoped(self) -> None:
        server.backend = RecordingBackend({})
        result = server.cancel_job("12345")
        self.assertTrue(result["ok"])
        self.assertEqual(server.backend.calls, [["scancel", "--user=alice", "12345"]])

    def test_job_history_newest_first(self) -> None:
        rows = "1|1|a|gpu|def-pi|4|COMPLETED|0:0|00:01:00\n2|2|b|gpu|def-pi|4|FAILED|1:0|00:02:00\n"
        server.backend = RecordingBackend({"sacct": CommandResult(0, rows, "")})
        result = server.job_history(days=3)
        self.assertEqual([job["job_id"] for job in result["jobs"]], ["2", "1"])
        self.assertIn("-u", server.backend.calls[0])

    def test_project_record_can_be_disabled(self) -> None:
        config_path = Path(self._tmp.name) / "no-state.yaml"
        config_path.write_text(
            yaml.safe_dump({"state_db": None, "clusters": {"narval": {"host": "narval", "user": "alice"}}}),
            encoding="utf-8",
        )
        server.settings = load_settings(config_path)
        server.backend = RecordingBackend({})
        self.assertTrue(server.cancel_job("12345")["ok"])
        with self.assertRaisesRegex(ValueError, "disabled"):
            server.project_status()
        self.assertFalse(list(Path(self._tmp.name).glob("*.sqlite3")))

    def test_slurm_accounts_strips_cpu_gpu_suffix(self) -> None:
        rows = "def-pi_cpu|alice|1|0.1|10|0.01|0.9\ndef-pi_gpu|alice|1|0.1|10|0.01|0.9\nrrg-pi_gpu|alice|1|0.1|1|0.0|0.5\n"
        server.backend = RecordingBackend({"sshare": CommandResult(0, rows, "")})
        self.assertEqual(server.slurm_accounts()["accounts"], ["def-pi", "rrg-pi"])


if __name__ == "__main__":
    unittest.main()
