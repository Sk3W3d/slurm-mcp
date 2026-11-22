from __future__ import annotations

import json
import shlex
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from slurm_mcp import huggingface as hf
from slurm_mcp import python_env, remote_tasks
from slurm_mcp.local_files import check_transfer_size, resolve_local_path
from slurm_mcp.remote_git import (
    git_bundle_fetch_argv,
    git_clone_argv,
    validate_git_ref,
    validate_git_url,
)
from slurm_mcp.slurm import build_sbatch_argv, export_env_option, merge_env, merge_job_options, scancel_argv


class RemoteTaskTests(unittest.TestCase):
    def test_task_id_round_trip(self) -> None:
        task_id = remote_tasks.new_task_id("venv", datetime(2026, 10, 10, 12, 30, 0, tzinfo=timezone.utc))
        self.assertTrue(task_id.startswith("venv-20261010-123000-"))
        self.assertEqual(remote_tasks.validate_task_id(task_id), task_id)
        self.assertEqual(
            remote_tasks.task_dir("/scratch/u/runs/", task_id),
            f"/scratch/u/runs/mcp-tasks/{task_id}",
        )

    def test_task_id_rejects_traversal(self) -> None:
        for bad in ("../etc", "venv-20261010-123000-abc", "venv/../../x", ""):
            with self.assertRaises(ValueError):
                remote_tasks.validate_task_id(bad)

    def test_wrapped_script_records_exit_code_and_description(self) -> None:
        script = remote_tasks.wrap_task_script("/s/runs/mcp-tasks/t", "echo hi", description="setup\nvenv")
        lines = script.splitlines()
        self.assertEqual(lines[1], "# slurm-mcp task: setup venv")
        self.assertIn("TASK_DIR=/s/runs/mcp-tasks/t", script)
        self.assertIn('exit_code"\' EXIT', script)
        self.assertIn("echo hi", script)

    def test_parse_status_states(self) -> None:
        done = remote_tasks.parse_task_status(
            "exit_code=0\npid=login1 42\nhost_now=login2\nalive=0\ndescription=x\n__LOG__\nok\n"
        )
        self.assertEqual(done["state"], "succeeded")
        self.assertEqual(done["log_tail"], "ok\n")

        failed = remote_tasks.parse_task_status("exit_code=2\npid=login1 42\nhost_now=login1\nalive=0\n__LOG__\n")
        self.assertEqual(failed["state"], "failed")
        self.assertEqual(failed["exit_code"], 2)

        running = remote_tasks.parse_task_status("exit_code=\npid=login1 42\nhost_now=login1\nalive=1\n__LOG__\n")
        self.assertEqual(running["state"], "running")

        lost = remote_tasks.parse_task_status("exit_code=\npid=login1 42\nhost_now=login1\nalive=0\n__LOG__\n")
        self.assertEqual(lost["state"], "lost")

        elsewhere = remote_tasks.parse_task_status("exit_code=\npid=login1 42\nhost_now=login3\nalive=0\n__LOG__\n")
        self.assertEqual(elsewhere["state"], "running_or_unknown")

        self.assertEqual(remote_tasks.parse_task_status("missing=1\n")["state"], "missing")

    def test_parse_task_list_newest_first(self) -> None:
        tasks = remote_tasks.parse_task_list(
            "venv-20261001-000000-aaaaaa|0|setup_venv a\nhf-20261005-000000-bbbbbb||hf_download model x\n"
        )
        self.assertEqual([task["task_id"] for task in tasks], ["hf-20261005-000000-bbbbbb", "venv-20261001-000000-aaaaaa"])
        self.assertEqual(tasks[0]["state"], "running_or_unknown")
        self.assertEqual(tasks[1]["state"], "succeeded")


class PythonEnvTests(unittest.TestCase):
    def test_resolve_modules_orders_python_first_and_dedupes(self) -> None:
        modules = python_env.resolve_modules({"module": "python/3.11", "modules": ["arrow", "python/3.11"]}, None, ["opencv"])
        self.assertEqual(modules, ["python/3.11", "arrow", "opencv"])
        self.assertEqual(python_env.resolve_modules({}, None, None), ["python"])

    def test_rejects_unsafe_input(self) -> None:
        with self.assertRaises(ValueError):
            python_env.validate_module("python; rm -rf ~")
        with self.assertRaises(ValueError):
            python_env.validate_packages(["torch && curl evil"])
        with self.assertRaises(ValueError):
            python_env.validate_packages(["git+https://example.com/x.git"])
        with self.assertRaises(ValueError):
            python_env.venv_path("/project/venvs", "../escape")

    def test_accepts_common_requirement_specifiers(self) -> None:
        packages = ["torch", "numpy>=1.26", "transformers==4.46.0", "accelerate[torch]", "scipy>=1.11,<2"]
        self.assertEqual(python_env.validate_packages(packages), packages)

    def test_venv_script_uses_wheelhouse(self) -> None:
        script = python_env.build_venv_script(
            venv="/project/def-x/venvs/default",
            modules=["python/3.11", "arrow"],
            remote_repo="/scratch/u/repo",
            requirements="requirements.txt",
            packages=["numpy>=1.26"],
            no_index=True,
            recreate=False,
        )
        self.assertIn("module load python/3.11 arrow", script)
        self.assertIn("virtualenv --no-download", script)
        self.assertIn("python -m pip install --no-index -r requirements.txt", script)
        self.assertIn("python -m pip install --no-index 'numpy>=1.26'", script)
        self.assertNotIn("rm -rf", script)

    def test_venv_script_can_use_pypi_and_recreate(self) -> None:
        script = python_env.build_venv_script(
            venv="/v/env",
            modules=["python"],
            remote_repo=None,
            requirements=None,
            packages=["flash-attn"],
            no_index=False,
            recreate=True,
        )
        self.assertIn('rm -rf -- "$VENV"', script)
        self.assertIn("python -m pip install flash-attn", script)
        self.assertNotIn("--no-index flash-attn", script)

    def test_requirements_need_repo(self) -> None:
        with self.assertRaises(ValueError):
            python_env.build_venv_script(
                venv="/v/env", modules=[], remote_repo=None, requirements="r.txt", packages=[], no_index=True, recreate=False
            )

    def test_avail_wheels_strips_specifiers(self) -> None:
        argv = python_env.avail_wheels_argv(["torch>=2", "numpy"], "3.11")
        self.assertEqual(argv, ["bash", "-lc", "avail_wheels torch numpy --python 3.11"])

    def test_parse_venv_inspect(self) -> None:
        missing = python_env.parse_venv_inspect("exists=0\n")
        self.assertFalse(missing["exists"])
        found = python_env.parse_venv_inspect(
            "exists=1\nmodules=python/3.11 arrow \npython=/cvmfs/python3.11\n__FREEZE__\nnumpy==1.26.4\ntorch==2.4.0\n"
        )
        self.assertEqual(found["modules"], ["python/3.11", "arrow"])
        self.assertEqual(found["packages"], ["numpy==1.26.4", "torch==2.4.0"])


class HuggingFaceTests(unittest.TestCase):
    def test_job_env_offline_only_without_compute_internet(self) -> None:
        self.assertEqual(hf.job_env(None, False), {})
        offline = hf.job_env("/project/hf", False)
        self.assertEqual(offline["HF_HOME"], "/project/hf")
        self.assertEqual(offline["HF_HUB_OFFLINE"], "1")
        self.assertEqual(hf.job_env("/project/hf", True), {"HF_HOME": "/project/hf"})

    def test_download_script_passes_arguments_as_json(self) -> None:
        script = hf.build_download_script(
            hf_home="/project/hf",
            modules=["python/3.11"],
            venv="/project/venvs/_slurm-mcp-hf",
            create_venv=True,
            repo_id="meta-llama/Llama-3.1-8B",
            repo_type="model",
            revision="main",
            allow_patterns=["*.json", "*.safetensors"],
            ignore_patterns=None,
        )
        self.assertIn("export HF_HOME=/project/hf", script)
        self.assertIn("unset HF_HUB_OFFLINE", script)
        export_line = next(line for line in script.splitlines() if line.startswith("export SLURM_MCP_HF_ARGS="))
        payload = json.loads(shlex.split(export_line.split("=", 1)[1])[0])
        self.assertEqual(
            payload,
            {
                "allow_patterns": ["*.json", "*.safetensors"],
                "repo_id": "meta-llama/Llama-3.1-8B",
                "repo_type": "model",
                "revision": "main",
            },
        )

    def test_rejects_bad_repo_input(self) -> None:
        for bad in ("../x", "org/name; rm", "a/b/c", ""):
            with self.assertRaises(ValueError):
                hf.validate_repo_id(bad)
        with self.assertRaises(ValueError):
            hf.validate_repo_type("weights")
        with self.assertRaises(ValueError):
            hf.validate_patterns(["$(whoami)"], "allow_patterns")

    def test_parse_cache_list(self) -> None:
        entries = hf.parse_cache_list(
            "models--meta-llama--Llama-3.1-8B|main=abc123 |abc123 \ndatasets--squad||\n"
        )
        self.assertEqual(entries[0]["repo_id"], "meta-llama/Llama-3.1-8B")
        self.assertEqual(entries[0]["repo_type"], "model")
        self.assertEqual(entries[0]["refs"], {"main": "abc123"})
        self.assertEqual(entries[1], {"repo_type": "dataset", "repo_id": "squad", "refs": {}, "snapshots": []})


class LocalFileTests(unittest.TestCase):
    def test_resolve_local_path_stays_in_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            self.assertEqual(resolve_local_path("a/b.txt", (root,)), root / "a" / "b.txt")
            self.assertEqual(resolve_local_path(None, (root,)), root)
            with self.assertRaises(ValueError):
                resolve_local_path("../outside.txt", (root,))
            with self.assertRaises(ValueError):
                resolve_local_path(str(root.parent), (root,))

    def test_transfer_size_limit(self) -> None:
        check_transfer_size(10, 10, "file")
        with self.assertRaises(ValueError):
            check_transfer_size(11, 10, "file")


class GitHelperTests(unittest.TestCase):
    def test_validate_ref(self) -> None:
        for good in ("main", "feature/x-1", "v1.2.3", "a1b2c3d"):
            self.assertEqual(validate_git_ref(good), good)
        for bad in ("-rf", "a..b", "x;y", "main.lock", "@{-1}", ""):
            with self.assertRaises(ValueError):
                validate_git_ref(bad)

    def test_validate_url(self) -> None:
        self.assertEqual(validate_git_url("https://github.com/o/r.git"), "https://github.com/o/r.git")
        self.assertEqual(validate_git_url("git@github.com:o/r.git"), "git@github.com:o/r.git")
        for bad in ("--upload-pack=x", "file:///etc", "https://x/y z", "ext::sh -c id"):
            with self.assertRaises(ValueError):
                validate_git_url(bad)

    def test_clone_and_bundle_argv(self) -> None:
        self.assertEqual(
            git_clone_argv("https://github.com/o/r.git", "/scratch/u/r", "dev"),
            ["git", "clone", "--branch", "dev", "--", "https://github.com/o/r.git", "/scratch/u/r"],
        )
        self.assertEqual(
            git_bundle_fetch_argv("/r", "/r/.git/b.bundle", "main"),
            ["git", "-C", "/r", "fetch", "/r/.git/b.bundle", "main:main"],
        )


class SlurmExtrasTests(unittest.TestCase):
    def test_export_env_option(self) -> None:
        self.assertEqual(
            export_env_option({"HF_HOME": "/p/hf", "HF_HUB_OFFLINE": "1"}),
            "--export=ALL,HF_HOME=/p/hf,HF_HUB_OFFLINE=1",
        )
        with self.assertRaises(ValueError):
            export_env_option({"BAD-NAME": "1"})
        with self.assertRaises(ValueError):
            export_env_option({"X": "a,b"})

    def test_merge_env_none_removes(self) -> None:
        self.assertEqual(merge_env({"A": "1", "B": "2"}, {"B": None, "C": 3}), {"A": "1", "C": "3"})

    def test_overrides_and_default_account(self) -> None:
        merged = merge_job_options({"time": "01:00:00", "gpus": 1}, {"time": "02:00:00", "array": "0-3"}, "def-pi")
        self.assertEqual(merged, {"time": "02:00:00", "gpus": 1, "array": "0-3", "account": "def-pi"})
        argv = build_sbatch_argv(merged, "run.sh", env={"HF_HOME": "/p/hf"})
        self.assertIn("--array=0-3", argv)
        self.assertIn("--account=def-pi", argv)
        self.assertIn("--export=ALL,HF_HOME=/p/hf", argv)
        self.assertEqual(argv[-1], "run.sh")

    def test_cpu_account_used_only_for_jobs_without_gpus(self) -> None:
        accounts = {"default_account": "rrg-pi", "cpu_account": "def-pi"}
        self.assertEqual(merge_job_options({"cpus_per_task": 2}, None, **accounts)["account"], "def-pi")
        self.assertEqual(merge_job_options({"gpus": 1}, None, **accounts)["account"], "rrg-pi")
        self.assertEqual(merge_job_options({"gres": "gpu:a100:1"}, None, **accounts)["account"], "rrg-pi")
        # Turning GPUs off per call switches to the CPU account; an explicit account always wins.
        self.assertEqual(merge_job_options({"gpus": 1}, {"gpus": 0}, **accounts)["account"], "def-pi")
        self.assertEqual(merge_job_options({}, {"account": "def-other"}, **accounts)["account"], "def-other")
        self.assertEqual(merge_job_options({}, None, default_account="rrg-pi")["account"], "rrg-pi")

    def test_profile_env_key_is_allowed(self) -> None:
        argv = build_sbatch_argv({"time": "00:10:00", "env": {"A": "1"}}, "run.sh")
        self.assertNotIn("--env", " ".join(argv))

    def test_unknown_override_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_sbatch_argv(merge_job_options({}, {"wrap": "rm -rf /"}), "run.sh")

    def test_scancel_scoped_to_user(self) -> None:
        self.assertEqual(scancel_argv("123", "alice"), ["scancel", "--user=alice", "123"])


if __name__ == "__main__":
    unittest.main()
