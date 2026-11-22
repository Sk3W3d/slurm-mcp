from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import yaml  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env setup
    raise unittest.SkipTest("PyYAML is required for config tests.") from exc

from slurm_mcp.config import load_settings


class ConfigTests(unittest.TestCase):
    def test_cluster_info_merges_defaults_and_cluster_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            (root / "docs" / "alliance.md").write_text("Use modules.", encoding="utf-8")
            config = root / "config.yaml"
            config.write_text(
                """
default_cluster: narval
ssh_command:
  - ssh
cluster_defaults:
  docs:
    - docs/alliance.md
  policies:
    allow_conda: false
    no_heavy_login_node_commands: true
  setup_hints:
    python_virtualenv:
      - module load python
      - python -m venv $VENV_PATH
  slurm:
    submit_command: sbatch
  storage:
    preferred_work_root: scratch
clusters:
  narval:
    host: narval
    user: alice
    docs:
      - https://docs.alliancecan.ca/wiki/Narval
    policies:
      max_login_node_seconds: 30
    storage:
      remote_repo: /scratch/alice/paper
      run_root: /scratch/alice/paper/runs
profiles:
  debug_gpu:
    time: "00:30:00"
    gpus: 1
""",
                encoding="utf-8",
            )

            settings = load_settings(config)
            info = settings.cluster_info("narval")

            self.assertEqual(info["name"], "narval")
            self.assertEqual(
                info["docs"],
                ["docs/alliance.md", "https://docs.alliancecan.ca/wiki/Narval"],
            )
            self.assertEqual(info["policies"]["allow_conda"], False)
            self.assertEqual(info["policies"]["max_login_node_seconds"], 30)
            self.assertEqual(info["storage"]["remote_repo"], "/scratch/alice/paper")
            self.assertEqual(info["profiles"]["debug_gpu"]["gpus"], 1)
            self.assertEqual(
                info["setup_hints"]["python_virtualenv"],
                ["module load python", "python -m venv $VENV_PATH"],
            )

    def test_read_doc_source_reads_relative_docs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            (root / "docs" / "alliance.md").write_text("Use virtualenv.", encoding="utf-8")
            config = root / "config.yaml"
            config.write_text(
                """
clusters:
  narval:
    host: narval
cluster_defaults:
  docs:
    - docs/alliance.md
""",
                encoding="utf-8",
            )

            settings = load_settings(config)
            doc = settings.read_doc_source("docs/alliance.md")

            self.assertEqual(doc["kind"], "local_markdown")
            self.assertEqual(doc["content"], "Use virtualenv.")

    def test_read_doc_source_rejects_paths_outside_config_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.yaml"
            config.write_text(
                """
clusters:
  narval:
    host: narval
""",
                encoding="utf-8",
            )

            settings = load_settings(config)

            with self.assertRaises(ValueError):
                settings.read_doc_source("../secret.md")


def _write_config(root: Path, text: str) -> Path:
    config = root / "config.yaml"
    config.write_text(text, encoding="utf-8")
    return config


_PROJECT_CONFIG = """
default_cluster: narval
cluster_defaults:
  storage:
    workspace_root: /scratch/{user}/projects
    remote_repo: "{workspace_root}/{project}"
    run_root: /scratch/{user}/runs/{project}
    venv_root: /home/{user}/venvs
    hf_home: /project/def-pi/{user}/hf
clusters:
  narval:
    host: narval
    user: alice
  fixed:
    host: fixed
    user: alice
    storage:
      remote_repo: /scratch/alice/only
      run_root: /scratch/alice/only/runs
projects:
  old-paper:
    description: Existing checkout outside the workspace.
    clusters:
      narval:
        storage:
          remote_repo: /project/def-pi/alice/old-paper
"""


class ProjectTemplateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.settings = load_settings(_write_config(Path(self._tmp.name), _PROJECT_CONFIG))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_project_fills_templates_including_nested_keys(self) -> None:
        storage = self.settings.storage_for("narval", "llm-eval")
        self.assertEqual(storage.values["remote_repo"], "/scratch/alice/projects/llm-eval")
        self.assertEqual(storage.values["run_root"], "/scratch/alice/runs/llm-eval")
        self.assertEqual(storage.values["venv_root"], "/home/alice/venvs")
        self.assertEqual(storage.pending_project, ())
        self.assertTrue(storage.project_applies)

    def test_without_project_project_paths_are_pending_and_shared_paths_resolve(self) -> None:
        storage = self.settings.storage_for("narval", None)
        self.assertEqual(storage.pending_project, ("remote_repo", "run_root"))
        self.assertNotIn("remote_repo", storage.values)
        self.assertEqual(storage.values["hf_home"], "/project/def-pi/alice/hf")
        self.assertFalse(storage.project_applies)

    def test_project_override_wins_for_one_cluster(self) -> None:
        info = self.settings.cluster_info("narval", project="old-paper")
        self.assertEqual(info["storage"]["remote_repo"], "/project/def-pi/alice/old-paper")
        self.assertEqual(info["storage"]["run_root"], "/scratch/alice/runs/old-paper")
        self.assertEqual(info["project"], "old-paper")
        self.assertEqual(self.settings.projects["old-paper"].description, "Existing checkout outside the workspace.")

    def test_cluster_without_project_template_reports_project_not_applying(self) -> None:
        info = self.settings.cluster_info("fixed", project="llm-eval")
        self.assertFalse(info["project_applies"])
        self.assertIsNone(info["project"])
        self.assertEqual(info["storage"]["remote_repo"], "/scratch/alice/only")

    def test_rejects_bad_project_names(self) -> None:
        for bad in ("../x", "a/b", "", ".hidden", "x" * 65):
            with self.assertRaises(ValueError):
                self.settings.storage_for("narval", bad)

    def test_config_errors(self) -> None:
        cases = {
            "unknown placeholder": "    remote_repo: /scratch/{nope}/x\n",
            "cycle": "    remote_repo: '{run_root}/a'\n    run_root: '{remote_repo}/b'\n",
            "no user": "    remote_repo: /scratch/{user}/x\n",
        }
        for label, storage in cases.items():
            with self.subTest(label), tempfile.TemporaryDirectory() as tmp:
                text = "clusters:\n  c:\n    host: c\n    storage:\n" + storage
                settings = load_settings(_write_config(Path(tmp), text))
                with self.assertRaises(ValueError):
                    settings.storage_for("c", "p")

    def test_unknown_or_misplaced_settings_are_errors(self) -> None:
        cases = {
            "under clusters.c.storage": "clusters:\n  c:\n    host: c\n    workspace_root: /x\n",
            "Did you mean 'default_account'": "clusters:\n  c:\n    host: c\n    slurm:\n      defualt_account: def-x\n",
            "Did you mean 'local_roots'": "local_root: [D:/x]\nclusters:\n  c:\n    host: c\n",
            "cluster_defaults: unknown setting 'storge'": "cluster_defaults:\n  storge: {}\nclusters:\n  c:\n    host: c\n",
            "cluster_defaults.python": "cluster_defaults:\n  python:\n    venv: x\nclusters:\n  c:\n    host: c\n",
        }
        for message, text in cases.items():
            with self.subTest(message), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, message):
                    load_settings(_write_config(Path(tmp), text))

    def test_shipped_example_config_loads(self) -> None:
        example = Path(__file__).resolve().parents[1] / "config.example.yaml"
        settings = load_settings(example)
        self.assertIn("narval", settings.clusters)

    def test_reserved_storage_keys_and_unknown_project_cluster_rejected(self) -> None:
        bad_configs = [
            "clusters:\n  c:\n    host: c\n    storage:\n      project: /x\n",
            "clusters:\n  c:\n    host: c\nprojects:\n  p:\n    clusters:\n      other:\n        storage: {remote_repo: /x}\n",
            "clusters:\n  c:\n    host: c\nprojects:\n  p:\n    remote_repo: /x\n",
            "clusters:\n  c:\n    host: c\ndefault_project: ../x\n",
        ]
        for text in bad_configs:
            with self.subTest(text), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):
                    load_settings(_write_config(Path(tmp), text))


if __name__ == "__main__":
    unittest.main()
