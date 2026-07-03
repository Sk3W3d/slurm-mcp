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


if __name__ == "__main__":
    unittest.main()
