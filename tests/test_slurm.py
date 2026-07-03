from __future__ import annotations

import unittest

from slurm_mcp.slurm import (
    build_sbatch_argv,
    parse_sacct,
    parse_sbatch_job_id,
    parse_squeue,
)
from slurm_mcp.validators import validate_job_name, validate_remote_relative_path


class SlurmParsingTests(unittest.TestCase):
    def test_parse_squeue_rows(self) -> None:
        output = "123|gpu|train|alice|RUNNING|00:01:02|01:00:00|1|8|64G|node001\n"

        jobs = parse_squeue(output)

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["job_id"], "123")
        self.assertEqual(jobs[0]["state"], "RUNNING")
        self.assertEqual(jobs[0]["reason"], "node001")

    def test_parse_sacct_rows(self) -> None:
        output = (
            "123|123|train|gpu|def-pi|8|COMPLETED|0:0|00:10:00|01:00:00|"
            "2026-01-01T00:00:00|2026-01-01T00:01:00|2026-01-01T00:11:00|"
            "node001|billing=8,cpu=8,mem=64G|billing=8,cpu=8,mem=64G|1024K\n"
        )

        records = parse_sacct(output)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["job_id"], "123")
        self.assertEqual(records[0]["state"], "COMPLETED")
        self.assertEqual(records[0]["exit_code"], "0:0")

    def test_parse_sbatch_job_id(self) -> None:
        self.assertEqual(parse_sbatch_job_id("Submitted batch job 12345"), "12345")


class SlurmSubmissionTests(unittest.TestCase):
    def test_build_sbatch_argv_uses_profile_and_log_dir(self) -> None:
        argv = build_sbatch_argv(
            {
                "time": "00:30:00",
                "gpus": 1,
                "gpu_type": "h100",
                "cpus_per_task": 4,
                "mem": "32G",
            },
            "scripts/train.slurm",
            job_name="smoke",
            script_args=["--lr", "1e-4"],
            cluster_slurm={"gpu_request_style": "gres"},
            log_dir="/scratch/alice/paper/runs/logs",
        )

        self.assertEqual(argv[0], "sbatch")
        self.assertIn("--job-name=smoke", argv)
        self.assertIn("--time=00:30:00", argv)
        self.assertIn("--gres=gpu:h100:1", argv)
        self.assertIn("--output=/scratch/alice/paper/runs/logs/%x-%j.out", argv)
        self.assertEqual(argv[-3:], ["scripts/train.slurm", "--lr", "1e-4"])

    def test_build_sbatch_argv_rejects_unknown_profile_keys(self) -> None:
        with self.assertRaises(ValueError):
            build_sbatch_argv({"time": "00:30:00", "conda_env": "base"}, "train.slurm")

    def test_validate_remote_relative_path_rejects_escape(self) -> None:
        with self.assertRaises(ValueError):
            validate_remote_relative_path("../train.slurm", "script")

    def test_validate_job_name_rejects_spaces(self) -> None:
        with self.assertRaises(ValueError):
            validate_job_name("bad job name")


if __name__ == "__main__":
    unittest.main()
