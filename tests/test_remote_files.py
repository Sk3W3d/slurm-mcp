from __future__ import annotations

import unittest

from slurm_mcp.remote_files import (
    allowed_roots_from_storage,
    classify_job_log,
    parse_file_listing,
    parse_grep_output,
    resolve_remote_path,
    select_job_log,
)


class RemoteFileTests(unittest.TestCase):
    def test_resolve_relative_path_under_run_root(self) -> None:
        roots = allowed_roots_from_storage(
            {
                "remote_repo": "/scratch/alice/paper",
                "run_root": "/scratch/alice/paper/runs",
            }
        )

        resolved = resolve_remote_path("logs/train-123.out", roots, base="run_root")

        self.assertEqual(resolved.path, "/scratch/alice/paper/runs/logs/train-123.out")
        self.assertEqual(resolved.root_name, "run_root")
        self.assertEqual(resolved.relative, "logs/train-123.out")

    def test_resolve_absolute_path_under_remote_repo(self) -> None:
        roots = allowed_roots_from_storage(
            {
                "remote_repo": "/scratch/alice/paper",
                "run_root": "/scratch/alice/paper/runs",
            }
        )

        resolved = resolve_remote_path("/scratch/alice/paper/scripts/train.slurm", roots)

        self.assertEqual(resolved.root_name, "remote_repo")
        self.assertEqual(resolved.relative, "scripts/train.slurm")

    def test_resolve_absolute_path_prefers_more_specific_root(self) -> None:
        roots = allowed_roots_from_storage(
            {
                "remote_repo": "/scratch/alice/paper",
                "run_root": "/scratch/alice/paper/runs",
            }
        )

        resolved = resolve_remote_path("/scratch/alice/paper/runs/logs/train.out", roots)

        self.assertEqual(resolved.root_name, "run_root")
        self.assertEqual(resolved.relative, "logs/train.out")

    def test_resolve_rejects_outside_roots(self) -> None:
        roots = allowed_roots_from_storage({"run_root": "/scratch/alice/paper/runs"})

        with self.assertRaises(ValueError):
            resolve_remote_path("/scratch/alice/other/file.txt", roots)

    def test_resolve_rejects_relative_escape(self) -> None:
        roots = allowed_roots_from_storage({"run_root": "/scratch/alice/paper/runs"})

        with self.assertRaises(ValueError):
            resolve_remote_path("../secrets.txt", roots)

    def test_parse_file_listing_sorts_newest_first(self) -> None:
        roots = allowed_roots_from_storage({"run_root": "/scratch/alice/paper/runs"})
        output = (
            "100.0|10|/scratch/alice/paper/runs/logs/old.out\n"
            "200.0|20|/scratch/alice/paper/runs/logs/new.out\n"
        )

        files = parse_file_listing(output, roots, limit=10)

        self.assertEqual(files[0]["relative_path"], "logs/new.out")
        self.assertEqual(files[0]["size_bytes"], 20)
        self.assertEqual(files[1]["relative_path"], "logs/old.out")

    def test_parse_grep_output_limits_and_normalizes(self) -> None:
        roots = allowed_roots_from_storage({"run_root": "/scratch/alice/paper/runs"})
        output = (
            "/scratch/alice/paper/runs/logs/a.err:4:Traceback here\n"
            "/scratch/alice/paper/runs/logs/b.err:8:Traceback again\n"
        )

        matches = parse_grep_output(output, roots, limit=1)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["relative_path"], "logs/a.err")
        self.assertEqual(matches[0]["line"], 4)

    def test_classify_and_select_job_logs(self) -> None:
        files = [
            {"path": "/runs/logs/train-123.err", "stream": classify_job_log("/runs/logs/train-123.err", "123")},
            {"path": "/runs/logs/train-123.out", "stream": classify_job_log("/runs/logs/train-123.out", "123")},
        ]

        selected = select_job_log(files, "stdout")

        self.assertEqual(selected["path"], "/runs/logs/train-123.out")


if __name__ == "__main__":
    unittest.main()
