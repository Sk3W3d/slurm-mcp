from __future__ import annotations

import unittest

from slurm_mcp.remote_git import (
    git_fetch_dry_run_argv,
    git_pull_argv,
    git_rev_argv,
    git_status_argv,
    parse_git_status,
)


class RemoteGitTests(unittest.TestCase):
    def test_parse_clean_branch_status(self) -> None:
        status = parse_git_status("## main...origin/main\n")

        self.assertEqual(status["branch"], "main")
        self.assertEqual(status["upstream"], "origin/main")
        self.assertEqual(status["ahead"], 0)
        self.assertEqual(status["behind"], 0)
        self.assertEqual(status["entries"], [])
        self.assertTrue(status["clean"])

    def test_parse_ahead_behind_and_entries(self) -> None:
        output = "## main...origin/main [ahead 1, behind 2]\n M train.py\n?? notes.txt\n"

        status = parse_git_status(output)

        self.assertEqual(status["branch"], "main")
        self.assertEqual(status["upstream"], "origin/main")
        self.assertEqual(status["ahead"], 1)
        self.assertEqual(status["behind"], 2)
        self.assertFalse(status["clean"])
        self.assertEqual(status["entries"][0]["worktree"], "M")
        self.assertEqual(status["entries"][0]["path"], "train.py")
        self.assertEqual(status["entries"][1]["index"], "?")
        self.assertEqual(status["entries"][1]["path"], "notes.txt")

    def test_git_argv_builders(self) -> None:
        repo = "/scratch/alice/paper"

        self.assertEqual(git_status_argv(repo), ["git", "-C", repo, "status", "--short", "--branch"])
        self.assertEqual(git_rev_argv(repo), ["git", "-C", repo, "rev-parse", "HEAD"])
        self.assertEqual(
            git_rev_argv(repo, short=True),
            ["git", "-C", repo, "rev-parse", "--short=12", "HEAD"],
        )
        self.assertEqual(git_fetch_dry_run_argv(repo), ["git", "-C", repo, "fetch", "--dry-run", "--verbose"])
        self.assertEqual(git_pull_argv(repo), ["git", "-C", repo, "pull", "--ff-only"])


if __name__ == "__main__":
    unittest.main()
