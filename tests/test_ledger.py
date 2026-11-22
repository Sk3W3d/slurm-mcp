from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from slurm_mcp.ledger import ANY, Ledger, is_terminal_job_state, normalize_job_state
from slurm_mcp.slurm import sacct_jobs_argv, summarize_job_rows


def _job(ledger: Ledger, job_id: str, project: str | None = "alpha", cluster: str = "narval") -> None:
    ledger.add_job(
        cluster=cluster,
        job_id=job_id,
        project=project,
        job_name="train.sh",
        profile="cpu",
        script="train.sh",
        purpose="baseline",
        git_commit="abc123",
        git_dirty=False,
        details={"expected_stdout": "/runs/logs/train.sh-%s.out" % job_id},
    )


class LedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.ledger = Ledger(Path(self._tmp.name) / "nested" / "state.sqlite3")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_job_states_cache_and_finish_event_logged_once(self) -> None:
        _job(self.ledger, "100")
        job = self.ledger.get_job("narval", "100")
        self.assertEqual(job["state"], "SUBMITTED")
        self.assertFalse(job["git_dirty"])
        self.assertEqual(job["details"]["expected_stdout"], "/runs/logs/train.sh-100.out")

        self.ledger.update_job_states("narval", {"100": {"state": "RUNNING", "elapsed": "00:01:00"}})
        self.assertEqual([j["job_id"] for j in self.ledger.jobs(project="alpha", active_only=True)], ["100"])

        for _ in range(2):
            self.ledger.update_job_states("narval", {"100": {"state": "COMPLETED", "exit_code": "0:0"}})
        job = self.ledger.get_job("narval", "100")
        self.assertTrue(job["terminal"])
        self.assertEqual(job["elapsed"], "00:01:00")
        finished = self.ledger.events(project="alpha", kind="job_finished")
        self.assertEqual(len(finished), 1)
        self.assertIn("COMPLETED", finished[0]["summary"])
        self.assertEqual(self.ledger.jobs(project="alpha", active_only=True), [])

    def test_unknown_jobs_are_not_stored(self) -> None:
        self.assertEqual(self.ledger.update_job_states("narval", {"999": {"state": "RUNNING"}}), 0)
        self.assertIsNone(self.ledger.get_job("narval", "999"))

    def test_project_filters_and_summaries(self) -> None:
        _job(self.ledger, "1", project="alpha")
        _job(self.ledger, "2", project="beta", cluster="rorqual")
        _job(self.ledger, "3", project=None)
        self.ledger.add_event("job_submitted", "a", project="alpha", cluster="narval")
        self.ledger.add_event("job_submitted", "b", project="beta", cluster="rorqual")
        self.assertEqual([j["job_id"] for j in self.ledger.jobs(project=None)], ["3"])
        self.assertEqual(len(self.ledger.jobs(project=ANY)), 3)
        self.assertEqual(self.ledger.clusters_for("beta"), ["rorqual"])
        self.assertEqual(self.ledger.job_projects("narval", ["1", "3", "404"]), {"1": "alpha", "3": None})
        self.assertEqual(self.ledger.job_state_counts(project="alpha"), {"SUBMITTED": 1})
        projects = {row["project"]: row for row in self.ledger.projects()}
        self.assertEqual(set(projects), {"alpha", "beta"})
        self.assertEqual(projects["beta"]["clusters"], ["rorqual"])
        self.assertEqual(projects["alpha"]["unfinished_jobs"], 1)

    def test_job_projects_handles_many_ids(self) -> None:
        _job(self.ledger, "1500")
        ids = [str(i) for i in range(2000)]
        self.assertEqual(self.ledger.job_projects("narval", ids), {"1500": "alpha"})

    def test_locations_merge_and_ignore_none(self) -> None:
        self.ledger.update_location("alpha", "narval", remote_repo="/r", commit="aaa")
        self.ledger.update_location("alpha", "narval", commit="bbb", venv=None)
        location = self.ledger.locations("alpha")["narval"]
        self.assertEqual(location["remote_repo"], "/r")
        self.assertEqual(location["commit"], "bbb")
        self.assertNotIn("venv", location)
        self.assertIn("updated_at", location)

    def test_task_finish_event(self) -> None:
        self.ledger.add_task(cluster="narval", task_id="venv-1", project="alpha", kind="venv", description="setup_venv")
        self.ledger.update_task_state("narval", "venv-1", "running", None)
        self.assertEqual(self.ledger.events(kind="task_finished"), [])
        self.ledger.update_task_state("narval", "venv-1", "failed", 2)
        self.ledger.update_task_state("narval", "venv-1", "failed", 2)
        events = self.ledger.events(kind="task_finished")
        self.assertEqual(len(events), 1)
        self.assertIn("exit 2", events[0]["summary"])


class JobStateTests(unittest.TestCase):
    def test_normalize(self) -> None:
        self.assertEqual(normalize_job_state("CANCELLED by 1234"), "CANCELLED")
        self.assertIsNone(normalize_job_state(""))
        self.assertTrue(is_terminal_job_state("TIMEOUT"))
        self.assertFalse(is_terminal_job_state("PENDING"))

    def test_summarize_single_job_ignores_steps(self) -> None:
        rows = [
            {"job_id": "7", "state": "COMPLETED", "exit_code": "0:0", "elapsed": "00:05:00"},
            {"job_id": "7.batch", "state": "COMPLETED", "exit_code": "0:0", "elapsed": "00:05:00"},
            {"job_id": "8", "state": "RUNNING", "exit_code": "0:0", "elapsed": "00:01:00"},
        ]
        summary = summarize_job_rows(rows, ["7"])
        self.assertEqual(summary, {"7": {"state": "COMPLETED", "exit_code": "0:0", "elapsed": "00:05:00", "array_tasks": None}})

    def test_summarize_array_jobs(self) -> None:
        running = [
            {"job_id": "9_0", "state": "COMPLETED"},
            {"job_id": "9_1", "state": "RUNNING"},
            {"job_id": "9_[2-5]", "state": "PENDING"},
        ]
        summary = summarize_job_rows(running, ["9"])["9"]
        self.assertEqual(summary["state"], "RUNNING")
        self.assertEqual(summary["array_tasks"], {"COMPLETED": 1, "RUNNING": 1, "PENDING": 1})

        done = [{"job_id": "9_0", "state": "COMPLETED"}, {"job_id": "9_1", "state": "FAILED"}]
        self.assertEqual(summarize_job_rows(done, ["9"])["9"]["state"], "MIXED")
        self.assertTrue(is_terminal_job_state("MIXED"))

    def test_sacct_jobs_argv(self) -> None:
        argv = sacct_jobs_argv(["1", "2"])
        self.assertEqual(argv[:4], ["sacct", "-X", "-j", "1,2"])
        with self.assertRaises(ValueError):
            sacct_jobs_argv([])


if __name__ == "__main__":
    unittest.main()
