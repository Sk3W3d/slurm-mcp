from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from slurm_mcp.audit import preview_text, write_audit_event


class AuditTests(unittest.TestCase):
    def test_write_audit_event_appends_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"

            write_audit_event(path, {"tool": "list_jobs", "returncode": 0})

            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(records[0]["tool"], "list_jobs")
            self.assertEqual(records[0]["returncode"], 0)
            self.assertIn("timestamp", records[0])

    def test_preview_text_truncates(self) -> None:
        preview = preview_text("abcdef", limit=3)

        self.assertEqual(preview, "abc... <truncated 3 chars>")


if __name__ == "__main__":
    unittest.main()
