from __future__ import annotations

import unittest

from slurm_mcp.quota import parse_diskusage_report

NARVAL = """                            Description                Space         # of files
                       /home (user jzj)          33GB/  50GB         210K/ 500K
                    /scratch (user jzj)        6551MB/  20TB          50K/1000K
         /project (project def-mijungp)         479GB/1000GB     ->  497K/ 500K
         /project (project rrg-mijungp)        4579GB/ 200TB         419K/ 500K
        /nearline (project def-mijungp)         515KB/1000GB          21 /5000

-> Usage near or over quota (warning threshold at 95% of the quota)

Disk usage can be explored using the following commands
diskusage_explorer /home/jzj                 # Last update: 2026-11-22 09:50:27
"""


class QuotaParseTests(unittest.TestCase):
    def test_parses_rows_and_flags(self) -> None:
        rows = {row["location"]: row for row in parse_diskusage_report(NARVAL)}
        self.assertEqual(len(rows), 5)
        home = rows["/home (user jzj)"]
        self.assertEqual((home["space"], home["files"]), ("33GB / 50GB", "210K / 500K"))
        self.assertAlmostEqual(home["space_used_fraction"], 0.66)
        self.assertFalse(home["near_quota"])
        full = rows["/project (project def-mijungp)"]
        self.assertTrue(full["near_quota"])  # flagged with -> and at 99% of files
        self.assertAlmostEqual(full["files_used_fraction"], 0.994)
        self.assertFalse(rows["/project (project rrg-mijungp)"]["near_quota"])  # 84%
        self.assertEqual(rows["/nearline (project def-mijungp)"]["files"], "21 / 5000")

    def test_ignores_unrelated_text(self) -> None:
        self.assertEqual(parse_diskusage_report("no quota information here\n"), [])


if __name__ == "__main__":
    unittest.main()
