"""Parse Alliance `diskusage_report` output into structured quota rows.

Example input:

                            Description                Space         # of files
                       /home (user jzj)          33GB/  50GB         210K/ 500K
         /project (project def-mijungp)         479GB/1000GB     ->  497K/ 500K
"""

from __future__ import annotations

import re
from typing import Any

_PAIR = re.compile(r"([\d.]+)\s*([KMGTP]?i?B?)\s*/\s*([\d.]+)\s*([KMGTP]?i?B?)", re.IGNORECASE)
_BYTES = {"": 1, "B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4, "PB": 1024**5}
_COUNTS = {"": 1, "K": 1000, "M": 1000**2, "G": 1000**3}
NEAR_QUOTA = 0.9


def _amount(number: str, unit: str, table: dict[str, int]) -> float | None:
    unit = unit.upper().replace("IB", "B")
    factor = table.get(unit)
    return None if factor is None else float(number) * factor


def parse_diskusage_report(output: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        pairs = list(_PAIR.finditer(line))
        if len(pairs) < 2:
            continue
        space, files = pairs[0], pairs[1]
        description = line[: space.start()].strip()
        if not description.startswith("/"):
            continue
        warned = "->" in line[space.end() : files.start()]
        used = _amount(space.group(1), space.group(2), _BYTES)
        quota = _amount(space.group(3), space.group(4), _BYTES)
        files_used = _amount(files.group(1), files.group(2), _COUNTS)
        files_quota = _amount(files.group(3), files.group(4), _COUNTS)
        space_fraction = used / quota if used is not None and quota else None
        files_fraction = files_used / files_quota if files_used is not None and files_quota else None
        fullest = max((value for value in (space_fraction, files_fraction) if value is not None), default=None)
        rows.append(
            {
                "location": description,
                "space": f"{space.group(1)}{space.group(2)} / {space.group(3)}{space.group(4)}",
                "files": f"{files.group(1)}{files.group(2)} / {files.group(3)}{files.group(4)}",
                "space_used_fraction": round(space_fraction, 3) if space_fraction is not None else None,
                "files_used_fraction": round(files_fraction, 3) if files_fraction is not None else None,
                "near_quota": warned or (fullest is not None and fullest >= NEAR_QUOTA),
            }
        )
    return rows
