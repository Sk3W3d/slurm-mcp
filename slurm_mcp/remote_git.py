from __future__ import annotations

import re
from typing import Any


_AHEAD_BEHIND_RE = re.compile(r"ahead (?P<ahead>\d+)|behind (?P<behind>\d+)")


def git_status_argv(repo: str) -> list[str]:
    return ["git", "-C", repo, "status", "--short", "--branch"]


def git_rev_argv(repo: str, *, short: bool = False) -> list[str]:
    if short:
        return ["git", "-C", repo, "rev-parse", "--short=12", "HEAD"]
    return ["git", "-C", repo, "rev-parse", "HEAD"]


def git_branch_argv(repo: str) -> list[str]:
    return ["git", "-C", repo, "branch", "--show-current"]


def git_fetch_dry_run_argv(repo: str) -> list[str]:
    return ["git", "-C", repo, "fetch", "--dry-run", "--verbose"]


def git_pull_argv(repo: str) -> list[str]:
    return ["git", "-C", repo, "pull", "--ff-only"]


def parse_git_status(output: str) -> dict[str, Any]:
    branch: str | None = None
    upstream: str | None = None
    ahead = 0
    behind = 0
    entries: list[dict[str, str]] = []

    for line in output.splitlines():
        if not line:
            continue
        if line.startswith("## "):
            branch_info = line[3:].strip()
            branch, upstream, ahead, behind = _parse_branch_line(branch_info)
            continue

        if len(line) >= 4:
            entries.append(
                {
                    "index": line[0],
                    "worktree": line[1],
                    "path": line[3:],
                    "raw": line,
                }
            )
        else:
            entries.append({"index": "", "worktree": "", "path": line, "raw": line})

    return {
        "branch": branch,
        "upstream": upstream,
        "ahead": ahead,
        "behind": behind,
        "clean": not entries,
        "entries": entries,
        "raw": output,
    }


def _parse_branch_line(branch_info: str) -> tuple[str | None, str | None, int, int]:
    if "..." not in branch_info:
        return branch_info or None, None, 0, 0

    branch, rest = branch_info.split("...", maxsplit=1)
    upstream = rest
    detail = ""
    if " [" in rest and rest.endswith("]"):
        upstream, detail = rest.rsplit(" [", maxsplit=1)
        detail = detail[:-1]

    ahead = 0
    behind = 0
    for match in _AHEAD_BEHIND_RE.finditer(detail):
        if match.group("ahead") is not None:
            ahead = int(match.group("ahead"))
        if match.group("behind") is not None:
            behind = int(match.group("behind"))

    return branch or None, upstream or None, ahead, behind
