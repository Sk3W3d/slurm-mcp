from __future__ import annotations

import re
from typing import Any


_AHEAD_BEHIND_RE = re.compile(r"ahead (?P<ahead>\d+)|behind (?P<behind>\d+)")
_REF_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/@+-]{0,199}")
_URL_RE = re.compile(r"(?:https://|ssh://|git@)[A-Za-z0-9._~:/@+%-]+")
SYNC_BUNDLE_NAME = "slurm-mcp-sync.bundle"


def validate_git_ref(ref: str, label: str = "ref") -> str:
    if not isinstance(ref, str) or not _REF_RE.fullmatch(ref):
        raise ValueError(f"Invalid Git {label}. Use letters, numbers, '.', '_', '/', '@', '+', or '-'.")
    if ".." in ref or ref.endswith((".lock", "/", ".")) or "//" in ref or "@{" in ref:
        raise ValueError(f"Invalid Git {label}.")
    return ref


def validate_git_url(url: str) -> str:
    if not isinstance(url, str) or not _URL_RE.fullmatch(url):
        raise ValueError("Git URL must start with https://, ssh://, or git@ and contain no spaces or shell characters.")
    return url


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


def git_fetch_argv(repo: str) -> list[str]:
    return ["git", "-C", repo, "fetch", "--prune"]


def git_checkout_argv(repo: str, ref: str) -> list[str]:
    return ["git", "-C", repo, "checkout", validate_git_ref(ref)]


def git_clone_argv(url: str, destination: str, branch: str | None = None) -> list[str]:
    argv = ["git", "clone"]
    if branch:
        argv.extend(["--branch", validate_git_ref(branch, "branch")])
    argv.extend(["--", validate_git_url(url), destination])
    return argv


def git_bundle_pull_argv(repo: str, bundle: str, branch: str) -> list[str]:
    """Fast-forward the checked-out branch from a bundle."""
    return ["git", "-C", repo, "pull", "--ff-only", bundle, validate_git_ref(branch, "branch")]


def git_bundle_fetch_argv(repo: str, bundle: str, branch: str) -> list[str]:
    """Update a branch that is not checked out; Git refuses non-fast-forward updates without '+'."""
    safe_branch = validate_git_ref(branch, "branch")
    return ["git", "-C", repo, "fetch", bundle, f"{safe_branch}:{safe_branch}"]


def git_has_commit_argv(repo: str, commit: str) -> list[str]:
    return ["git", "-C", repo, "cat-file", "-e", f"{validate_git_ref(commit, 'commit')}^{{commit}}"]


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
