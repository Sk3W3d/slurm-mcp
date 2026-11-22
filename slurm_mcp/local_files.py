from __future__ import annotations

import subprocess
from pathlib import Path


def resolve_local_path(path: str | None, roots: tuple[Path, ...], *, label: str = "local path") -> Path:
    """Resolve a local path and require it to stay inside one of the allowed roots.

    Relative paths are resolved against the first root.
    """
    if not roots:
        raise ValueError("No local roots are configured.")
    if path is None or path == "":
        return roots[0]
    if not isinstance(path, str) or "\x00" in path:
        raise ValueError(f"Invalid {label}.")
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = roots[0] / candidate
    resolved = candidate.resolve()
    for root in roots:
        if resolved == root or root in resolved.parents:
            return resolved
    known = ", ".join(str(root) for root in roots)
    raise ValueError(f"{label} {resolved} is outside allowed local roots: {known}")


def check_transfer_size(size: int, limit: int, label: str) -> None:
    if size > limit:
        raise ValueError(f"{label} is {size} bytes, above max_transfer_bytes={limit}.")


def run_local_git(repo: Path, args: list[str], *, timeout: int = 120) -> str:
    # stdin=DEVNULL matters: the server's stdin is the MCP stdio pipe. On Windows a child
    # that inherits it blocks behind the server's pending read on that pipe, so git never
    # starts, and subprocess.run's timeout cannot recover (it waits for the pipes again).
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"local git {' '.join(args)} failed: {stderr or completed.returncode}")
    return completed.stdout.decode("utf-8", "replace").strip()


def local_git_ok(repo: Path, args: list[str], *, timeout: int = 60) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return completed.returncode == 0
