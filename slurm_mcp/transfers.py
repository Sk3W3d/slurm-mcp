"""Directory transfers: tar archives packed on one side and unpacked on the other.

Archives are plain (uncompressed) tar: model weights and checkpoints do not
compress, and it keeps login-node CPU use low. Local unpacking uses tarfile's
"data" filter, which refuses absolute paths, '..', device files, and links
that point outside the destination.
"""

from __future__ import annotations

import fnmatch
import os
import tarfile
import uuid
from pathlib import Path, PurePosixPath

MAX_EXCLUDES = 50


def validate_excludes(patterns: list[str] | None) -> list[str]:
    if patterns is None:
        return []
    if not isinstance(patterns, list) or len(patterns) > MAX_EXCLUDES:
        raise ValueError(f"exclude must be a list of at most {MAX_EXCLUDES} patterns.")
    safe: list[str] = []
    for pattern in patterns:
        if not isinstance(pattern, str) or not pattern.strip():
            raise ValueError("exclude patterns must be non-empty strings.")
        if any(char in pattern for char in "\x00\n\r"):
            raise ValueError("exclude patterns cannot contain control characters.")
        safe.append(pattern.strip())
    return safe


def is_excluded(relative: str, patterns: list[str]) -> bool:
    """Match like GNU tar --exclude: a pattern without '/' matches any path component."""
    parts = PurePosixPath(relative).parts
    for pattern in patterns:
        if "/" in pattern.strip("/"):
            if fnmatch.fnmatch(relative, pattern.strip("/")) or fnmatch.fnmatch(relative, pattern.strip("/") + "/*"):
                return True
        elif any(fnmatch.fnmatch(part, pattern.strip("/")) for part in parts):
            return True
    return False


def make_local_tar(source: Path, tar_path: Path, excludes: list[str]) -> tuple[int, int]:
    """Pack the contents of a local directory. Returns (entries, bytes of regular files)."""
    entries = 0
    total = 0
    with tarfile.open(tar_path, "w", format=tarfile.PAX_FORMAT) as archive:
        for root, dirs, files in os.walk(source):
            relative_root = Path(root).relative_to(source).as_posix()
            relative_root = "" if relative_root == "." else relative_root
            kept_dirs = []
            for name in sorted(dirs):
                relative = f"{relative_root}/{name}" if relative_root else name
                if is_excluded(relative, excludes):
                    continue
                kept_dirs.append(name)
                archive.add(Path(root) / name, arcname=relative, recursive=False)
                entries += 1
            dirs[:] = kept_dirs  # prune excluded folders; symlinked folders are stored as links, not followed
            for name in sorted(files):
                relative = f"{relative_root}/{name}" if relative_root else name
                if is_excluded(relative, excludes):
                    continue
                path = Path(root) / name
                archive.add(path, arcname=relative, recursive=False)
                entries += 1
                if path.is_file() and not path.is_symlink():
                    total += path.stat().st_size
    return entries, total


def count_entries(tar_path: Path) -> int:
    with tarfile.open(tar_path) as archive:
        return sum(1 for member in archive.getmembers() if member.name not in {".", "./"})


def safe_extract(tar_path: Path, destination: Path) -> int:
    """Unpack into destination, refusing entries that would land outside it."""
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as archive:
        members = [member for member in archive.getmembers() if member.name not in {".", "./"}]
        try:
            archive.extractall(destination, members=members, filter="data")
        except tarfile.FilterError as exc:
            raise ValueError(f"Refused to unpack an unsafe archive entry: {exc}") from exc
        except OSError as exc:
            raise ValueError(
                f"Could not unpack into {destination}: {exc}. (On Windows, names with characters like "
                "':' or '?' cannot be created; exclude those files.)"
            ) from exc
    return len(members)


def transfer_name(suffix: str = ".tar") -> str:
    return f".slurm-mcp-transfer-{uuid.uuid4().hex[:12]}{suffix}"


# -- remote commands (argv lists; paths are passed as arguments, never spliced into the script) --

_TAR_CREATE = (
    'src=$1; out=$2; shift 2; mkdir -p -- "$(dirname -- "$out")" && '
    'tar -C "$src" -cf "$out" "$@" . && stat -c %s -- "$out"'
)
_TAR_EXTRACT = (
    'archive=$1; dest=$2; mkdir -p -- "$dest" && tar -xf "$archive" -C "$dest" --no-same-owner; '
    'rc=$?; rm -f -- "$archive"; exit $rc'
)
_COPY_DIR = (
    'src=$1; dest=$2; shift 2; mkdir -p -- "$dest" && '
    'tar -C "$src" -cf - "$@" . | tar -xf - -C "$dest" --no-same-owner'
)
_COPY_FILE = 'mkdir -p -- "$(dirname -- "$2")" && cp -p -- "$1" "$2"'


def _exclude_args(excludes: list[str]) -> list[str]:
    return [f"--exclude={pattern}" for pattern in excludes]


def remote_tar_create_argv(source_dir: str, archive: str, excludes: list[str]) -> list[str]:
    """Pack source_dir's contents into archive; prints the archive size in bytes."""
    return ["bash", "-c", _TAR_CREATE, "_", source_dir, archive, *_exclude_args(excludes)]


def remote_tar_extract_argv(archive: str, destination: str) -> list[str]:
    """Unpack archive into destination (created if needed), then delete the archive."""
    return ["bash", "-c", _TAR_EXTRACT, "_", archive, destination]


def remote_copy_argv(source: str, destination: str, *, is_dir: bool, excludes: list[str]) -> list[str]:
    """Copy within one cluster: a directory's contents through a tar pipe, or one file with cp -p."""
    if is_dir:
        return ["bash", "-c", _COPY_DIR, "_", source, destination, *_exclude_args(excludes)]
    return ["bash", "-c", _COPY_FILE, "_", source, destination]


def remote_size_argv(path: str, excludes: list[str]) -> list[str]:
    """Apparent size in bytes of a file or directory tree."""
    return ["du", "-sb", *_exclude_args(excludes), "--", path]


def remote_kind_argv(path: str) -> list[str]:
    return ["stat", "-c", "%F", "--", path]


def parse_size(output: str) -> int | None:
    first = output.strip().split()
    return int(first[0]) if first and first[0].isdigit() else None
