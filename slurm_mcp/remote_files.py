from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
import re
from typing import Any

from .validators import validate_job_id, validate_remote_relative_path


_REMOTE_ROOT_KEYS = ("remote_repo", "run_root")
_GREP_PATTERN_MAX_CHARS = 500
_FIND_FIELDS = ("modified_epoch", "size_bytes", "path")


@dataclass(frozen=True)
class ResolvedRemotePath:
    path: str
    root_name: str
    root: str
    relative: str


def allowed_roots_from_storage(storage: dict[str, Any]) -> dict[str, str]:
    roots: dict[str, str] = {}
    for key in _REMOTE_ROOT_KEYS:
        value = storage.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value:
            raise ValueError(f"storage.{key} must be a non-empty string when set.")
        roots[key] = _normalize_absolute_remote_path(value, f"storage.{key}")
    if not roots:
        raise ValueError("Cluster must define storage.remote_repo or storage.run_root.")
    return roots


def resolve_remote_path(
    path: str | None,
    roots: dict[str, str],
    *,
    base: str = "run_root",
    label: str = "path",
) -> ResolvedRemotePath:
    if path is None or path == "":
        return _root_path(roots, base)
    if _is_absolute_remote_path(path):
        absolute = _normalize_absolute_remote_path(path, label)
        return _resolve_absolute_path(absolute, roots, label)

    if base not in roots:
        known = ", ".join(sorted(roots))
        raise ValueError(f"Relative {label} needs base '{base}'. Known roots: {known}")
    relative = validate_remote_relative_path(path, label)
    absolute = _join_remote_path(roots[base], relative)
    return ResolvedRemotePath(
        path=absolute,
        root_name=base,
        root=roots[base],
        relative=relative,
    )


def validate_file_base(base: str) -> str:
    if base not in _REMOTE_ROOT_KEYS:
        known = ", ".join(_REMOTE_ROOT_KEYS)
        raise ValueError(f"Invalid file base '{base}'. Valid bases: {known}")
    return base


def validate_line_count(lines: int, *, max_lines: int = 2000) -> int:
    if not isinstance(lines, int) or lines < 1 or lines > max_lines:
        raise ValueError(f"lines must be an integer between 1 and {max_lines}.")
    return lines


def validate_limit(limit: int, *, max_limit: int = 1000) -> int:
    if not isinstance(limit, int) or limit < 1 or limit > max_limit:
        raise ValueError(f"limit must be an integer between 1 and {max_limit}.")
    return limit


def validate_max_bytes(max_bytes: int, *, hard_limit: int = 1_000_000) -> int:
    if not isinstance(max_bytes, int) or max_bytes < 1 or max_bytes > hard_limit:
        raise ValueError(f"max_bytes must be an integer between 1 and {hard_limit}.")
    return max_bytes


def validate_max_depth(max_depth: int, *, hard_limit: int = 10) -> int:
    if not isinstance(max_depth, int) or max_depth < 0 or max_depth > hard_limit:
        raise ValueError(f"max_depth must be an integer between 0 and {hard_limit}.")
    return max_depth


def validate_grep_pattern(pattern: str) -> str:
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("grep pattern must be a non-empty string.")
    if len(pattern) > _GREP_PATTERN_MAX_CHARS:
        raise ValueError(f"grep pattern must be at most {_GREP_PATTERN_MAX_CHARS} characters.")
    if "\x00" in pattern or "\n" in pattern or "\r" in pattern:
        raise ValueError("grep pattern cannot contain control characters.")
    return pattern


def list_files_argv(path: str, max_depth: int) -> list[str]:
    return [
        "find",
        path,
        "-maxdepth",
        str(validate_max_depth(max_depth)),
        "-type",
        "f",
        "-printf",
        "%T@|%s|%p\n",
    ]


def tail_file_argv(path: str, lines: int) -> list[str]:
    return ["tail", "-n", str(validate_line_count(lines)), "--", path]


def read_file_argv(path: str, max_bytes: int) -> list[str]:
    return ["head", "-c", str(validate_max_bytes(max_bytes)), "--", path]


def stat_file_argv(path: str) -> list[str]:
    return ["stat", "-c", "%s|%F|%y", "--", path]


def grep_files_argv(pattern: str, path: str) -> list[str]:
    return ["grep", "-R", "-n", "--fixed-strings", "--", validate_grep_pattern(pattern), path]


def job_log_patterns(job_id: str) -> tuple[str, str]:
    safe_job_id = validate_job_id(job_id)
    return (f"*-{safe_job_id}.out", f"*-{safe_job_id}.err")


def find_job_logs_argv(log_root: str, job_id: str) -> list[str]:
    stdout_pattern, stderr_pattern = job_log_patterns(job_id)
    return [
        "find",
        log_root,
        "-maxdepth",
        "3",
        "-type",
        "f",
        "(",
        "-name",
        stdout_pattern,
        "-o",
        "-name",
        stderr_pattern,
        ")",
        "-printf",
        "%T@|%s|%p\n",
    ]


def parse_file_listing(output: str, roots: dict[str, str], limit: int) -> list[dict[str, Any]]:
    safe_limit = validate_limit(limit)
    files = [_parse_find_line(line, roots) for line in output.splitlines() if line.strip()]
    files.sort(key=lambda item: float(item["modified_epoch"]), reverse=True)
    return files[:safe_limit]


def parse_grep_output(output: str, roots: dict[str, str], limit: int) -> list[dict[str, Any]]:
    safe_limit = validate_limit(limit)
    matches: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split(":", maxsplit=2)
        if len(parts) != 3:
            continue
        path, line_number, text = parts
        try:
            resolved = _resolve_absolute_path(_normalize_absolute_remote_path(path, "grep result"), roots, "grep result")
        except ValueError:
            continue
        matches.append(
            {
                "path": resolved.path,
                "root": resolved.root_name,
                "relative_path": resolved.relative,
                "line": int(line_number) if line_number.isdigit() else None,
                "text": text,
            }
        )
        if len(matches) >= safe_limit:
            break
    return matches


def parse_stat_output(output: str) -> dict[str, str | int | None]:
    text = output.strip()
    if not text:
        return {"size_bytes": None, "file_type": None, "modified_time": None}
    size, file_type, modified_time = (text.split("|", maxsplit=2) + ["", ""])[:3]
    return {
        "size_bytes": int(size) if size.isdigit() else None,
        "file_type": file_type,
        "modified_time": modified_time,
    }


def classify_job_log(path: str, job_id: str) -> str:
    safe_job_id = validate_job_id(job_id)
    if re.search(rf"-{re.escape(safe_job_id)}\.out$", path):
        return "stdout"
    if re.search(rf"-{re.escape(safe_job_id)}\.err$", path):
        return "stderr"
    return "unknown"


def select_job_log(files: list[dict[str, Any]], stream: str) -> dict[str, Any]:
    if stream not in {"stdout", "stderr"}:
        raise ValueError("stream must be 'stdout' or 'stderr'.")
    for file_info in files:
        if file_info.get("stream") == stream:
            return file_info
    raise ValueError(f"No {stream} log file found for this job.")


def _parse_find_line(line: str, roots: dict[str, str]) -> dict[str, Any]:
    modified_epoch, size_bytes, path = (line.split("|", maxsplit=2) + ["", ""])[:3]
    resolved = _resolve_absolute_path(
        _normalize_absolute_remote_path(path, "find result"),
        roots,
        "find result",
    )
    return {
        "path": resolved.path,
        "root": resolved.root_name,
        "relative_path": resolved.relative,
        "size_bytes": int(size_bytes) if size_bytes.isdigit() else None,
        "modified_epoch": modified_epoch,
    }


def _root_path(roots: dict[str, str], base: str) -> ResolvedRemotePath:
    if base not in roots:
        known = ", ".join(sorted(roots))
        raise ValueError(f"Unknown file base '{base}'. Known roots: {known}")
    return ResolvedRemotePath(path=roots[base], root_name=base, root=roots[base], relative="")


def _resolve_absolute_path(path: str, roots: dict[str, str], label: str) -> ResolvedRemotePath:
    for root_name, root in sorted(roots.items(), key=lambda item: len(item[1]), reverse=True):
        if path == root:
            return ResolvedRemotePath(path=path, root_name=root_name, root=root, relative="")
        prefix = root.rstrip("/") + "/"
        if path.startswith(prefix):
            return ResolvedRemotePath(
                path=path,
                root_name=root_name,
                root=root,
                relative=path[len(prefix) :],
            )
    known = ", ".join(f"{name}={root}" for name, root in sorted(roots.items()))
    raise ValueError(f"Remote {label} is outside allowed roots: {known}")


def _normalize_absolute_remote_path(path: str, label: str) -> str:
    if not isinstance(path, str) or not path:
        raise ValueError(f"Remote {label} must be a non-empty absolute path.")
    if "\x00" in path or "\n" in path or "\r" in path:
        raise ValueError(f"Remote {label} cannot contain control characters.")
    if "\\" in path:
        raise ValueError(f"Remote {label} must use POSIX '/' separators.")
    remote_path = PurePosixPath(path)
    if not remote_path.is_absolute():
        raise ValueError(f"Remote {label} must be an absolute path.")
    if any(part in {"", ".", ".."} for part in remote_path.parts[1:]):
        raise ValueError(f"Remote {label} cannot contain '.', '..', or empty path segments.")
    normalized = str(remote_path)
    return "/" if normalized == "/" else normalized.rstrip("/")


def _is_absolute_remote_path(path: str) -> bool:
    return path.startswith("/")


def _join_remote_path(root: str, relative: str) -> str:
    return root.rstrip("/") + "/" + relative
