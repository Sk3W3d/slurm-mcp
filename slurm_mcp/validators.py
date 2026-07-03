from __future__ import annotations

from pathlib import PurePosixPath
import re


_JOB_ID_RE = re.compile(r"[0-9]+(?:_[0-9]+)?")
_JOB_NAME_RE = re.compile(r"[A-Za-z0-9_.:@+-]{1,128}")
_STATE_RE = re.compile(r"[A-Za-z_,]+")


def validate_job_id(job_id: str) -> str:
    if not _JOB_ID_RE.fullmatch(job_id):
        raise ValueError("Invalid Slurm job id.")
    return job_id


def validate_job_name(job_name: str) -> str:
    if not _JOB_NAME_RE.fullmatch(job_name):
        raise ValueError(
            "Invalid Slurm job name. Use 1-128 letters, numbers, '.', '_', ':', '@', '+', or '-'."
        )
    return job_name


def validate_state_filter(states: str | None) -> str | None:
    if states is None:
        return None
    normalized = states.strip().upper()
    if not normalized:
        return None
    if not _STATE_RE.fullmatch(normalized):
        raise ValueError("Invalid Slurm state filter.")
    return normalized


def validate_remote_relative_path(path: str, label: str = "path") -> str:
    if not isinstance(path, str) or not path:
        raise ValueError(f"Remote {label} must be a non-empty relative path.")
    if "\x00" in path or "\n" in path or "\r" in path:
        raise ValueError(f"Remote {label} cannot contain control characters.")
    if "\\" in path:
        raise ValueError(f"Remote {label} must use POSIX '/' separators.")
    if path.startswith("~"):
        raise ValueError(f"Remote {label} cannot start with '~'.")

    remote_path = PurePosixPath(path)
    if remote_path.is_absolute():
        raise ValueError(f"Remote {label} must be relative to the configured repository.")
    if any(part in {"", ".", ".."} for part in remote_path.parts):
        raise ValueError(f"Remote {label} cannot contain '.', '..', or empty path segments.")
    return str(remote_path)


def validate_argument_list(values: list[str] | None, label: str) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError(f"{label} must be a list of strings.")
    safe_values: list[str] = []
    for value in values:
        if not isinstance(value, str) or value == "":
            raise ValueError(f"{label} must contain only non-empty strings.")
        if "\x00" in value or "\n" in value or "\r" in value:
            raise ValueError(f"{label} cannot contain control characters.")
        safe_values.append(value)
    return safe_values
