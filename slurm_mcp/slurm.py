from __future__ import annotations

import re
from typing import Any

from .validators import validate_argument_list


SQUEUE_FIELDS = (
    "job_id",
    "partition",
    "name",
    "user",
    "state",
    "elapsed",
    "time_limit",
    "nodes",
    "cpus",
    "memory",
    "reason",
)
SQUEUE_FORMAT = "%i|%P|%j|%u|%T|%M|%l|%D|%C|%m|%R"

# Job states after which a job no longer changes. MIXED is this module's summary
# for an array job whose tasks all finished with different states.
TERMINAL_JOB_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "MIXED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "TIMEOUT",
    }
)

SACCT_FIELDS = (
    "job_id_raw",
    "job_id",
    "job_name",
    "partition",
    "account",
    "allocated_cpus",
    "state",
    "exit_code",
    "elapsed",
    "time_limit",
    "submit_time",
    "start_time",
    "end_time",
    "node_list",
    "allocated_tres",
    "requested_tres",
    "max_rss",
)
SACCT_FORMAT = ",".join(
    (
        "JobIDRaw",
        "JobID",
        "JobName",
        "Partition",
        "Account",
        "AllocCPUS",
        "State",
        "ExitCode",
        "Elapsed",
        "Timelimit",
        "Submit",
        "Start",
        "End",
        "NodeList",
        "AllocTRES",
        "ReqTRES",
        "MaxRSS",
    )
)

_ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
_SUBMITTED_JOB_RE = re.compile(r"Submitted batch job (?P<job_id>[0-9]+(?:_[0-9]+)?)")

_PROFILE_SBATCH_OPTIONS = {
    "account": "--account",
    "array": "--array",
    "constraint": "--constraint",
    "cpus_per_task": "--cpus-per-task",
    "dependency": "--dependency",
    "error": "--error",
    "exclusive": "--exclusive",
    "mail_type": "--mail-type",
    "mail_user": "--mail-user",
    "mem": "--mem",
    "mem_per_cpu": "--mem-per-cpu",
    "nodes": "--nodes",
    "ntasks": "--ntasks",
    "ntasks_per_node": "--ntasks-per-node",
    "output": "--output",
    "partition": "--partition",
    "qos": "--qos",
    "signal": "--signal",
    "time": "--time",
}
# "env" is applied by the caller through build_sbatch_argv(env=...), not as an sbatch flag.
_PROFILE_SPECIAL_KEYS = {"description", "env", "gpus", "gpu_type", "gres", "sbatch_options"}
_VALID_GPU_REQUEST_STYLES = {"gres", "gpus", "gpus_per_node"}


def squeue_argv(user: str, states: str | None = None) -> list[str]:
    argv = [
        "squeue",
        "-u",
        user,
        "--noheader",
        f"--format={SQUEUE_FORMAT}",
    ]
    if states:
        argv.append(f"--states={states}")
    return argv


def squeue_job_argv(job_id: str) -> list[str]:
    return [
        "squeue",
        "-j",
        job_id,
        "--noheader",
        f"--format={SQUEUE_FORMAT}",
    ]


def sacct_argv(job_id: str) -> list[str]:
    return [
        "sacct",
        "-j",
        job_id,
        f"--format={SACCT_FORMAT}",
        "--parsable2",
        "--noheader",
    ]


def sacct_jobs_argv(job_ids: list[str]) -> list[str]:
    """Allocation-level accounting rows (-X) for several jobs in one call."""
    if not job_ids:
        raise ValueError("job_ids must not be empty.")
    return [
        "sacct",
        "-X",
        "-j",
        ",".join(job_ids),
        f"--format={SACCT_FORMAT}",
        "--parsable2",
        "--noheader",
    ]


def summarize_job_rows(rows: list[dict[str, str]], job_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Fold sacct rows into one state per submitted job id.

    Array jobs report one row per task (123_0, 123_1, 123_[2-9]); they are
    combined into counts by state. The overall state is the only state if
    there is one; otherwise RUNNING or PENDING while any task is, another
    unfinished state if any, or MIXED once every task has finished.
    """
    wanted = set(job_ids)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        row_id = row.get("job_id", "")
        if "." in row_id:
            # Job steps (123.batch, 123.extern) repeat the allocation's state.
            continue
        base = row_id.split("_", 1)[0]
        if base in wanted:
            grouped.setdefault(base, []).append(row)

    summary: dict[str, dict[str, Any]] = {}
    for job_id, job_rows in grouped.items():
        states = [(row.get("state") or "").split()[0].rstrip("+").upper() for row in job_rows]
        counts: dict[str, int] = {}
        for state in states:
            counts[state] = counts.get(state, 0) + 1
        unfinished = [state for state in counts if state not in TERMINAL_JOB_STATES]
        if len(counts) == 1:
            overall = states[0]
        elif "RUNNING" in counts:
            overall = "RUNNING"
        elif "PENDING" in counts:
            overall = "PENDING"
        elif unfinished:
            overall = unfinished[0]
        else:
            overall = "MIXED"
        first = job_rows[0]
        summary[job_id] = {
            "state": overall,
            "exit_code": first.get("exit_code") if len(job_rows) == 1 else None,
            "elapsed": first.get("elapsed") if len(job_rows) == 1 else None,
            "array_tasks": counts if len(job_rows) > 1 else None,
        }
    return summary


def sacct_user_argv(user: str, start: str) -> list[str]:
    return [
        "sacct",
        "-u",
        user,
        "-X",
        f"--starttime={start}",
        f"--format={SACCT_FORMAT}",
        "--parsable2",
        "--noheader",
    ]


def scancel_argv(job_id: str, user: str) -> list[str]:
    # --user makes scancel act only when the job belongs to this user.
    return ["scancel", f"--user={user}", job_id]


SSHARE_FIELDS = ("account", "user", "raw_shares", "norm_shares", "raw_usage", "effective_usage", "fair_share")


def sshare_argv(user: str) -> list[str]:
    return [
        "sshare",
        "-U",
        "-u",
        user,
        "--noheader",
        "--parsable2",
        "--format=Account,User,RawShares,NormShares,RawUsage,EffectvUsage,FairShare",
    ]


def parse_sshare(output: str) -> list[dict[str, str]]:
    return _parse_delimited_rows(output, SSHARE_FIELDS)


def submit_accounts(sshare_rows: list[dict[str, str]]) -> list[str]:
    """Return account names usable with --account (Alliance lists def-pi_cpu / def-pi_gpu)."""
    accounts: list[str] = []
    for row in sshare_rows:
        name = row.get("account", "")
        base = re.sub(r"_(cpu|gpu)$", "", name)
        if base and base not in accounts:
            accounts.append(base)
    return accounts


SINFO_FIELDS = ("partition", "availability", "time_limit", "nodes", "state", "cpus", "memory_mb", "gres")
SINFO_FORMAT = "%P|%a|%l|%D|%t|%c|%m|%G"


def sinfo_argv() -> list[str]:
    return ["sinfo", "--noheader", f"--format={SINFO_FORMAT}"]


def parse_sinfo(output: str) -> list[dict[str, str]]:
    return _parse_delimited_rows(output, SINFO_FIELDS)


def requests_gpus(options: dict[str, Any]) -> bool:
    """Whether merged job options ask for GPUs (through gpus or a gpu gres)."""
    gpus = options.get("gpus")
    if gpus not in (None, "", 0, "0"):
        return True
    gres = options.get("gres")
    return isinstance(gres, str) and "gpu" in gres


def merge_job_options(
    profile: dict[str, Any] | None,
    overrides: dict[str, Any] | None,
    default_account: str | None = None,
    cpu_account: str | None = None,
) -> dict[str, Any]:
    """Apply overrides to a profile and fill in --account when neither sets one.

    cpu_account, when given, is used for jobs that request no GPUs (some
    allocations, such as a GPU-only RAC, cannot run CPU jobs); otherwise
    default_account is used.
    """
    merged: dict[str, Any] = dict(profile or {})
    for key, value in (overrides or {}).items():
        if not isinstance(key, str) or not key:
            raise ValueError("Override keys must be non-empty strings.")
        merged[key] = value
    if not merged.get("account"):
        account = cpu_account if cpu_account and not requests_gpus(merged) else default_account
        if account:
            merged["account"] = account
    return merged


def parse_squeue(output: str) -> list[dict[str, str]]:
    return _parse_delimited_rows(output, SQUEUE_FIELDS)


def parse_sacct(output: str) -> list[dict[str, str]]:
    return _parse_delimited_rows(output, SACCT_FIELDS)


def parse_sbatch_job_id(output: str) -> str:
    match = _SUBMITTED_JOB_RE.search(output)
    if match is None:
        raise ValueError(f"Could not parse Slurm job id from sbatch output: {output!r}")
    return match.group("job_id")


def build_sbatch_argv(
    profile: dict[str, Any],
    script: str,
    *,
    job_name: str | None = None,
    script_args: list[str] | None = None,
    cluster_slurm: dict[str, Any] | None = None,
    log_dir: str | None = None,
    env: dict[str, str] | None = None,
) -> list[str]:
    if not isinstance(profile, dict):
        raise ValueError("Job profile must be a mapping.")

    cluster_slurm = cluster_slurm or {}
    unsupported = set(profile) - set(_PROFILE_SBATCH_OPTIONS) - _PROFILE_SPECIAL_KEYS
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"Unsupported profile keys: {names}")

    argv = ["sbatch"]
    if job_name:
        argv.append(f"--job-name={job_name}")

    for key, option in _PROFILE_SBATCH_OPTIONS.items():
        if key not in profile or profile[key] is None:
            continue
        argv.append(_format_sbatch_option(option, profile[key], key))

    if "gres" in profile and profile["gres"] is not None:
        argv.append(_format_sbatch_option("--gres", profile["gres"], "gres"))
    elif "gpus" in profile and profile["gpus"] is not None:
        argv.append(_gpu_option(profile, cluster_slurm))

    if log_dir and "output" not in profile:
        argv.append(f"--output={log_dir}/%x-%j.out")
    if log_dir and "error" not in profile:
        argv.append(f"--error={log_dir}/%x-%j.err")

    if env:
        argv.append(export_env_option(env))

    argv.extend(_profile_sbatch_options(profile))
    argv.append(script)
    argv.extend(validate_argument_list(script_args, "script_args"))
    return argv


def export_env_option(env: dict[str, str]) -> str:
    """Build --export=ALL,NAME=value,... so the job keeps the login environment plus extras."""
    parts = ["ALL"]
    for name, value in env.items():
        if not isinstance(name, str) or not _ENV_NAME_RE.fullmatch(name):
            raise ValueError(f"Invalid environment variable name: {name!r}")
        text = str(value)
        if "," in text or "\x00" in text or "\n" in text or "\r" in text:
            raise ValueError(f"Environment variable {name} cannot contain commas or control characters.")
        parts.append(f"{name}={text}")
    return "--export=" + ",".join(parts)


def merge_env(*sources: dict[str, Any] | None) -> dict[str, str]:
    merged: dict[str, str] = {}
    for source in sources:
        for name, value in (source or {}).items():
            if value is None:
                merged.pop(name, None)
            else:
                merged[name] = str(value)
    return merged


def _parse_delimited_rows(output: str, fields: tuple[str, ...]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in output.splitlines():
        raw_line = line.rstrip("\n")
        if not raw_line.strip():
            continue
        values = raw_line.split("|", maxsplit=len(fields) - 1)
        if len(values) < len(fields):
            values.extend([""] * (len(fields) - len(values)))
        row = dict(zip(fields, (value.strip() for value in values), strict=True))
        row["raw_line"] = raw_line
        rows.append(row)
    return rows


def _format_sbatch_option(option: str, value: Any, key: str) -> str:
    if isinstance(value, bool):
        if value:
            return option
        raise ValueError(f"Profile key '{key}' cannot be false.")
    if isinstance(value, (str, int, float)):
        text = str(value)
        _validate_sbatch_value(text, key)
        return f"{option}={text}"
    raise ValueError(f"Profile key '{key}' must be a string, number, or true.")


def _gpu_option(profile: dict[str, Any], cluster_slurm: dict[str, Any]) -> str:
    gpus = str(profile["gpus"])
    _validate_sbatch_value(gpus, "gpus")
    gpu_type = profile.get("gpu_type")
    if gpu_type is not None:
        if not isinstance(gpu_type, str):
            raise ValueError("Profile key 'gpu_type' must be a string.")
        _validate_sbatch_value(gpu_type, "gpu_type")

    style = str(cluster_slurm.get("gpu_request_style", "gres"))
    if style not in _VALID_GPU_REQUEST_STYLES:
        valid = ", ".join(sorted(_VALID_GPU_REQUEST_STYLES))
        raise ValueError(f"Unsupported gpu_request_style '{style}'. Valid styles: {valid}")

    if style == "gres":
        if gpu_type:
            return f"--gres=gpu:{gpu_type}:{gpus}"
        return f"--gres=gpu:{gpus}"
    if style == "gpus":
        return f"--gpus={gpus}"
    return f"--gpus-per-node={gpus}"


def _profile_sbatch_options(profile: dict[str, Any]) -> list[str]:
    raw_options = profile.get("sbatch_options")
    if raw_options is None:
        return []
    options = validate_argument_list(raw_options, "sbatch_options")
    for option in options:
        if not option.startswith("-"):
            raise ValueError("Each sbatch_options entry must be an sbatch option.")
    return options


def _validate_sbatch_value(value: str, key: str) -> None:
    if not value:
        raise ValueError(f"Profile key '{key}' cannot be empty.")
    if "\x00" in value or "\n" in value or "\r" in value:
        raise ValueError(f"Profile key '{key}' cannot contain control characters.")
