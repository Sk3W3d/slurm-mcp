from __future__ import annotations

import argparse
import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from .audit import preview_text, write_audit_event
from .config import Cluster, Settings, load_settings
from .remote_files import (
    allowed_roots_from_storage,
    classify_job_log,
    find_job_logs_argv,
    grep_files_argv,
    list_files_argv,
    parse_file_listing,
    parse_grep_output,
    parse_stat_output,
    read_file_argv,
    resolve_remote_path,
    select_job_log,
    stat_file_argv,
    tail_file_argv,
    validate_file_base,
    validate_limit,
    validate_line_count,
    validate_max_bytes,
    validate_max_depth,
)
from .slurm import (
    build_sbatch_argv,
    parse_sacct,
    parse_sbatch_job_id,
    parse_squeue,
    sacct_argv,
    squeue_argv,
    squeue_job_argv,
)
from .ssh import CommandResult, check_master, remote_command_result, remote_command_result_in_dir
from .validators import (
    validate_argument_list,
    validate_job_id,
    validate_job_name,
    validate_remote_relative_path,
    validate_state_filter,
)


mcp = FastMCP("alliance-slurm")
settings: Settings | None = None


def get_settings() -> Settings:
    if settings is None:
        raise RuntimeError("Server settings are not initialized.")
    return settings


@mcp.tool()
def list_clusters() -> list[str]:
    """List configured cluster names."""
    return sorted(get_settings().clusters)


@mcp.tool()
def cluster_info(cluster: str | None = None) -> dict[str, object]:
    """Return structured cluster configuration, policies, and setup hints."""
    return get_settings().cluster_info(cluster)


@mcp.tool()
def cluster_docs(cluster: str | None = None) -> list[dict[str, object]]:
    """Return configured documentation sources for a cluster.

    Local markdown docs are read from paths relative to the config file.
    URL docs are returned as source references.
    """
    current_settings = get_settings()
    docs: list[dict[str, object]] = []
    for source in current_settings.doc_sources(cluster):
        try:
            docs.append(current_settings.read_doc_source(source))
        except OSError as exc:
            docs.append(
                {
                    "source": source,
                    "kind": "error",
                    "error": f"Could not read documentation source: {exc}",
                }
            )
        except ValueError as exc:
            docs.append(
                {
                    "source": source,
                    "kind": "error",
                    "error": str(exc),
                }
            )
    return docs


@mcp.tool()
def cluster_auth_status(cluster: str | None = None) -> dict[str, object]:
    """Check whether an SSH master connection is alive for a cluster."""
    current_settings = get_settings()
    target = current_settings.get_cluster(cluster)
    result = check_master(current_settings.ssh_command, target.host)
    message = (result.stderr or result.stdout).strip()
    return {
        "cluster": target.name,
        "host": target.host,
        "connected": result.returncode == 0,
        "message": message,
    }


@mcp.tool()
def remote_whoami(cluster: str | None = None) -> dict[str, str]:
    """Run a non-interactive whoami through the existing SSH connection."""
    current_settings = get_settings()
    target = current_settings.get_cluster(cluster)
    return {
        "cluster": target.name,
        "host": target.host,
        "user": _run_remote_output(
            current_settings,
            target,
            ["whoami"],
            tool="remote_whoami",
            timeout=15,
        ),
    }


@mcp.tool()
def list_job_profiles(cluster: str | None = None) -> dict[str, object]:
    """List configured Slurm job profiles and relevant cluster submission metadata."""
    current_settings = get_settings()
    info = current_settings.cluster_info(cluster)
    return {
        "cluster": info["name"],
        "host": info["host"],
        "profiles": info["profiles"],
        "slurm": info["slurm"],
        "policies": info["policies"],
    }


@mcp.tool()
def list_jobs(cluster: str | None = None, states: str | None = None) -> dict[str, object]:
    """List current Slurm jobs for the remote user as structured records."""
    current_settings = get_settings()
    target = current_settings.get_cluster(cluster)
    user = _remote_user(current_settings, target, tool="list_jobs")
    safe_states = validate_state_filter(states)
    output = _run_remote_output(
        current_settings,
        target,
        squeue_argv(user, safe_states),
        tool="list_jobs",
        timeout=30,
    )
    jobs = parse_squeue(output)
    return {
        "cluster": target.name,
        "host": target.host,
        "user": user,
        "states": safe_states,
        "count": len(jobs),
        "jobs": jobs,
    }


@mcp.tool()
def job_status(job_id: str, cluster: str | None = None) -> dict[str, object]:
    """Read live queue and accounting information for one Slurm job."""
    current_settings = get_settings()
    target = current_settings.get_cluster(cluster)
    safe_job_id = validate_job_id(job_id)

    queue_result = _run_remote_result(
        current_settings,
        target,
        squeue_job_argv(safe_job_id),
        tool="job_status",
        timeout=30,
    )
    accounting_result = _run_remote_result(
        current_settings,
        target,
        sacct_argv(safe_job_id),
        tool="job_status",
        timeout=30,
    )

    queue = parse_squeue(queue_result.stdout) if queue_result.returncode == 0 else []
    accounting = parse_sacct(accounting_result.stdout) if accounting_result.returncode == 0 else []
    return {
        "cluster": target.name,
        "host": target.host,
        "job_id": safe_job_id,
        "summary": _job_summary(queue, accounting),
        "queue": queue,
        "accounting": accounting,
        "queue_error": _command_error(queue_result.returncode, queue_result.stderr),
        "accounting_error": _command_error(accounting_result.returncode, accounting_result.stderr),
    }


@mcp.tool()
def submit_job(
    profile: str,
    script: str,
    cluster: str | None = None,
    job_name: str | None = None,
    script_args: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    """Submit a Slurm job using a configured profile and a script under remote_repo."""
    current_settings = get_settings()
    target = current_settings.get_cluster(cluster)
    info = current_settings.cluster_info(cluster)
    storage = info.get("storage", {})
    if not isinstance(storage, dict):
        raise ValueError("Cluster storage configuration must be a mapping.")
    remote_repo = storage.get("remote_repo") or target.remote_repo
    run_root = storage.get("run_root") or target.run_root
    if not isinstance(remote_repo, str) or not remote_repo:
        raise ValueError("Cluster must define storage.remote_repo before jobs can be submitted.")

    profile_data = current_settings.get_profile(profile)
    safe_script = validate_remote_relative_path(script, "script")
    safe_job_name = validate_job_name(job_name) if job_name else None
    safe_script_args = validate_argument_list(script_args, "script_args")

    slurm_info = info.get("slurm", {})
    if not isinstance(slurm_info, dict):
        raise ValueError("Cluster Slurm configuration must be a mapping.")
    submit_command = str(slurm_info.get("submit_command", "sbatch"))
    if submit_command != "sbatch":
        raise ValueError("Only the sbatch submit command is currently supported.")

    log_dir = f"{run_root}/logs" if isinstance(run_root, str) and run_root else None
    argv = build_sbatch_argv(
        profile_data,
        safe_script,
        job_name=safe_job_name,
        script_args=safe_script_args,
        cluster_slurm=slurm_info,
        log_dir=log_dir,
    )

    preview = {
        "cluster": target.name,
        "host": target.host,
        "profile": profile,
        "script": safe_script,
        "remote_repo": remote_repo,
        "run_root": run_root,
        "log_dir": log_dir,
        "argv": argv,
        "dry_run": dry_run,
    }
    if dry_run:
        return {
            **preview,
            "submitted": False,
            "job_id": None,
            "stdout": "",
        }

    if log_dir:
        _run_remote_output(
            current_settings,
            target,
            ["mkdir", "-p", log_dir],
            tool="submit_job",
            timeout=30,
        )
    _run_remote_output(
        current_settings,
        target,
        ["test", "-f", safe_script],
        tool="submit_job",
        cwd=remote_repo,
        timeout=30,
    )
    stdout = _run_remote_output(
        current_settings,
        target,
        argv,
        tool="submit_job",
        cwd=remote_repo,
        timeout=30,
    )
    job_id = parse_sbatch_job_id(stdout)
    return {
        **preview,
        "submitted": True,
        "job_id": job_id,
        "stdout": stdout,
        "expected_stdout": f"{log_dir}/%x-{job_id}.out" if log_dir else None,
        "expected_stderr": f"{log_dir}/%x-{job_id}.err" if log_dir else None,
    }


@mcp.tool()
def list_run_files(
    cluster: str | None = None,
    path: str | None = None,
    limit: int = 100,
    max_depth: int = 4,
) -> dict[str, object]:
    """List files under the configured run_root, newest first."""
    current_settings = get_settings()
    target, roots = _file_context(current_settings, cluster)
    safe_limit = validate_limit(limit)
    safe_depth = validate_max_depth(max_depth)
    resolved = resolve_remote_path(path, roots, base="run_root", label="path")
    output = _run_remote_output(
        current_settings,
        target,
        list_files_argv(resolved.path, safe_depth),
        tool="list_run_files",
        timeout=60,
        strip=False,
    )
    files = parse_file_listing(output, roots, safe_limit)
    return {
        "cluster": target.name,
        "host": target.host,
        "path": resolved.path,
        "root": resolved.root_name,
        "relative_path": resolved.relative,
        "limit": safe_limit,
        "max_depth": safe_depth,
        "count": len(files),
        "files": files,
    }


@mcp.tool()
def read_file(
    path: str,
    cluster: str | None = None,
    max_bytes: int = 20000,
    base: str = "run_root",
) -> dict[str, object]:
    """Read the beginning of an allowlisted remote text file."""
    current_settings = get_settings()
    target, roots = _file_context(current_settings, cluster)
    safe_base = validate_file_base(base)
    safe_max_bytes = validate_max_bytes(max_bytes)
    resolved = resolve_remote_path(path, roots, base=safe_base, label="path")
    metadata_result = _run_remote_result(
        current_settings,
        target,
        stat_file_argv(resolved.path),
        tool="read_file",
        timeout=15,
    )
    content = _run_remote_output(
        current_settings,
        target,
        read_file_argv(resolved.path, safe_max_bytes),
        tool="read_file",
        timeout=30,
        strip=False,
    )
    metadata = parse_stat_output(metadata_result.stdout) if metadata_result.returncode == 0 else {}
    size_bytes = metadata.get("size_bytes")
    return {
        "cluster": target.name,
        "host": target.host,
        "path": resolved.path,
        "root": resolved.root_name,
        "relative_path": resolved.relative,
        "max_bytes": safe_max_bytes,
        "truncated": isinstance(size_bytes, int) and size_bytes > safe_max_bytes,
        "metadata": metadata,
        "content": content,
    }


@mcp.tool()
def tail_file(
    path: str,
    cluster: str | None = None,
    lines: int = 100,
    base: str = "run_root",
) -> dict[str, object]:
    """Tail an allowlisted remote text file."""
    current_settings = get_settings()
    target, roots = _file_context(current_settings, cluster)
    safe_base = validate_file_base(base)
    safe_lines = validate_line_count(lines)
    resolved = resolve_remote_path(path, roots, base=safe_base, label="path")
    content = _run_remote_output(
        current_settings,
        target,
        tail_file_argv(resolved.path, safe_lines),
        tool="tail_file",
        timeout=30,
        strip=False,
    )
    return {
        "cluster": target.name,
        "host": target.host,
        "path": resolved.path,
        "root": resolved.root_name,
        "relative_path": resolved.relative,
        "lines": safe_lines,
        "content": content,
    }


@mcp.tool()
def grep_files(
    pattern: str,
    path: str | None = None,
    cluster: str | None = None,
    limit: int = 50,
    base: str = "run_root",
) -> dict[str, object]:
    """Search allowlisted remote files with fixed-string grep."""
    current_settings = get_settings()
    target, roots = _file_context(current_settings, cluster)
    safe_base = validate_file_base(base)
    safe_limit = validate_limit(limit)
    resolved = resolve_remote_path(path, roots, base=safe_base, label="path")
    result = _run_remote_result(
        current_settings,
        target,
        grep_files_argv(pattern, resolved.path),
        tool="grep_files",
        timeout=60,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr.strip() or "remote grep failed")
    matches = parse_grep_output(result.stdout, roots, safe_limit)
    return {
        "cluster": target.name,
        "host": target.host,
        "path": resolved.path,
        "root": resolved.root_name,
        "relative_path": resolved.relative,
        "limit": safe_limit,
        "count": len(matches),
        "matches": matches,
    }


@mcp.tool()
def job_logs(job_id: str, cluster: str | None = None, limit: int = 20) -> dict[str, object]:
    """Find stdout/stderr log files for a Slurm job under run_root."""
    current_settings = get_settings()
    target, roots = _file_context(current_settings, cluster)
    safe_job_id = validate_job_id(job_id)
    safe_limit = validate_limit(limit)
    files = _job_log_files(current_settings, target, roots, safe_job_id, safe_limit)
    return {
        "cluster": target.name,
        "host": target.host,
        "job_id": safe_job_id,
        "count": len(files),
        "files": files,
    }


@mcp.tool()
def tail_job_log(
    job_id: str,
    stream: str = "stdout",
    cluster: str | None = None,
    lines: int = 100,
) -> dict[str, object]:
    """Tail the stdout or stderr log file for a Slurm job."""
    current_settings = get_settings()
    target, roots = _file_context(current_settings, cluster)
    safe_job_id = validate_job_id(job_id)
    safe_lines = validate_line_count(lines)
    files = _job_log_files(current_settings, target, roots, safe_job_id, limit=20)
    selected = select_job_log(files, stream)
    content = _run_remote_output(
        current_settings,
        target,
        tail_file_argv(str(selected["path"]), safe_lines),
        tool="tail_job_log",
        timeout=30,
        strip=False,
    )
    return {
        "cluster": target.name,
        "host": target.host,
        "job_id": safe_job_id,
        "stream": stream,
        "path": selected["path"],
        "root": selected["root"],
        "relative_path": selected["relative_path"],
        "lines": safe_lines,
        "content": content,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alliance Slurm MCP server")
    parser.add_argument(
        "--config",
        default=os.environ.get("ALLIANCE_SLURM_MCP_CONFIG", "config.yaml"),
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    global settings
    args = parse_args()
    settings = load_settings(args.config)
    mcp.run()


def _remote_user(settings: Settings, target: Cluster, tool: str) -> str:
    return target.user or _run_remote_output(settings, target, ["whoami"], tool=tool, timeout=15)


def _command_error(returncode: int, stderr: str) -> str | None:
    if returncode == 0:
        return None
    return stderr.strip() or f"command failed with exit code {returncode}"


def _job_summary(
    queue: list[dict[str, str]],
    accounting: list[dict[str, str]],
) -> dict[str, str] | None:
    if accounting:
        return accounting[0]
    if queue:
        return queue[0]
    return None


def _file_context(settings: Settings, cluster: str | None) -> tuple[Cluster, dict[str, str]]:
    target = settings.get_cluster(cluster)
    info = settings.cluster_info(target.name)
    storage = info.get("storage", {})
    if not isinstance(storage, dict):
        raise ValueError("Cluster storage configuration must be a mapping.")
    return target, allowed_roots_from_storage(storage)


def _job_log_files(
    settings: Settings,
    target: Cluster,
    roots: dict[str, str],
    job_id: str,
    limit: int,
) -> list[dict[str, Any]]:
    run_root = resolve_remote_path(None, roots, base="run_root", label="run_root")
    output = _run_remote_output(
        settings,
        target,
        find_job_logs_argv(run_root.path, job_id),
        tool="job_logs",
        timeout=60,
        strip=False,
    )
    files = parse_file_listing(output, roots, limit)
    return [{**file_info, "stream": classify_job_log(str(file_info["path"]), job_id)} for file_info in files]


def _run_remote_output(
    settings: Settings,
    target: Cluster,
    argv: list[str],
    *,
    tool: str,
    timeout: int,
    cwd: str | None = None,
    strip: bool = True,
) -> str:
    result = _run_remote_result(settings, target, argv, tool=tool, timeout=timeout, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"remote command failed with exit code {result.returncode}")
    return result.stdout.strip() if strip else result.stdout


def _run_remote_result(
    settings: Settings,
    target: Cluster,
    argv: list[str],
    *,
    tool: str,
    timeout: int,
    cwd: str | None = None,
) -> CommandResult:
    event: dict[str, Any] = {
        "tool": tool,
        "cluster": target.name,
        "host": target.host,
        "cwd": cwd,
        "argv": argv,
    }
    try:
        if cwd is None:
            result = remote_command_result(settings.ssh_command, target.host, argv, timeout=timeout)
        else:
            result = remote_command_result_in_dir(
                settings.ssh_command,
                target.host,
                cwd,
                argv,
                timeout=timeout,
            )
    except Exception as exc:
        _write_remote_audit(settings, {**event, "exception": repr(exc)})
        raise

    _write_remote_audit(
        settings,
        {
            **event,
            "returncode": result.returncode,
            "stdout_preview": preview_text(result.stdout),
            "stderr_preview": preview_text(result.stderr),
        },
    )
    return result


def _write_remote_audit(settings: Settings, event: dict[str, Any]) -> None:
    try:
        write_audit_event(settings.audit_log_path(), event)
    except OSError:
        # Audit logging should never make a read/status command fail.
        pass


if __name__ == "__main__":
    main()
