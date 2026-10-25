"""Detached login-node tasks (virtualenv builds, model downloads).

These run too long for a single MCP call, so each task is a bash script under
``run_root/mcp-tasks/<task_id>/`` started with setsid/nohup. The script records
its host and PID, writes all output to ``output.log``, and writes its exit code
to ``exit_code`` when it finishes. Status is read back from those files, which
works even when a later SSH connection lands on a different login node.
"""

from __future__ import annotations

import re
import secrets
from shlex import quote
from datetime import datetime, timezone
from typing import Any


TASKS_DIRNAME = "mcp-tasks"
_TASK_ID_RE = re.compile(r"[a-z][a-z0-9-]{0,40}-[0-9]{8}-[0-9]{6}-[0-9a-f]{6}")
_KIND_RE = re.compile(r"[a-z][a-z0-9-]{0,40}")


def new_task_id(kind: str, now: datetime | None = None) -> str:
    if not _KIND_RE.fullmatch(kind):
        raise ValueError("Invalid task kind.")
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
    return f"{kind}-{stamp}-{secrets.token_hex(3)}"


def validate_task_id(task_id: str) -> str:
    if not isinstance(task_id, str) or not _TASK_ID_RE.fullmatch(task_id):
        raise ValueError("Invalid task id.")
    return task_id


def tasks_root(run_root: str) -> str:
    return run_root.rstrip("/") + "/" + TASKS_DIRNAME


def task_dir(run_root: str, task_id: str) -> str:
    return tasks_root(run_root) + "/" + validate_task_id(task_id)


def wrap_task_script(task_directory: str, body: str, *, description: str) -> str:
    """Wrap a task body with logging, PID bookkeeping, and exit-code capture."""
    safe_description = description.replace("\n", " ").replace("\r", " ")
    return "\n".join(
        [
            "#!/bin/bash",
            f"# slurm-mcp task: {safe_description}",
            f"TASK_DIR={quote(task_directory)}",
            'exec > "$TASK_DIR/output.log" 2>&1',
            'echo "$(hostname) $$" > "$TASK_DIR/pid"',
            "trap 'echo $? > \"$TASK_DIR/exit_code\"' EXIT",
            "set -eo pipefail",
            "# Lmod's `module` is only defined in login shells; source it if needed.",
            "type module >/dev/null 2>&1 || source /etc/profile >/dev/null 2>&1 || true",
            'echo "[slurm-mcp] started $(date -Is) on $(hostname)"',
            body.rstrip("\n"),
            'echo "[slurm-mcp] finished $(date -Is)"',
            "",
        ]
    )


def launch_task_argv(script_path: str) -> list[str]:
    # setsid + nohup detach the task from the SSH channel so it survives the call returning.
    return [
        "bash",
        "-c",
        'setsid nohup bash -l "$1" > /dev/null 2>&1 < /dev/null & echo $!',
        "_",
        script_path,
    ]


_STATUS_SCRIPT = r"""
d="$1"; n="$2"
if [ ! -d "$d" ]; then echo "missing=1"; exit 0; fi
printf 'exit_code=%s\n' "$(cat "$d/exit_code" 2>/dev/null)"
printf 'pid=%s\n' "$(cat "$d/pid" 2>/dev/null)"
printf 'host_now=%s\n' "$(hostname)"
read -r host pid < "$d/pid" 2>/dev/null || true
if [ -n "$pid" ] && [ "$host" = "$(hostname)" ] && kill -0 "$pid" 2>/dev/null; then echo "alive=1"; else echo "alive=0"; fi
printf 'description=%s\n' "$(sed -n '2s/^# slurm-mcp task: //p' "$d/task.sh" 2>/dev/null)"
echo "__LOG__"
tail -n "$n" "$d/output.log" 2>/dev/null || true
"""


def task_status_argv(task_directory: str, lines: int) -> list[str]:
    return ["bash", "-c", _STATUS_SCRIPT, "_", task_directory, str(lines)]


def parse_task_status(output: str) -> dict[str, Any]:
    header, _, log = output.partition("__LOG__\n")
    fields: dict[str, str] = {}
    for line in header.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            fields[key.strip()] = value.strip()
    if fields.get("missing") == "1":
        return {"state": "missing", "exit_code": None, "log_tail": ""}

    exit_code_text = fields.get("exit_code", "")
    exit_code = int(exit_code_text) if exit_code_text.lstrip("-").isdigit() else None
    pid_host, _, pid = fields.get("pid", "").partition(" ")
    same_host = bool(pid_host) and pid_host == fields.get("host_now")
    if exit_code is not None:
        state = "succeeded" if exit_code == 0 else "failed"
    elif fields.get("alive") == "1":
        state = "running"
    elif not pid_host:
        state = "starting"
    elif same_host:
        # The process is gone without writing an exit code (killed, or node rebooted).
        state = "lost"
    else:
        state = "running_or_unknown"
    return {
        "state": state,
        "exit_code": exit_code,
        "host": pid_host or None,
        "pid": pid or None,
        "checked_from_host": fields.get("host_now"),
        "description": fields.get("description") or None,
        "log_tail": log,
    }


_LIST_SCRIPT = r"""
root="$1"
[ -d "$root" ] || exit 0
for d in "$root"/*/; do
  [ -d "$d" ] || continue
  id="$(basename "$d")"
  code="$(cat "$d/exit_code" 2>/dev/null)"
  desc="$(sed -n '2s/^# slurm-mcp task: //p' "$d/task.sh" 2>/dev/null)"
  printf '%s|%s|%s\n' "$id" "$code" "$desc"
done
"""


def list_tasks_argv(root: str) -> list[str]:
    return ["bash", "-c", _LIST_SCRIPT, "_", root]


def parse_task_list(output: str) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        task_id, _, rest = line.partition("|")
        code, _, description = rest.partition("|")
        exit_code = int(code) if code.lstrip("-").isdigit() else None
        state = "running_or_unknown" if exit_code is None else ("succeeded" if exit_code == 0 else "failed")
        tasks.append(
            {
                "task_id": task_id,
                "state": state,
                "exit_code": exit_code,
                "description": description or None,
            }
        )
    tasks.sort(key=lambda item: item["task_id"].split("-", 1)[-1], reverse=True)
    return tasks


_CANCEL_SCRIPT = r"""
d="$1"
read -r host pid < "$d/pid" 2>/dev/null || { echo "Task has no PID file yet." >&2; exit 3; }
if [ "$host" != "$(hostname)" ]; then echo "Task runs on $host; this connection is on $(hostname)." >&2; exit 4; fi
kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"
"""


def cancel_task_argv(task_directory: str) -> list[str]:
    return ["bash", "-c", _CANCEL_SCRIPT, "_", task_directory]
