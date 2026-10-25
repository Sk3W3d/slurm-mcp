"""Hugging Face Hub caching for clusters whose compute nodes have no internet.

Models and datasets are downloaded on a login node into a shared ``HF_HOME``
(project or scratch storage). Jobs then point ``HF_HOME`` at the same place
and set the offline flags, so ``from_pretrained("org/name")`` resolves from
the cache without network access.
"""

from __future__ import annotations

import json
import re
from shlex import quote
from typing import Any

from .python_env import module_load_lines


HF_TOOLS_VENV_NAME = "_slurm-mcp-hf"
_REPO_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}(/[A-Za-z0-9][A-Za-z0-9._-]{0,95})?")
_REVISION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}")
_PATTERN_RE = re.compile(r"[A-Za-z0-9*?._/\[\]-]{1,200}")
REPO_TYPES = ("model", "dataset", "space")
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}


def validate_repo_id(repo_id: str) -> str:
    if not isinstance(repo_id, str) or not _REPO_ID_RE.fullmatch(repo_id) or ".." in repo_id:
        raise ValueError("repo_id must look like 'org/name'.")
    return repo_id


def validate_repo_type(repo_type: str) -> str:
    if repo_type not in REPO_TYPES:
        raise ValueError(f"repo_type must be one of: {', '.join(REPO_TYPES)}.")
    return repo_type


def validate_revision(revision: str | None) -> str | None:
    if revision is None or revision == "":
        return None
    if not _REVISION_RE.fullmatch(revision) or ".." in revision:
        raise ValueError("Invalid revision.")
    return revision


def validate_patterns(patterns: list[str] | None, label: str) -> list[str] | None:
    if patterns is None:
        return None
    if not isinstance(patterns, list) or not patterns:
        raise ValueError(f"{label} must be a non-empty list of glob patterns.")
    for pattern in patterns:
        if not isinstance(pattern, str) or not _PATTERN_RE.fullmatch(pattern):
            raise ValueError(f"Invalid {label} entry: {pattern!r}")
    return patterns


def job_env(hf_home: str | None, compute_node_internet: bool) -> dict[str, str]:
    """Environment variables Slurm jobs need to use the shared cache."""
    if not hf_home:
        return {}
    env = {"HF_HOME": hf_home}
    if not compute_node_internet:
        env.update(OFFLINE_ENV)
    return env


_DOWNLOAD_PY = r"""
import json, os
from huggingface_hub import snapshot_download
kwargs = json.loads(os.environ["SLURM_MCP_HF_ARGS"])
path = snapshot_download(**kwargs)
print("[slurm-mcp] SNAPSHOT_PATH=" + path)
"""


def build_download_script(
    *,
    hf_home: str,
    modules: list[str],
    venv: str,
    create_venv: bool,
    repo_id: str,
    repo_type: str,
    revision: str | None,
    allow_patterns: list[str] | None,
    ignore_patterns: list[str] | None,
) -> str:
    kwargs: dict[str, Any] = {
        "repo_id": validate_repo_id(repo_id),
        "repo_type": validate_repo_type(repo_type),
    }
    if validate_revision(revision):
        kwargs["revision"] = revision
    if validate_patterns(allow_patterns, "allow_patterns"):
        kwargs["allow_patterns"] = allow_patterns
    if validate_patterns(ignore_patterns, "ignore_patterns"):
        kwargs["ignore_patterns"] = ignore_patterns

    lines = [*module_load_lines(modules), f"VENV={quote(venv)}"]
    if create_venv:
        lines.extend(
            [
                'if [ ! -x "$VENV/bin/python" ]; then',
                '  mkdir -p -- "$(dirname -- "$VENV")"',
                '  virtualenv --no-download "$VENV" || python -m venv "$VENV"',
                "fi",
                'source "$VENV/bin/activate"',
                # Prefer the Alliance wheelhouse; fall back to PyPI (login nodes have internet).
                'python -c "import huggingface_hub" 2>/dev/null || python -m pip install --no-index huggingface_hub '
                "|| python -m pip install huggingface_hub",
            ]
        )
    else:
        lines.extend(
            [
                'source "$VENV/bin/activate"',
                'python -c "import huggingface_hub" || { echo "huggingface_hub is not installed in $VENV" >&2; exit 2; }',
            ]
        )
    lines.extend(
        [
            f"export HF_HOME={quote(hf_home)}",
            'mkdir -p -- "$HF_HOME"',
            "export HF_HUB_DISABLE_TELEMETRY=1",
            # The login node must be online even if the server's own env sets offline flags.
            "unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE",
            f"export SLURM_MCP_HF_ARGS={quote(json.dumps(kwargs, sort_keys=True))}",
            "python - <<'SLURM_MCP_PY'",
            _DOWNLOAD_PY.strip(),
            "SLURM_MCP_PY",
            'du -sh "$HF_HOME/hub" 2>/dev/null || true',
        ]
    )
    return "\n".join(lines)


_CACHE_LIST_SCRIPT = r"""
hub="$1/hub"
[ -d "$hub" ] || exit 0
for repo in "$hub"/models--* "$hub"/datasets--* "$hub"/spaces--*; do
  [ -d "$repo" ] || continue
  name="$(basename "$repo")"
  refs="$(cd "$repo/refs" 2>/dev/null && for r in *; do [ -f "$r" ] && printf '%s=%s ' "$r" "$(cat "$r")"; done)"
  snaps="$(ls "$repo/snapshots" 2>/dev/null | tr '\n' ' ')"
  printf '%s|%s|%s\n' "$name" "$refs" "$snaps"
done
"""


def cache_list_argv(hf_home: str) -> list[str]:
    return ["bash", "-c", _CACHE_LIST_SCRIPT, "_", hf_home]


def parse_cache_list(output: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        name, _, rest = line.partition("|")
        refs_text, _, snapshots_text = rest.partition("|")
        kind, _, encoded = name.partition("--")
        refs: dict[str, str] = {}
        for item in refs_text.split():
            ref, sep, commit = item.partition("=")
            if sep:
                refs[ref] = commit
        entries.append(
            {
                "repo_type": kind.rstrip("s"),
                "repo_id": encoded.replace("--", "/"),
                "refs": refs,
                "snapshots": snapshots_text.split(),
            }
        )
    return entries


def store_token_argv(hf_home: str) -> list[str]:
    """Write a token read from stdin to $HF_HOME/token with owner-only permissions."""
    return [
        "bash",
        "-c",
        'umask 077 && mkdir -p -- "$1" && cat > "$1/token" && chmod 600 -- "$1/token"',
        "_",
        hf_home,
    ]
