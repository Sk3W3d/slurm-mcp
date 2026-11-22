from __future__ import annotations

from copy import deepcopy
import difflib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import yaml


_PROJECT_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
# Placeholders filled from the cluster and the selected project rather than from storage keys.
BUILTIN_PLACEHOLDERS = ("user", "cluster", "project")


def validate_project_name(name: str) -> str:
    if not isinstance(name, str) or not _PROJECT_NAME_RE.fullmatch(name):
        raise ValueError(
            "Invalid project name. Use 1-64 letters, numbers, '.', '_', or '-', starting with a letter or number."
        )
    return name


class NeedsProject(Exception):
    """A template uses {project} (directly or through another key) but no project was selected."""


def expand_template(
    template: str,
    variables: dict[str, str],
    lookup: Callable[[str], str] | None = None,
) -> str:
    """Replace {name} placeholders.

    Built-in names come from variables; any other name goes to lookup(name).
    Raises NeedsProject when {project} is used without a project.
    """

    def substitute(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in variables:
            return variables[name]
        if name == "project":
            raise NeedsProject()
        if name == "user":
            raise ValueError("Template uses {user}, but the cluster has no 'user' configured.")
        if lookup is not None:
            return lookup(name)
        raise ValueError(f"Unknown placeholder {{{name}}}. Use {{user}}, {{cluster}}, or {{project}}.")

    return _PLACEHOLDER_RE.sub(substitute, template)


@dataclass(frozen=True)
class ResolvedStorage:
    values: dict[str, str]
    # Storage keys left out because they need {project} and no project was selected.
    pending_project: tuple[str, ...]
    # Whether the project changed any path (through a {project} template or a project override).
    project_applies: bool


@dataclass(frozen=True)
class Project:
    name: str
    description: str | None = None
    storage: dict[str, str] | None = None
    # Per-cluster storage overrides for this project.
    clusters: dict[str, dict[str, str]] | None = None


@dataclass(frozen=True)
class ClusterDefaults:
    docs: tuple[str, ...] = ()
    policies: dict[str, Any] | None = None
    setup_hints: dict[str, tuple[str, ...]] | None = None
    slurm: dict[str, Any] | None = None
    storage: dict[str, str] | None = None
    python: dict[str, Any] | None = None


@dataclass(frozen=True)
class Cluster:
    name: str
    host: str
    user: str | None = None
    docs: tuple[str, ...] = ()
    remote_repo: str | None = None
    run_root: str | None = None
    policies: dict[str, Any] | None = None
    setup_hints: dict[str, tuple[str, ...]] | None = None
    slurm: dict[str, Any] | None = None
    storage: dict[str, str] | None = None
    python: dict[str, Any] | None = None


@dataclass(frozen=True)
class Settings:
    default_cluster: str
    clusters: dict[str, Cluster]
    ssh_command: tuple[str, ...] = ("ssh",)
    cluster_defaults: ClusterDefaults = field(default_factory=ClusterDefaults)
    profiles: dict[str, Any] | None = None
    config_dir: Path = Path(".")
    audit_log: str | None = ".slurm-mcp-audit.jsonl"
    # SQLite record of projects, jobs, and tasks. Relative to the config file; None disables it.
    state_db: str | None = ".slurm-mcp-state.sqlite3"
    # "auto" picks paramiko on Windows (no usable ControlMaster) and openssh elsewhere.
    ssh_backend: str = "auto"
    ssh_config: str | None = None
    known_hosts: str | None = None
    duo_option: str = "1"
    keepalive_seconds: int = 60
    allow_remote_shell: bool = False
    # DANGEROUS when True: run_remote_command stops asking the user to approve each command.
    remote_shell_auto_approve: bool = False
    # Read-only allowlisted commands (squeue, ls, git log, ...) run without asking.
    remote_shell_auto_approve_readonly: bool = True
    # Extra command names the user trusts as read-only (any arguments).
    remote_shell_readonly_commands: tuple[str, ...] = ()
    max_transfer_bytes: int = 200_000_000
    # Local directories that upload/download/sync tools may touch. Empty means the
    # server's working directory at startup.
    local_roots: tuple[str, ...] = ()
    projects: dict[str, Project] = field(default_factory=dict)
    # Project selected when the server starts; project_use changes it at runtime.
    default_project: str | None = None

    def resolved_ssh_backend(self) -> str:
        if self.ssh_backend != "auto":
            return self.ssh_backend
        return "paramiko" if sys.platform == "win32" else "openssh"

    def local_root_paths(self) -> tuple[Path, ...]:
        if not self.local_roots:
            return (Path.cwd().resolve(),)
        paths: list[Path] = []
        for root in self.local_roots:
            path = Path(root).expanduser()
            if not path.is_absolute():
                path = self.config_dir / path
            paths.append(path.resolve())
        return tuple(paths)

    def get_cluster(self, name: str | None = None) -> Cluster:
        cluster_name = name or self.default_cluster
        try:
            return self.clusters[cluster_name]
        except KeyError as exc:
            known = ", ".join(sorted(self.clusters))
            raise ValueError(f"Unknown cluster '{cluster_name}'. Known clusters: {known}") from exc

    def template_variables(self, name: str | None = None, project: str | None = None) -> dict[str, str]:
        cluster = self.get_cluster(name)
        variables = {"cluster": cluster.name}
        if cluster.user:
            variables["user"] = cluster.user
        if project is not None:
            variables["project"] = validate_project_name(project)
        return variables

    def storage_for(self, name: str | None = None, project: str | None = None) -> ResolvedStorage:
        """Merge the storage layers for a cluster and project, then expand {placeholders}.

        Layers, later wins: cluster_defaults.storage, the cluster's storage, the
        project's storage, and the project's per-cluster storage.
        """
        cluster = self.get_cluster(name)
        raw = _merge_dicts(self.cluster_defaults.storage, cluster.storage)
        if cluster.remote_repo is not None:
            raw["remote_repo"] = cluster.remote_repo
        if cluster.run_root is not None:
            raw["run_root"] = cluster.run_root

        overridden = False
        if project is not None:
            spec = self.projects.get(validate_project_name(project))
            if spec is not None:
                for layer in (spec.storage, (spec.clusters or {}).get(cluster.name)):
                    if layer:
                        raw = _merge_dicts(raw, layer)
                        overridden = True

        variables = self.template_variables(cluster.name, project)
        values, pending, used_project = _expand_storage(raw, variables, cluster.name)
        return ResolvedStorage(
            values=values,
            pending_project=pending,
            project_applies=project is not None and (overridden or used_project),
        )

    def cluster_info(self, name: str | None = None, project: str | None = None) -> dict[str, Any]:
        cluster = self.get_cluster(name)
        resolved = self.storage_for(cluster.name, project)

        return {
            "name": cluster.name,
            "host": cluster.host,
            "user": cluster.user,
            "project": project if resolved.project_applies else None,
            "project_applies": resolved.project_applies,
            "storage_pending_project": list(resolved.pending_project),
            "docs": list(_merge_docs(self.cluster_defaults.docs, cluster.docs)),
            "policies": _merge_dicts(self.cluster_defaults.policies, cluster.policies),
            "setup_hints": _setup_hints_to_lists(
                _merge_dicts(self.cluster_defaults.setup_hints, cluster.setup_hints)
            ),
            "slurm": _merge_dicts(self.cluster_defaults.slurm, cluster.slurm),
            "storage": resolved.values,
            "python": _merge_dicts(self.cluster_defaults.python, cluster.python),
            "profiles": deepcopy(self.profiles or {}),
        }

    def doc_sources(self, name: str | None = None) -> tuple[str, ...]:
        cluster = self.get_cluster(name)
        return _merge_docs(self.cluster_defaults.docs, cluster.docs)

    def get_profile(self, name: str) -> dict[str, Any]:
        profiles = self.profiles or {}
        try:
            profile = profiles[name]
        except KeyError as exc:
            known = ", ".join(sorted(profiles)) or "none"
            raise ValueError(f"Unknown job profile '{name}'. Known profiles: {known}") from exc
        if not isinstance(profile, dict):
            raise ValueError(f"Job profile '{name}' must be a mapping.")
        return deepcopy(profile)

    def audit_log_path(self) -> Path | None:
        return self._config_relative(self.audit_log)

    def state_db_path(self) -> Path | None:
        return self._config_relative(self.state_db)

    def _config_relative(self, value: str | None) -> Path | None:
        if value is None or value == "":
            return None
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        return self.config_dir / path

    def read_doc_source(self, source: str) -> dict[str, Any]:
        parsed = urlparse(source)
        if parsed.scheme in {"http", "https"}:
            return {
                "source": source,
                "kind": "url",
                "content": None,
            }
        if parsed.scheme:
            raise ValueError(f"Unsupported documentation source scheme: {parsed.scheme}")

        path = Path(source)
        if path.is_absolute():
            raise ValueError("Local documentation sources must be relative to the config file.")

        base = self.config_dir.resolve()
        resolved = (base / path).resolve()
        try:
            resolved.relative_to(base)
        except ValueError as exc:
            raise ValueError("Local documentation sources cannot escape the config directory.") from exc

        return {
            "source": source,
            "kind": "local_markdown",
            "path": str(resolved),
            "content": resolved.read_text(encoding="utf-8"),
        }


_TOP_LEVEL_KEYS = frozenset(
    {
        "default_cluster", "clusters", "ssh_command", "cluster_defaults", "profiles", "audit_log", "state_db",
        "ssh_backend", "ssh_config", "known_hosts", "duo_option", "keepalive_seconds", "allow_remote_shell",
        "remote_shell_auto_approve", "remote_shell_auto_approve_readonly", "remote_shell_readonly_commands",
        "max_transfer_bytes", "local_roots", "projects", "default_project",
    }
)
_SECTION_KEYS = frozenset({"docs", "policies", "setup_hints", "slurm", "storage", "python"})
_CLUSTER_KEYS = _SECTION_KEYS | {"host", "user", "remote_repo", "run_root"}
_SLURM_KEYS = frozenset(
    {
        "submit_command", "status_command", "accounting_command", "gpu_request_style", "default_account",
        "cpu_account", "job_env",
    }
)
_PYTHON_KEYS = frozenset({"module", "modules", "venv_name"})
# Storage keys people tend to put one level too high.
_STORAGE_HINT_KEYS = frozenset({"workspace_root", "venv_root", "hf_home", "tasks_root", "record_root"})


def _check_keys(mapping: dict[str, Any], allowed: frozenset[str], label: str) -> None:
    """Reject unknown keys, so a misplaced or misspelled setting is an error instead of silently ignored."""
    for key in mapping:
        if key in allowed:
            continue
        if key in _STORAGE_HINT_KEYS and "storage" in allowed:
            raise ValueError(f"{label}: '{key}' is a storage setting; put it under {label}.storage instead.")
        close = difflib.get_close_matches(str(key), sorted(allowed), n=1)
        suggestion = f" Did you mean '{close[0]}'?" if close else ""
        raise ValueError(f"{label}: unknown setting '{key}'.{suggestion}")


def _check_sections(mapping: dict[str, Any], label: str) -> None:
    for section, allowed in (("slurm", _SLURM_KEYS), ("python", _PYTHON_KEYS)):
        value = mapping.get(section)
        if isinstance(value, dict):
            _check_keys(value, allowed, f"{label}.{section}")


def load_settings(path: str | Path) -> Settings:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config file must contain a YAML mapping.")
    _check_keys(raw, _TOP_LEVEL_KEYS, "config")

    clusters_raw = raw.get("clusters")
    if not isinstance(clusters_raw, dict) or not clusters_raw:
        raise ValueError("Config must define at least one cluster under 'clusters'.")

    defaults_raw = raw.get("cluster_defaults") or {}
    if isinstance(defaults_raw, dict):
        _check_keys(defaults_raw, _SECTION_KEYS, "cluster_defaults")
        _check_sections(defaults_raw, "cluster_defaults")
    defaults = _cluster_defaults(defaults_raw)

    clusters: dict[str, Cluster] = {}
    for name, value in clusters_raw.items():
        if not isinstance(value, dict):
            raise ValueError(f"Cluster '{name}' must be a mapping.")
        _check_keys(value, _CLUSTER_KEYS, f"clusters.{name}")
        _check_sections(value, f"clusters.{name}")
        host = value.get("host")
        if not isinstance(host, str) or not host:
            raise ValueError(f"Cluster '{name}' must define a non-empty host.")
        storage = _storage_mapping(value, f"clusters.{name}.storage")
        clusters[name] = Cluster(
            name=name,
            host=host,
            user=_optional_string(value, "user"),
            docs=_string_tuple(value, "docs"),
            remote_repo=_optional_string(value, "remote_repo") or storage.get("remote_repo"),
            run_root=_optional_string(value, "run_root") or storage.get("run_root"),
            policies=_plain_mapping(value, "policies"),
            setup_hints=_setup_hints(value, "setup_hints"),
            slurm=_plain_mapping(value, "slurm"),
            storage=storage,
            python=_plain_mapping(value, "python"),
        )

    default_cluster = raw.get("default_cluster") or next(iter(clusters))
    if default_cluster not in clusters:
        raise ValueError(f"default_cluster '{default_cluster}' is not defined in clusters.")

    default_project = _optional_string(raw, "default_project")
    if default_project is not None:
        validate_project_name(default_project)

    return Settings(
        default_cluster=str(default_cluster),
        clusters=clusters,
        ssh_command=_string_list(raw, "ssh_command", default=("ssh",)),
        cluster_defaults=defaults,
        profiles=_plain_mapping(raw, "profiles"),
        config_dir=config_path.parent,
        audit_log=_nullable_path_setting(raw, "audit_log", ".slurm-mcp-audit.jsonl"),
        state_db=_nullable_path_setting(raw, "state_db", ".slurm-mcp-state.sqlite3"),
        ssh_backend=_choice(raw, "ssh_backend", {"auto", "openssh", "paramiko"}, "auto"),
        ssh_config=_optional_string(raw, "ssh_config"),
        known_hosts=_optional_string(raw, "known_hosts"),
        duo_option=str(raw.get("duo_option", "1")),
        keepalive_seconds=_positive_int(raw, "keepalive_seconds", 60),
        allow_remote_shell=_bool(raw, "allow_remote_shell", False),
        remote_shell_auto_approve=_bool(raw, "remote_shell_auto_approve", False),
        remote_shell_auto_approve_readonly=_bool(raw, "remote_shell_auto_approve_readonly", True),
        remote_shell_readonly_commands=_command_names(raw, "remote_shell_readonly_commands"),
        max_transfer_bytes=_positive_int(raw, "max_transfer_bytes", 200_000_000),
        local_roots=_string_tuple(raw, "local_roots"),
        projects=_projects(raw.get("projects"), clusters),
        default_project=default_project,
    )


def _command_names(mapping: dict[str, Any], key: str) -> tuple[str, ...]:
    names = _string_tuple(mapping, key)
    for name in names:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", name):
            raise ValueError(f"'{key}' must list plain command names (no paths, spaces, or shell syntax): {name!r}")
    return names


def _projects(raw: Any, clusters: dict[str, Cluster]) -> dict[str, Project]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("'projects' must be a mapping of project names to settings.")
    projects: dict[str, Project] = {}
    for name, value in raw.items():
        validate_project_name(name)
        value = value or {}
        if not isinstance(value, dict):
            raise ValueError(f"Project '{name}' must be a mapping.")
        unknown = set(value) - {"description", "storage", "clusters"}
        if unknown:
            raise ValueError(f"Project '{name}' has unknown keys: {', '.join(sorted(unknown))}.")
        clusters_raw = value.get("clusters") or {}
        if not isinstance(clusters_raw, dict):
            raise ValueError(f"projects.{name}.clusters must be a mapping.")
        per_cluster: dict[str, dict[str, str]] = {}
        for cluster_name, cluster_value in clusters_raw.items():
            if cluster_name not in clusters:
                raise ValueError(f"projects.{name}.clusters.{cluster_name}: unknown cluster.")
            if not isinstance(cluster_value, dict) or set(cluster_value) - {"storage"}:
                raise ValueError(f"projects.{name}.clusters.{cluster_name} may only set 'storage'.")
            per_cluster[cluster_name] = _storage_mapping(
                cluster_value, f"projects.{name}.clusters.{cluster_name}.storage"
            )
        projects[name] = Project(
            name=name,
            description=_optional_string(value, "description"),
            storage=_storage_mapping(value, f"projects.{name}.storage"),
            clusters=per_cluster,
        )
    return projects


def _storage_mapping(mapping: dict[str, Any], label: str) -> dict[str, str]:
    storage = _string_mapping(mapping, "storage")
    reserved = sorted(set(storage) & set(BUILTIN_PLACEHOLDERS))
    if reserved:
        raise ValueError(f"{label} cannot define {', '.join(reserved)}: those names are reserved placeholders.")
    return storage


def _expand_storage(
    raw: dict[str, Any],
    variables: dict[str, str],
    cluster_name: str,
) -> tuple[dict[str, str], tuple[str, ...], bool]:
    """Expand {placeholders} in storage values. Values may reference other storage keys.

    Returns the expanded values, the keys that need a project that was not
    given, and whether {project} was substituted anywhere.
    """
    resolved: dict[str, str] = {}
    pending: set[str] = set()
    used_project = False

    def resolve(key: str, stack: tuple[str, ...]) -> str:
        nonlocal used_project
        if key in resolved:
            return resolved[key]
        if key in pending:
            raise NeedsProject()
        if key in stack:
            chain = " -> ".join((*stack, key))
            raise ValueError(f"Cluster '{cluster_name}': storage placeholders form a cycle: {chain}.")

        def lookup(name: str) -> str:
            if name not in raw:
                raise ValueError(
                    f"Cluster '{cluster_name}': storage.{key} uses unknown placeholder {{{name}}}. "
                    "Use {user}, {cluster}, {project}, or another storage key."
                )
            return resolve(name, (*stack, key))

        try:
            value = expand_template(raw[key], variables, lookup)
        except ValueError as exc:
            if "no 'user'" in str(exc):
                raise ValueError(
                    f"Cluster '{cluster_name}': storage.{key} uses {{user}}, but the cluster sets no user."
                ) from exc
            raise
        if "project" in variables and "{project}" in raw[key]:
            used_project = True
        resolved[key] = value
        return value

    for key in raw:
        try:
            resolve(key, ())
        except NeedsProject:
            pending.add(key)
    return resolved, tuple(sorted(pending)), used_project


def _choice(mapping: dict[str, Any], key: str, choices: set[str], default: str) -> str:
    value = mapping.get(key, default)
    if value not in choices:
        raise ValueError(f"'{key}' must be one of: {', '.join(sorted(choices))}.")
    return str(value)


def _positive_int(mapping: dict[str, Any], key: str, default: int) -> int:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"'{key}' must be a positive integer.")
    return value


def _bool(mapping: dict[str, Any], key: str, default: bool) -> bool:
    value = mapping.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"'{key}' must be true or false.")
    return value


def _cluster_defaults(raw: Any) -> ClusterDefaults:
    if not isinstance(raw, dict):
        raise ValueError("'cluster_defaults' must be a mapping when set.")
    return ClusterDefaults(
        docs=_string_tuple(raw, "docs"),
        policies=_plain_mapping(raw, "policies"),
        setup_hints=_setup_hints(raw, "setup_hints"),
        slurm=_plain_mapping(raw, "slurm"),
        storage=_string_mapping(raw, "storage"),
        python=_plain_mapping(raw, "python"),
    )


def _optional_string(mapping: dict[str, Any], key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"'{key}' must be a string when set.")
    return value


def _nullable_path_setting(mapping: dict[str, Any], key: str, default: str) -> str | None:
    if key not in mapping:
        return default
    value = mapping[key]
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError(f"'{key}' must be a string, null, or an empty string.")
    return value


def _string_list(
    mapping: dict[str, Any],
    key: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    value = mapping.get(key)
    if value is None:
        return default
    if not isinstance(value, list) or not value:
        raise ValueError(f"'{key}' must be a non-empty list of strings.")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"'{key}' must be a non-empty list of strings.")
    return tuple(value)


def _string_tuple(mapping: dict[str, Any], key: str) -> tuple[str, ...]:
    value = mapping.get(key)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"'{key}' must be a list of strings.")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"'{key}' must be a list of non-empty strings.")
    return tuple(value)


def _string_mapping(mapping: dict[str, Any], key: str) -> dict[str, str]:
    value = mapping.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"'{key}' must be a mapping when set.")
    result: dict[str, str] = {}
    for item_key, item_value in value.items():
        if not isinstance(item_key, str) or not item_key:
            raise ValueError(f"'{key}' keys must be non-empty strings.")
        if not isinstance(item_value, str) or not item_value:
            raise ValueError(f"'{key}.{item_key}' must be a non-empty string.")
        result[item_key] = item_value
    return result


def _setup_hints(mapping: dict[str, Any], key: str) -> dict[str, tuple[str, ...]]:
    value = mapping.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"'{key}' must be a mapping when set.")

    result: dict[str, tuple[str, ...]] = {}
    for hint_name, commands in value.items():
        if not isinstance(hint_name, str) or not hint_name:
            raise ValueError(f"'{key}' keys must be non-empty strings.")
        if not isinstance(commands, list) or not commands:
            raise ValueError(f"'{key}.{hint_name}' must be a non-empty list of strings.")
        if not all(isinstance(command, str) and command for command in commands):
            raise ValueError(f"'{key}.{hint_name}' must be a non-empty list of strings.")
        result[hint_name] = tuple(commands)
    return result


def _plain_mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"'{key}' must be a mapping when set.")
    return _plain_data(value, key)


def _plain_data(value: Any, path: str) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for item_key, item_value in value.items():
            if not isinstance(item_key, str) or not item_key:
                raise ValueError(f"'{path}' keys must be non-empty strings.")
            result[item_key] = _plain_data(item_value, f"{path}.{item_key}")
        return result
    if isinstance(value, list):
        return [_plain_data(item, path) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ValueError(f"'{path}' contains unsupported value type: {type(value).__name__}")


def _merge_docs(default_docs: tuple[str, ...], cluster_docs: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    merged: list[str] = []
    for source in (*default_docs, *cluster_docs):
        if source in seen:
            continue
        seen.add(source)
        merged.append(source)
    return tuple(merged)


def _merge_dicts(
    defaults: dict[str, Any] | None,
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = deepcopy(defaults or {})
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _setup_hints_to_lists(hints: dict[str, Any]) -> dict[str, list[str]]:
    return {name: list(commands) for name, commands in hints.items()}
