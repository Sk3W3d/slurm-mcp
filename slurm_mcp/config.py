from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml


@dataclass(frozen=True)
class ClusterDefaults:
    docs: tuple[str, ...] = ()
    policies: dict[str, Any] | None = None
    setup_hints: dict[str, tuple[str, ...]] | None = None
    slurm: dict[str, Any] | None = None
    storage: dict[str, str] | None = None


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


@dataclass(frozen=True)
class Settings:
    default_cluster: str
    clusters: dict[str, Cluster]
    ssh_command: tuple[str, ...] = ("ssh",)
    cluster_defaults: ClusterDefaults = field(default_factory=ClusterDefaults)
    profiles: dict[str, Any] | None = None
    config_dir: Path = Path(".")
    audit_log: str | None = ".slurm-mcp-audit.jsonl"

    def get_cluster(self, name: str | None = None) -> Cluster:
        cluster_name = name or self.default_cluster
        try:
            return self.clusters[cluster_name]
        except KeyError as exc:
            known = ", ".join(sorted(self.clusters))
            raise ValueError(f"Unknown cluster '{cluster_name}'. Known clusters: {known}") from exc

    def cluster_info(self, name: str | None = None) -> dict[str, Any]:
        cluster = self.get_cluster(name)
        storage = _merge_dicts(self.cluster_defaults.storage, cluster.storage)
        if cluster.remote_repo is not None:
            storage["remote_repo"] = cluster.remote_repo
        if cluster.run_root is not None:
            storage["run_root"] = cluster.run_root

        return {
            "name": cluster.name,
            "host": cluster.host,
            "user": cluster.user,
            "docs": list(_merge_docs(self.cluster_defaults.docs, cluster.docs)),
            "policies": _merge_dicts(self.cluster_defaults.policies, cluster.policies),
            "setup_hints": _setup_hints_to_lists(
                _merge_dicts(self.cluster_defaults.setup_hints, cluster.setup_hints)
            ),
            "slurm": _merge_dicts(self.cluster_defaults.slurm, cluster.slurm),
            "storage": storage,
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
        if self.audit_log is None or self.audit_log == "":
            return None
        path = Path(self.audit_log)
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


def load_settings(path: str | Path) -> Settings:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config file must contain a YAML mapping.")

    clusters_raw = raw.get("clusters")
    if not isinstance(clusters_raw, dict) or not clusters_raw:
        raise ValueError("Config must define at least one cluster under 'clusters'.")

    defaults = _cluster_defaults(raw.get("cluster_defaults") or {})

    clusters: dict[str, Cluster] = {}
    for name, value in clusters_raw.items():
        if not isinstance(value, dict):
            raise ValueError(f"Cluster '{name}' must be a mapping.")
        host = value.get("host")
        if not isinstance(host, str) or not host:
            raise ValueError(f"Cluster '{name}' must define a non-empty host.")
        storage = _string_mapping(value, "storage")
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
        )

    default_cluster = raw.get("default_cluster") or next(iter(clusters))
    if default_cluster not in clusters:
        raise ValueError(f"default_cluster '{default_cluster}' is not defined in clusters.")

    return Settings(
        default_cluster=str(default_cluster),
        clusters=clusters,
        ssh_command=_string_list(raw, "ssh_command", default=("ssh",)),
        cluster_defaults=defaults,
        profiles=_plain_mapping(raw, "profiles"),
        config_dir=config_path.parent,
        audit_log=_audit_log_setting(raw),
    )


def _cluster_defaults(raw: Any) -> ClusterDefaults:
    if not isinstance(raw, dict):
        raise ValueError("'cluster_defaults' must be a mapping when set.")
    return ClusterDefaults(
        docs=_string_tuple(raw, "docs"),
        policies=_plain_mapping(raw, "policies"),
        setup_hints=_setup_hints(raw, "setup_hints"),
        slurm=_plain_mapping(raw, "slurm"),
        storage=_string_mapping(raw, "storage"),
    )


def _optional_string(mapping: dict[str, Any], key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"'{key}' must be a string when set.")
    return value


def _audit_log_setting(mapping: dict[str, Any]) -> str | None:
    if "audit_log" not in mapping:
        return ".slurm-mcp-audit.jsonl"
    value = mapping["audit_log"]
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError("'audit_log' must be a string, null, or an empty string.")
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
