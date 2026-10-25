"""Virtualenv workflows that follow Alliance Canada conventions.

Alliance clusters provide Python through Lmod modules and a local wheelhouse of
prebuilt wheels. The recommended pattern is ``module load python/X.Y``, then
``virtualenv --no-download``, then ``pip install --no-index`` so packages come
from the wheelhouse instead of PyPI.
"""

from __future__ import annotations

import re
from shlex import quote
from typing import Any

from .validators import validate_remote_relative_path


DEFAULT_PYTHON_MODULE = "python"
_MODULE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+/-]{0,99}")
_VENV_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
# PEP 508-ish requirement specifiers without URLs, paths, or shell metacharacters.
_PACKAGE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9._,-]+\])?([<>=!~]=?[A-Za-z0-9.*+_-]+(,[<>=!~]=?[A-Za-z0-9.*+_-]+)*)?")


def validate_module(name: str) -> str:
    if not isinstance(name, str) or not _MODULE_RE.fullmatch(name) or ".." in name:
        raise ValueError(f"Invalid module name: {name!r}")
    return name


def validate_venv_name(name: str) -> str:
    if not isinstance(name, str) or not _VENV_NAME_RE.fullmatch(name):
        raise ValueError("Virtualenv name must be 1-64 letters, numbers, '.', '_', or '-'.")
    return name


def validate_packages(packages: list[str] | None) -> list[str]:
    if packages is None:
        return []
    if not isinstance(packages, list):
        raise ValueError("packages must be a list of requirement strings.")
    safe: list[str] = []
    for package in packages:
        if not isinstance(package, str) or not _PACKAGE_RE.fullmatch(package):
            raise ValueError(f"Invalid package requirement: {package!r}. Use forms like 'torch' or 'numpy>=1.26'.")
        safe.append(package)
    return safe


def resolve_modules(python_config: dict[str, Any], python_module: str | None, extra_modules: list[str] | None) -> list[str]:
    """Modules to load: the Python module first, then configured and extra modules."""
    configured = python_config.get("modules") or []
    if not isinstance(configured, list):
        raise ValueError("python.modules must be a list of module names.")
    python = python_module or python_config.get("module") or DEFAULT_PYTHON_MODULE
    modules: list[str] = []
    for name in (python, *configured, *(extra_modules or [])):
        safe = validate_module(str(name))
        if safe not in modules:
            modules.append(safe)
    return modules


def venv_path(venv_root: str, name: str) -> str:
    return venv_root.rstrip("/") + "/" + validate_venv_name(name)


def module_load_lines(modules: list[str]) -> list[str]:
    if not modules:
        return []
    return ["module load " + " ".join(quote(validate_module(name)) for name in modules)]


def build_venv_script(
    *,
    venv: str,
    modules: list[str],
    remote_repo: str | None,
    requirements: str | None,
    packages: list[str],
    no_index: bool,
    recreate: bool,
) -> str:
    pip_flags = " --no-index" if no_index else ""
    lines = [
        *module_load_lines(modules),
        "module list 2>&1 || true",
        f"VENV={quote(venv)}",
    ]
    if recreate:
        lines.append('rm -rf -- "$VENV"')
    lines.extend(
        [
            'if [ ! -x "$VENV/bin/python" ]; then',
            '  mkdir -p -- "$(dirname -- "$VENV")"',
            '  virtualenv --no-download "$VENV" || python -m venv "$VENV"',
            "fi",
            'source "$VENV/bin/activate"',
            f"python -m pip install{pip_flags} --upgrade pip",
        ]
    )
    if requirements:
        if not remote_repo:
            raise ValueError("A requirements file needs storage.remote_repo to be configured.")
        safe_requirements = validate_remote_relative_path(requirements, "requirements")
        lines.append(f"cd {quote(remote_repo)}")
        lines.append(f"python -m pip install{pip_flags} -r {quote(safe_requirements)}")
    if packages:
        lines.append(f"python -m pip install{pip_flags} " + " ".join(quote(p) for p in packages))
    lines.extend(
        [
            'python -m pip freeze > "$VENV/slurm-mcp-freeze.txt"',
            # Record the modules so job scripts can reproduce the environment.
            f"printf '%s\\n' {' '.join(quote(m) for m in modules)} > \"$VENV/slurm-mcp-modules.txt\"",
            'python -c "import sys; print(\'[slurm-mcp] python\', sys.version)"',
            'echo "[slurm-mcp] packages: $(python -m pip list --format=freeze | wc -l)"',
        ]
    )
    return "\n".join(lines)


def avail_wheels_argv(packages: list[str], python_version: str | None = None) -> list[str]:
    names = []
    for package in validate_packages(packages):
        # avail_wheels takes names and optional --version; strip specifiers.
        names.append(re.split(r"[<>=!~\[]", package, maxsplit=1)[0])
    if not names:
        raise ValueError("Provide at least one package name.")
    command = "avail_wheels " + " ".join(quote(name) for name in names)
    if python_version:
        if not re.fullmatch(r"3\.[0-9]{1,2}", python_version):
            raise ValueError("python_version must look like '3.11'.")
        command += f" --python {python_version}"
    # avail_wheels lives on the CVMFS software stack, which a login shell puts on PATH.
    return ["bash", "-lc", command]


_INSPECT_SCRIPT = r"""
v="$1"
if [ ! -x "$v/bin/python" ]; then echo "exists=0"; exit 0; fi
echo "exists=1"
printf 'modules=%s\n' "$(tr '\n' ' ' < "$v/slurm-mcp-modules.txt" 2>/dev/null)"
printf 'python=%s\n' "$(readlink -f "$v/bin/python")"
echo "__FREEZE__"
cat "$v/slurm-mcp-freeze.txt" 2>/dev/null || true
"""


def inspect_venv_argv(venv: str) -> list[str]:
    return ["bash", "-c", _INSPECT_SCRIPT, "_", venv]


def parse_venv_inspect(output: str) -> dict[str, Any]:
    header, _, freeze = output.partition("__FREEZE__\n")
    fields: dict[str, str] = {}
    for line in header.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            fields[key] = value.strip()
    if fields.get("exists") != "1":
        return {"exists": False, "modules": [], "python": None, "packages": []}
    return {
        "exists": True,
        "modules": fields.get("modules", "").split(),
        "python": fields.get("python") or None,
        "packages": [line for line in freeze.splitlines() if line.strip()],
    }


def activation_snippet(venv: str, modules: list[str]) -> str:
    """Lines a Slurm script needs to use this environment."""
    return "\n".join([*module_load_lines(modules), f"source {quote(venv)}/bin/activate"])
