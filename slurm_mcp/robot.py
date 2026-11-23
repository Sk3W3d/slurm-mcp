"""Commands for Alliance automation ("robot") nodes.

Automation nodes accept only constrained SSH keys whose command= points to a
wrapper such as allowed_commands.sh (github.com/ComputeCanada/software-stack-custom,
bin/computecanada/allowed_commands). The wrapper:

- matches $SSH_ORIGINAL_COMMAND by prefix against a list (ls, cat, cd, echo,
  uname, id, groups, mv, cp, rm, mkdir, python*, git, tar/dar/gzip/zip/bzip2,
  rsync, scp, the SFTP server, squeue, scancel, sbatch, scontrol, sq*, sinfo,
  sacct), and
- runs it as `$SSH_ORIGINAL_COMMAND` unquoted: plain word splitting and
  globbing, no shell. Quotes are passed through literally, and there are no
  pipes, redirects, &&, or cd for the next command.
- On rejection it prints "Command rejected by <script>: ..." and exits 0.

So a robot command is a single program with arguments that contain no
whitespace, quotes, or glob characters, sent without any quoting.
"""

from __future__ import annotations

import re

# Prefixes accepted by allowed_commands.sh (the "all of the above" wrapper).
ALLOWED_PREFIXES = (
    "ls", "cat", "cd", "echo", "uname", "id", "groups", "mv", "cp", "rm", "mkdir", "python", "git",
    "tar", "dar", "gzip", "zip", "bzip2", "rsync", "/usr/libexec/openssh/sftp-server", "scp",
    "squeue", "scancel", "sbatch", "scontrol", "sq", "sinfo", "sacct",
)
_REJECTED = re.compile(r"^Command rejected by \S+:")
_UNSAFE = re.compile(r"[\s'\"\\`*?\[\]{}]")


class RobotCommandError(ValueError):
    """The command cannot be sent to an automation node as-is."""


def command_string(argv: list[str]) -> str:
    """Join argv for the wrapper, refusing arguments that word splitting or globbing would change."""
    if not argv:
        raise RobotCommandError("empty command")
    if not argv[0].startswith(ALLOWED_PREFIXES):
        raise RobotCommandError(f"'{argv[0]}' is not accepted by the automation node wrapper")
    for arg in argv:
        if not arg:
            raise RobotCommandError("empty arguments cannot be passed to the automation node")
        if _UNSAFE.search(arg) and not (arg.startswith("--exclude=") and not re.search(r"[\s'\"\\`]", arg)):
            raise RobotCommandError(
                f"argument {arg!r} contains spaces, quotes, or wildcards, which the automation node would mangle"
            )
    return " ".join(argv)


def was_rejected(stdout: str) -> bool:
    return bool(_REJECTED.match(stdout.lstrip()))
