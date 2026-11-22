"""Refuse clearly destructive commands sent to run_remote_command.

This is a guard against accidents, not a sandbox: a determined command can
always be written so that these checks miss it. The real safeguard is the
person approving each command (see remote_shell_auto_approve). The guard
catches the mistakes that would be hard to undo: deleting a whole home,
project, or scratch folder, cancelling jobs in bulk, changing permissions on
shared folders, touching SSH keys, reaching other hosts, or piping a download
into a shell.

Checks run on the command split into simple commands (at ;, &&, ||, |, &, and
newlines). Commands passed to bash -c, sh -c, or eval are checked too.
"""

from __future__ import annotations

import posixpath
import re
import shlex
from typing import Iterable

_SEPARATORS = {";", "&&", "||", "|", "&", "\n", "|&", ";;", "(", ")"}
_SHELLS = {"bash", "sh", "zsh", "dash"}
# Command prefixes that run the next word as the command.
_WRAPPERS = {"command", "builtin", "exec", "nice", "nohup", "time", "env", "xargs", "timeout", "stdbuf", "srun"}

# Targets that must never be deleted recursively: filesystem, home, project, and
# scratch roots (also through ~, $HOME, $SCRATCH, $PROJECT, and ~/projects/<group>),
# plus wildcards and the current or parent directory.
_ROOT_TARGET = re.compile(
    r"""^(
        /+
      | ~/* | ~/+projects(/+[^/]+)?/* | ~[^/]+/*
      | \$\{?(HOME|SCRATCH|PROJECT|USER)\}?/*
      | /+(home|scratch|project|projects|nearline|lustre\d*)(/+[^/]+){0,2}/*
      | \*+ | \.+/* | \.\.?/+\*
    )$""",
    re.VERBOSE,
)
_SSH_TOOLS = {"ssh", "scp", "sftp", "ssh-copy-id", "sshfs"}
_PIPE_TO_SHELL = re.compile(r"\b(curl|wget)\b[^|;&]*\|\s*(sudo\s+)?(ba|z|da)?sh\b")


class _Targets:
    """Decides whether a path is one that must never be deleted recursively.

    Protected: generic roots (see _ROOT_TARGET), every configured storage
    folder, and every folder above one (deleting a parent deletes the folder).
    Subfolders of a configured folder are fine to delete.
    """

    def __init__(self, user: str | None, protected: Iterable[str], cwd: str | None = None) -> None:
        self.cwd = cwd
        self.home = f"/home/{user}" if user else None
        self.scratch = f"/scratch/{user}" if user else None
        self.protected: set[str] = set()
        for path in [*protected, self.home, self.scratch]:
            if not path:
                continue
            path = posixpath.normpath(path)
            while True:
                self.protected.add(path)
                parent = posixpath.dirname(path)
                if parent == path:
                    break
                path = parent

    def expand(self, target: str) -> str | None:
        """Absolute form of a target where it can be worked out (~, $HOME, $SCRATCH), else None."""
        if self.home and (target == "~" or target.startswith("~/")):
            target = self.home + target[1:]
        for variable, value in (("HOME", self.home), ("SCRATCH", self.scratch)):
            for spelling in (f"${variable}", f"${{{variable}}}"):
                if value and (target == spelling or target.startswith(spelling + "/")):
                    target = value + target[len(spelling) :]
        if target.startswith("$") or target.startswith("~"):
            return None  # another variable or user's home: cannot tell
        if not target.startswith("/"):
            if not self.cwd:
                return None
            target = posixpath.join(self.cwd, target)
        return posixpath.normpath(target)

    def is_protected(self, target: str) -> bool:
        if _ROOT_TARGET.match(target):
            return True
        absolute = self.expand(target.rstrip("*") or target)
        return absolute is not None and absolute in self.protected


def check_command(
    command: str,
    *,
    user: str | None = None,
    protected: Iterable[str] = (),
    cwd: str | None = None,
) -> list[str]:
    """Return the reasons a command is refused; an empty list means it may run.

    protected lists configured folders (repo, runs, venvs, ...) that must not be
    deleted wholesale, along with every folder above them. cwd is where the
    command runs, used to resolve relative paths.
    """
    targets = _Targets(user, protected, cwd)
    problems: list[str] = []
    if _PIPE_TO_SHELL.search(command):
        problems.append("Piping a download into a shell (curl/wget ... | sh) is not allowed.")
    if re.search(r"(^|[\s'\"=:/])~?/?\.ssh(/|\b)|authorized_keys", command):
        problems.append("Commands that touch ~/.ssh or authorized_keys are not allowed.")
    for words in _simple_commands(command):
        problems.extend(_check_simple(words, targets))
    # Keep the first occurrence of each message, in order.
    return list(dict.fromkeys(problems))


def _simple_commands(command: str, depth: int = 0) -> list[list[str]]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()\n")
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    lexer.commenters = "#"
    try:
        tokens = list(lexer)
    except ValueError:
        # Unbalanced quotes: fall back to plain whitespace splitting.
        tokens = command.replace("\n", " ; ").split()

    commands: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token in _SEPARATORS:
            if current:
                commands.append(current)
            current = []
        else:
            current.append(token)
    if current:
        commands.append(current)

    nested: list[list[str]] = []
    if depth < 3:
        for words in commands:
            for index, word in enumerate(words):
                name = word.rsplit("/", 1)[-1]
                if name in _SHELLS and index + 2 < len(words) and words[index + 1] in {"-c", "-lc", "-ic"}:
                    nested.extend(_simple_commands(words[index + 2], depth + 1))
                elif name == "eval" and index + 1 < len(words):
                    nested.extend(_simple_commands(" ".join(words[index + 1 :]), depth + 1))
    return commands + nested


def _command_words(words: list[str]) -> list[str]:
    """Drop leading variable assignments and wrappers like env/nohup/xargs to find the real command."""
    index = 0
    while index < len(words):
        word = words[index]
        name = word.rsplit("/", 1)[-1]
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", word):
            index += 1
        elif name in _WRAPPERS:
            index += 1
            # Skip the wrapper's own options (e.g. xargs -0, timeout 10, nice -n 5).
            while index < len(words) and (words[index].startswith("-") or words[index].isdigit()):
                index += 1
        else:
            break
    return words[index:]


def _check_simple(words: list[str], targets: _Targets) -> list[str]:
    words = _command_words(words)
    if not words:
        return []
    name = words[0].rsplit("/", 1)[-1]
    args = words[1:]
    flags = [arg for arg in args if arg.startswith("-") and arg != "-"]
    operands = [arg for arg in args if not arg.startswith("-")]

    if name == "rm":
        recursive = any(
            arg in {"--recursive", "-r", "-R"} or (not arg.startswith("--") and re.search(r"[rR]", arg)) for arg in flags
        )
        if recursive:
            dangerous = [arg for arg in operands if targets.is_protected(arg)]
            if dangerous:
                return [
                    f"Recursive rm of {', '.join(dangerous)} is not allowed: it is a configured folder or one "
                    "above it. Delete specific subfolders instead."
                ]
    if name == "find" and ("-delete" in args or _execs_rm(args)):
        starts = [arg for arg in args[: _first_expression(args)] if not arg.startswith("-")]
        dangerous = [arg for arg in starts if targets.is_protected(arg)]
        if dangerous:
            return [f"find ... -delete on {', '.join(dangerous)} is not allowed: use a specific subfolder."]
    if name == "scancel":
        return ["scancel is not allowed in the shell: use the cancel_job tool, which only cancels your own jobs."]
    if name in {"chmod", "chown", "chgrp", "setfacl"} and any(
        flag in {"-R", "--recursive"} or (not flag.startswith("--") and "R" in flag) for flag in flags
    ):
        return [f"Recursive {name} is not allowed: it can change permissions on folders your group shares."]
    if name in _SSH_TOOLS:
        return [f"{name} to other hosts is not allowed from the cluster shell."]
    if name == "rsync" and any(re.match(r"^[^/\s]+:", arg) for arg in operands):
        return ["rsync to or from another host is not allowed from the cluster shell."]
    if name in {"sudo", "su", "doas"}:
        return [f"{name} is not allowed."]
    if name == "git" and "push" in args and any(
        arg in {"-f", "--force", "--force-with-lease", "--mirror", "--delete"} or arg.startswith("--force")
        or (arg.startswith("+") and ":" in arg)
        for arg in args
    ):
        return ["Force-pushing or deleting remote branches with git is not allowed."]
    if name == "crontab" and any(arg in {"-r", "-e"} for arg in args):
        return ["Editing or removing your crontab is not allowed from the shell."]
    return []


def _first_expression(args: list[str]) -> int:
    for index, arg in enumerate(args):
        if arg.startswith("-") or arg in {"(", "!"}:
            return index
    return len(args)


def _execs_rm(args: list[str]) -> bool:
    for index, arg in enumerate(args):
        if arg in {"-exec", "-execdir", "-ok", "-okdir"} and index + 1 < len(args):
            if args[index + 1].rsplit("/", 1)[-1] == "rm":
                return True
    return False


__all__ = ["check_command"]
