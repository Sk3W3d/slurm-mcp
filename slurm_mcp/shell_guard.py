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
from typing import Callable, Iterable

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


# -- read-only allowlist ---------------------------------------------------------------
#
# Commands on this list run without asking the user, as long as the whole command
# line is made only of them (joined by |, &&, ||, ;) and none of the following
# appear: output redirection to a file, $(...) or backticks, background &, (sub)shells,
# or VAR=value prefixes. Each command also gets an argument check, since read-only
# depends on arguments too (git branch -D, find -delete, sort -o, scontrol update).
# Anything else falls back to asking the user; nothing here refuses a command.

ArgCheck = Callable[[list[str]], "str | None"]


def _any(_: list[str]) -> str | None:
    return None


def _short_flag_letters(args: list[str]) -> set[str]:
    letters: set[str] = set()
    for arg in args:
        if arg.startswith("-") and not arg.startswith("--") and len(arg) > 1 and not arg[1:].isdigit():
            letters.update(arg[1:])
    return letters


def _forbid(*options: str) -> ArgCheck:
    """Reject the given options.

    --long options also match --long=value; two-character options like -o also
    match inside bundles (-fo); single-dash words like find's -exec match exactly.
    """
    longs = [option for option in options if option.startswith("--")]
    shorts = {option[1] for option in options if not option.startswith("--") and len(option) == 2}
    words = [option for option in options if not option.startswith("--") and len(option) != 2]

    def check(args: list[str]) -> str | None:
        for arg in args:
            if any(arg == option or arg.startswith(option + "=") for option in longs) or arg in words:
                return f"option {arg} needs approval"
        found = _short_flag_letters(args) & shorts
        if found:
            return f"option -{''.join(sorted(found))} needs approval"
        return None

    return check


def _positionals(args: list[str]) -> list[str]:
    return [arg for arg in args if not arg.startswith("-")]


def _subcommand(allowed: set[str], *, also: ArgCheck = _any, optional: bool = False) -> ArgCheck:
    """The first non-option argument must be one of the read-only subcommands."""

    def check(args: list[str]) -> str | None:
        positionals = _positionals(args)
        if not positionals:
            return None if optional else "no subcommand given"
        if positionals[0] not in allowed:
            return f"subcommand '{positionals[0]}' is not read-only"
        return also(args)

    return check


def _all(*checks: ArgCheck) -> ArgCheck:
    def check(args: list[str]) -> str | None:
        for item in checks:
            problem = item(args)
            if problem:
                return problem
        return None

    return check


def _max_positionals(count: int) -> ArgCheck:
    def check(args: list[str]) -> str | None:
        return None if len(_positionals(args)) <= count else "too many file arguments (the last one may be written)"

    return check


def _only_options(*prefixes: str) -> ArgCheck:
    def check(args: list[str]) -> str | None:
        for arg in args:
            if arg.startswith("-") and not any(arg == prefix or arg.startswith(prefix) for prefix in prefixes):
                return f"option {arg} is not a read-only query"
        return None

    return check


_GIT_READ_ONLY = {
    "status", "log", "diff", "show", "rev-parse", "ls-files", "ls-tree", "describe", "blame", "shortlog",
    "cat-file", "grep", "branch", "tag", "remote", "reflog", "stash", "rev-list", "show-ref", "count-objects",
}


def _git(args: list[str]) -> str | None:
    index = 0
    while index < len(args) and args[index].startswith("-"):
        option = args[index]
        if option in {"-C"}:
            index += 2
            continue
        if option in {"--no-pager", "-P", "--no-optional-locks"}:
            index += 1
            continue
        return f"git option {option} is not allowed without approval"
    if index >= len(args):
        return "no git subcommand given"
    sub, rest = args[index], args[index + 1 :]
    if sub not in _GIT_READ_ONLY:
        return f"git {sub} is not read-only"
    if any(arg == "--output" or arg.startswith("--output=") or arg == "--ext-diff" for arg in rest):
        return "git --output/--ext-diff can write or run programs"
    positionals = _positionals(rest)
    if sub == "branch":
        listing = {"-a", "-r", "-v", "-vv", "--all", "--remotes", "--list", "-l", "--show-current", "--merged",
                   "--no-merged", "--contains", "--no-contains", "--sort", "--format", "--color", "--no-color",
                   "--points-at"}
        for arg in rest:
            if arg.startswith("-") and arg.split("=", 1)[0] not in listing:
                return f"git branch {arg} can change branches"
        if positionals and not ({"--list", "-l", "--contains", "--merged", "--no-merged", "--points-at"} & set(rest)):
            return "git branch <name> creates a branch"
    if sub == "tag" and positionals and not ({"-l", "--list", "--contains", "--points-at"} & set(rest)):
        return "git tag <name> creates a tag"
    if sub == "tag" and _forbid("-d", "-a", "-s", "-f", "--delete", "--annotate", "--sign", "--force")(rest):
        return "git tag option can change tags"
    if sub == "remote" and positionals and positionals[0] not in {"show", "get-url"}:
        return f"git remote {positionals[0]} changes remotes"
    if sub == "reflog" and positionals and positionals[0] not in {"show"}:
        return f"git reflog {positionals[0]} changes the reflog"
    if sub == "stash" and (not positionals or positionals[0] not in {"list", "show"}):
        return "only git stash list/show are read-only"
    return None


def _python(args: list[str]) -> str | None:
    if args in (["--version"], ["-V"]):
        return None
    if len(args) >= 3 and args[:2] == ["-m", "pip"]:
        return _PIP_CHECK(args[2:])
    return "python can run arbitrary code"


_PIP_CHECK = _subcommand({"list", "show", "freeze", "check", "inspect", "--version", "-V"})

READ_ONLY_COMMANDS: dict[str, ArgCheck] = {
    # Slurm: queue, accounting, and cluster state.
    "squeue": _any,
    "sinfo": _any,
    "sacct": _any,
    "sshare": _any,
    "sprio": _any,
    "sstat": _any,
    "seff": _any,
    "sreport": _any,
    "scontrol": _subcommand({"show"}),
    "sacctmgr": _subcommand({"show", "list"}),
    # Alliance helpers.
    "diskusage_report": _any,
    "quota": _any,
    "partition-stats": _any,
    "avail_wheels": _any,
    "lfs": _subcommand({"quota", "df", "getstripe"}),
    "module": _subcommand({"avail", "spider", "list", "show", "whatis", "keyword", "help", "overview", "--version"}),
    "ml": _subcommand({"avail", "spider", "list", "show", "whatis", "keyword", "help", "overview"}, optional=True),
    # Python environments (inspection only).
    "pip": _PIP_CHECK,
    "pip3": _PIP_CHECK,
    "python": _python,
    "python3": _python,
    # Version control (read-only subcommands only).
    "git": _git,
    # GPUs (queries only; no resets or setting changes).
    "nvidia-smi": _only_options("--query", "--format", "-q", "-L", "--list-gpus", "-i", "--id", "-d", "--display"),
    # Files: listing, reading, searching, sizes.
    "ls": _any,
    "tree": _forbid("-o"),
    "du": _any,
    "df": _any,
    "stat": _any,
    "file": _forbid("-C", "--compile"),
    "cat": _any,
    "head": _any,
    "tail": _forbid("-f", "-F", "--follow", "--retry"),
    "wc": _any,
    "grep": _any,
    "egrep": _any,
    "fgrep": _any,
    "zgrep": _any,
    "zcat": _any,
    "rg": _forbid("--pre"),
    "find": _forbid("-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprint", "-fprint0", "-fprintf", "-fls"),
    "cut": _any,
    "tr": _any,
    "column": _any,
    "nl": _any,
    "sort": _forbid("-o", "--output"),
    "uniq": _max_positionals(1),
    "diff": _any,
    "cmp": _any,
    "jq": _any,
    "md5sum": _any,
    "sha1sum": _any,
    "sha256sum": _any,
    "realpath": _any,
    "readlink": _any,
    "basename": _any,
    "dirname": _any,
    "getfacl": _any,
    "numfmt": _any,
    "seq": _any,
    # Environment and identity.
    "pwd": _any,
    "cd": _any,
    "whoami": _any,
    "id": _any,
    "groups": _any,
    "hostname": _max_positionals(0),
    "date": _forbid("-s", "--set"),
    "uptime": _any,
    "echo": _any,
    "printf": _any,
    "printenv": _any,
    "which": _any,
    "true": _any,
    "test": _any,
    "[": _any,
}

_SAFE_REDIRECT = re.compile(r"\d?>>?\s*(/dev/null\b|&[12]\b)|&>\s*/dev/null\b")


def _syntax_problem(command: str) -> str | None:
    """Shell syntax that could write files or run hidden commands, outside quotes."""
    quote = ""
    index = 0
    while index < len(command):
        char = command[index]
        if quote == "'":
            if char == "'":
                quote = ""
        elif char == "\\":
            index += 1
        elif quote == '"':
            if char == '"':
                quote = ""
            elif char == "`" or command.startswith("$(", index):
                return "command substitution"
        elif char in "'\"":
            quote = char
        elif char == "`" or command.startswith("$(", index):
            return "command substitution"
        elif command.startswith("<(", index) or command.startswith(">(", index):
            return "process substitution"
        elif char in "()":
            return "a subshell"
        elif char == ">" or (char == "&" and command.startswith("&>", index)):
            start = index - 1 if index and command[index - 1].isdigit() else index
            match = _SAFE_REDIRECT.match(command, start)
            if not match:
                return "output redirection to a file"
            index = match.end()
            continue
        elif char == "&":
            if command.startswith("&&", index):
                index += 2
                continue
            if index and command[index - 1] in "><":
                pass
            else:
                return "a background job (&)"
        index += 1
    if quote:
        return "unbalanced quotes"
    return None


def auto_approvable(command: str, extra: Iterable[str] = ()) -> tuple[bool, str]:
    """Whether a command may run without asking: only read-only allowlisted commands.

    Returns (True, "") or (False, why it needs approval). extra adds command
    names the user trusts as read-only, with any arguments.
    """
    problem = _syntax_problem(command)
    if problem:
        return False, f"it uses {problem}"
    trusted = set(extra)
    # The syntax check allowed only harmless redirects (to /dev/null, 2>&1); drop them so
    # the '&' in 2>&1 does not look like a separator. depth=3 skips looking inside bash -c,
    # which is fine because bash itself is not on the list.
    commands = _simple_commands(_SAFE_REDIRECT.sub(" ", command), depth=3)
    if not commands:
        return False, "it is empty"
    for words in commands:
        first = words[0]
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", first):
            return False, "it sets environment variables"
        if "/" in first:
            return False, f"'{first}' is a path, not a known command"
        if first in trusted:
            continue
        check = READ_ONLY_COMMANDS.get(first)
        if check is None:
            return False, f"'{first}' is not on the read-only list"
        problem = check(words[1:])
        if problem:
            return False, f"{first}: {problem}"
    return True, ""


__all__ = ["READ_ONLY_COMMANDS", "auto_approvable", "check_command"]
