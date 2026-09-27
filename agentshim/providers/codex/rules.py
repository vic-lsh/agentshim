"""Codex exec-policy rules that let named commands run outside the sandbox.

Codex decides per command whether to sandbox it by evaluating exec-policy
rules, Starlark files named ``*.rules``. A command whose words start with a
``prefix_rule(..., decision="allow")`` pattern skips the sandbox; every other
command stays confined. That is the Codex counterpart of Claude Code's
``excludedCommands``, and the only per-command exemption Codex has: network
and filesystem settings apply to the whole session.

Codex reads rules only from files: ``$CODEX_HOME/rules/`` and the
``.codex/rules/`` of trusted projects. There is no ``--config`` key for them,
so ``CodexSandboxConfig.excluded_commands`` cannot travel in argv like the
rest of the sandbox. ``install_rules`` writes them into a Codex home the
caller dedicates to the sandboxed agent, and the turn points ``CODEX_HOME``
at it. A sandboxed turn without exemptions passes ``--ignore-rules`` instead,
so no rules file, the user's included, can widen it.
"""

from __future__ import annotations

import os
import re
import shlex
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sandbox import CodexSandboxConfig

#: The file ``install_rules`` owns inside ``$CODEX_HOME/rules/``.
RULES_FILENAME = "agentshim.rules"

_HEADER = (
    "# Written by agentshim from CodexSandboxConfig.excluded_commands.\n"
    "# Each rule lets commands starting with these words run outside Codex's sandbox.\n"
)
_RULE_RE = re.compile(r'^prefix_rule\(pattern=\[(.*)\], decision="allow"\)$')
_STRING_RE = re.compile(r'"((?:[^"\\]|\\["\\])*)"')


def command_words(command: str) -> list[str]:
    """Split one ``excluded_commands`` entry into the words Codex matches."""
    return shlex.split(command)


def render_rules(config: CodexSandboxConfig) -> str:
    """Render *config*'s ``excluded_commands`` as a Codex ``.rules`` file."""
    lines = [_HEADER]
    for command in config.excluded_commands:
        words = ", ".join(_starlark_str(word) for word in command_words(command))
        lines.append(f'prefix_rule(pattern=[{words}], decision="allow")\n')
    return "".join(lines)


def parse_rules(text: str) -> list[list[str]]:
    """Recover the word prefixes ``render_rules`` wrote, in order.

    Raises:
        ValueError: *text* holds a line ``render_rules`` never writes.
    """
    prefixes: list[list[str]] = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = _RULE_RE.match(line)
        if match is None:
            msg = f"not a rule agentshim writes: {line!r}"
            raise ValueError(msg)
        prefixes.append([_unescape(body) for body in _STRING_RE.findall(match.group(1))])
    return prefixes


def install_rules(codex_home: str | os.PathLike[str], config: CodexSandboxConfig) -> Path:
    """Write *config*'s exemptions to ``<codex_home>/rules/agentshim.rules``.

    Run the turn with ``CODEX_HOME`` set to *codex_home*. The home should be
    dedicated to the sandboxed agent: every Codex process that uses it, and
    does not pass ``--ignore-rules``, applies these rules. It must also lie
    outside every directory the sandbox lets commands write, or a command
    could add a rule of its own; ``CodexProvider`` refuses such a home.

    The write is atomic, so a turn starting concurrently reads either the old
    rules or the new ones.

    Returns:
        The path of the rules file.

    Raises:
        ValueError: ``rules/`` already holds another ``*.rules`` file. Codex
            would load it too, so it could exempt commands the config does not.
    """
    rules_dir = Path(codex_home) / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    others = sorted(p.name for p in rules_dir.glob("*.rules") if p.name != RULES_FILENAME)
    if others:
        msg = (
            f"{rules_dir} also holds {', '.join(others)}; Codex would load them "
            "alongside the exemptions, so use a dedicated CODEX_HOME"
        )
        raise ValueError(msg)
    target = rules_dir / RULES_FILENAME
    fd, tmp = tempfile.mkstemp(dir=rules_dir, prefix=".agentshim-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(render_rules(config))
        Path(tmp).replace(target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return target


def _starlark_str(word: str) -> str:
    """Quote *word* as a Starlark string literal.

    ``CodexSandboxConfig`` rejects control characters, so only the quote and
    the backslash need escaping; everything else is written literally, which
    Starlark accepts for any code point outside the surrogate range.
    """
    return '"' + word.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _unescape(body: str) -> str:
    return re.sub(r'\\(["\\])', r"\1", body)
