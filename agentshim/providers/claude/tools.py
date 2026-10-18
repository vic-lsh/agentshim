"""Which tools a Claude Code turn has, and which it may run without asking.

By default agentshim runs Claude Code with ``--dangerously-skip-permissions``:
every built-in tool is present and every call is allowed. That suits an agent
bounded by an outside ``Confinement``. A caller that wants the CLI itself to
withhold tools (a model that should only answer, or only call one MCP server)
passes a ``ClaudeToolPolicy`` instead, and the turn runs with
``--permission-mode dontAsk``: a call that no allow rule covers is denied
without a prompt, and the model is told it was refused.

See https://code.claude.com/docs/en/settings#permission-settings and
``claude --help`` (``--tools``, ``--allowedTools``, ``--permission-mode``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from .parser import STRUCTURED_OUTPUT_TOOL

#: The permission mode of a turn under a tool policy: anything not allowed is
#: denied without prompting.
DONT_ASK_MODE = "dontAsk"


@dataclass(frozen=True)
class ClaudeToolPolicy:
    """The tools a Claude Code turn is given and may call.

    Attributes:
        builtin: The built-in tools present in the session (``--tools``), such
            as ``("Read", "Grep")``. ``()`` removes every built-in tool;
            ``None`` keeps the CLI's default set. A tool that is absent cannot
            be called, whatever ``allowed`` says.
        allowed: Permission rules (``--allowedTools``) for calls that run
            without asking: a tool name (``"Read"``), an MCP server
            (``"mcp__github"``, all of its tools), one MCP tool
            (``"mcp__github__get_issue"``) or a scoped rule
            (``"Bash(git log:*)"``). Every other call is denied.

    When the turn asks for an output schema, the provider also allows Claude
    Code's ``StructuredOutput`` tool, which is how the answer is returned.
    """

    builtin: Sequence[str] | None = None
    allowed: Sequence[str] = ()

    def __post_init__(self) -> None:
        """Freeze both lists and reject entries the CLI would misread."""
        if self.builtin is not None:
            builtin = _as_names(self.builtin, "builtin")
            for name in builtin:
                # ``--tools`` takes one comma-separated list of plain names.
                if "," in name or any(ch.isspace() for ch in name) or "(" in name:
                    msg = f"builtin entries are plain tool names, got {name!r}"
                    raise ValueError(msg)
            object.__setattr__(self, "builtin", builtin)
        object.__setattr__(self, "allowed", _as_names(self.allowed, "allowed"))

    @classmethod
    def only(cls, allowed: Sequence[str]) -> ClaudeToolPolicy:
        """No built-in tools; only the calls *allowed* names (MCP servers or tools) run."""
        return cls(builtin=(), allowed=allowed)

    def argv(self, *, structured_output: bool) -> list[str]:
        """The flags that replace ``--dangerously-skip-permissions`` for one turn.

        Each allow rule is its own argument, so a rule containing a space
        (``"Bash(git log:*)"``) reaches the CLI whole.
        """
        argv: list[str] = []
        if self.builtin is not None:
            argv += ["--tools", ",".join(self.builtin)]
        allowed = list(self.allowed)
        if structured_output and STRUCTURED_OUTPUT_TOOL not in allowed:
            allowed.append(STRUCTURED_OUTPUT_TOOL)
        if allowed:
            argv += ["--allowedTools", *allowed]
        argv += ["--permission-mode", DONT_ASK_MODE]
        return argv


def resolve_tools(value: ClaudeToolPolicy | None) -> ClaudeToolPolicy | None:
    """Accept ``None`` (no policy) or a policy; reject anything else."""
    if value is not None and not isinstance(value, ClaudeToolPolicy):  # pyright: ignore[reportUnnecessaryIsInstance]
        msg = f"tools must be a ClaudeToolPolicy or None, got {type(value).__name__}"
        raise TypeError(msg)
    return value


def _as_names(value: object, field: str) -> tuple[str, ...]:
    """Validate a list of tool names or rules and freeze it into a tuple.

    A bare string is rejected rather than iterated, which would turn
    ``"Read"`` into the tools ``"R"``, ``"e"``, ...
    """
    if isinstance(value, str) or not isinstance(value, Sequence):
        msg = f"{field} must be a sequence of strings, not {type(value).__name__}"
        raise TypeError(msg)
    names = tuple(cast("Sequence[object]", value))
    for name in names:
        if not isinstance(name, str):
            msg = f"{field} entries must be str, got {type(name).__name__}"
            raise TypeError(msg)
        if not name.strip() or name != name.strip():
            msg = f"{field} entries must be non-empty with no surrounding space, got {name!r}"
            raise ValueError(msg)
        if "\x00" in name or name.startswith("-"):
            msg = f"{field} entry would be misread by the CLI: {name!r}"
            raise ValueError(msg)
    return cast("tuple[str, ...]", names)
