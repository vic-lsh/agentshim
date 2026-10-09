"""Declared provider capabilities.

Every optional behaviour is declared here (or on a protocol) so no caller
has to probe a provider object for attributes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING

from .permissions import NativeMode

if TYPE_CHECKING:
    from collections.abc import Mapping


class McpMechanism(Enum):
    """How a provider is told about MCP servers."""

    NONE = "none"
    CONFIG_FILE = "config_file"
    CLI_FLAGS = "cli_flags"


class OutputSchemaStyle(Enum):
    """How a provider accepts a JSON Schema for structured output."""

    NONE = "none"
    INLINE_JSON = "inline_json"
    FILE_PATH = "file_path"


class SchemaDialect(Enum):
    """Which JSON Schema subset a provider accepts.

    ``STRICT`` is Codex's ``--output-schema`` subset, which is OpenAI's strict
    structured outputs: every object declares ``properties``, sets
    ``additionalProperties: false`` and lists every property in ``required`` (optional values are made
    nullable instead), the root is not an ``anyOf``, and every ``$ref``
    resolves. ``OPEN`` also accepts optional properties and a schema-valued,
    ``true`` or absent ``additionalProperties``.
    """

    STRICT = "strict"
    OPEN = "open"


class SkillSignal(Enum):
    """How much a provider's output stream says about skills.

    ``NONE``: nothing; a caller must treat skill use as unknown, never as
    zero. ``STRUCTURED``: the CLI emits a dedicated, documented frame.
    ``INFERRED``: agentshim derives it from tool activity (for example a
    shell read of a skill's ``SKILL.md``), so a load by other means can be
    missed.
    """

    NONE = "none"
    STRUCTURED = "structured"
    INFERRED = "inferred"


class SkillScope(Enum):
    """Which skills a session's CLI may discover.

    ``ALL``: whatever the CLI finds by default, including the user's own
    skills and installed plugins. ``PROJECT``: only the skills under the
    session's working directory (the profile's ``skill_dirs``) plus the
    CLI's built-in ones, so the user running agentshim does not change what
    the agent is offered. A provider declares the scopes it can enforce in
    ``ProviderProfile.skill_scopes``; asking for another one is an error,
    never a silent ``ALL``.
    """

    ALL = "all"
    PROJECT = "project"


class McpScope(Enum):
    """Which MCP servers a session's CLI may connect to.

    ``ALL``: whatever the CLI finds by default, including the user's and the
    project's own MCP configuration, plugin servers and account-level
    connectors. ``SESSION``: only the servers the session was given through
    ``TurnRequest.mcp_servers``; a session given none sees no MCP server at
    all. A provider declares the scopes it can enforce in
    ``ProviderProfile.mcp_scopes``; asking for another one is an error, never
    a silent ``ALL``.
    """

    ALL = "all"
    SESSION = "session"


class ConfigScope(Enum):
    """Which of the user's own CLI configuration a session loads.

    ``ALL``: whatever the CLI loads by default: the user's settings, hooks,
    global instructions (``~/.claude/CLAUDE.md``, ``$CODEX_HOME/AGENTS.md``),
    notify commands, profiles and memory. ``PROJECT``: none of that; the
    session sees the workspace's own instructions and settings plus what
    agentshim passes for the turn, so the user running agentshim does not
    change what the agent does. Credentials keep working. A provider that
    lists a ``ProviderProfile.config_home_files`` can only isolate through a
    dedicated home: ``prepare_config_home`` builds it and returns the
    environment that points the CLI at it. A provider declares the scopes it
    can enforce in ``ProviderProfile.config_scopes``; asking for another one
    is an error, never a silent ``ALL``.
    """

    ALL = "all"
    PROJECT = "project"


def _no_container_env() -> Mapping[str, str]:
    """Empty read-only default: a frozen spec must not carry a mutable one.

    A factory rather than a shared constant because Python 3.10 and 3.11
    reject an unhashable dataclass default, and ``mappingproxy`` is one. See
    ``agentshim.core.mcp._no_env`` for the same pattern.
    """
    return MappingProxyType({})


@dataclass(frozen=True)
class RenewalBudget:
    """When a long conversation should be retired and restarted cold.

    A conversation that grows without bound gets slower and more expensive, so
    a provider can declare limits. They are checked after a successful turn:
    reaching ``max_turns`` successful turns in the conversation, or any one turn
    reaching ``max_turn_input_tokens`` input tokens or ``max_turn_duration_ms``
    milliseconds (a "heavy turn"), retires it. ``None`` leaves a limit off.
    """

    max_turns: int | None = None
    max_turn_input_tokens: int | None = None
    max_turn_duration_ms: int | None = None


@dataclass(frozen=True)
class ProviderProfile:
    """Everything a caller needs to know about a provider without running it.

    The last four fields are additive (0.6.1+) and all default, so an
    existing keyword-built ``ProviderProfile`` keeps working unchanged.
    ``container_env`` is environment a CLI needs to run as root in a
    container, beyond auth (e.g. Claude Code's ``IS_SANDBOX=1``).
    ``state_root_env`` is the documented variable that relocates
    ``state_dirs[0]``, or ``None`` when the CLI names none. ``auth_files`` are
    the home-relative paths, each inside a ``state_dirs`` entry, whose
    contents are the minimal set to copy so the CLI is logged in elsewhere.
    ``mcp_config_file`` is the workspace-relative path a ``CONFIG_FILE``
    provider writes its MCP config to for a turn, and ``None`` for
    ``CLI_FLAGS``/``NONE`` providers.
    """

    name: str
    display_name: str
    binary: str
    supports_resume: bool
    supports_reasoning_effort: bool
    mcp: McpMechanism
    output_schema: OutputSchemaStyle
    schema_dialect: SchemaDialect | None
    state_dirs: tuple[str, ...]
    darwin_state_dirs: tuple[str, ...]
    auth_env_vars: tuple[str, ...]
    skill_dirs: tuple[str, ...]
    container_install: tuple[str, ...]
    container_env: Mapping[str, str] = field(default_factory=_no_container_env)
    state_root_env: str | None = None
    auth_files: tuple[str, ...] = ()
    mcp_config_file: str | None = None
    #: Whether the stream lists the skills offered (``SkillsDiscovered``).
    skill_discovery: SkillSignal = SkillSignal.NONE
    #: Whether the stream reveals skill loads (``SkillInvoked``).
    skill_invocation: SkillSignal = SkillSignal.NONE
    #: The ``SkillScope`` values a session on this provider may request.
    skill_scopes: frozenset[SkillScope] = frozenset({SkillScope.ALL})
    #: The ``McpScope`` values a session on this provider may request.
    mcp_scopes: frozenset[McpScope] = frozenset({McpScope.ALL})
    #: The ``ConfigScope`` values a session on this provider may request.
    config_scopes: frozenset[ConfigScope] = frozenset({ConfigScope.ALL})
    #: Files, relative to the state root (``state_dirs[0]``, or
    #: ``$<state_root_env>``), that ``prepare_config_home`` copies into a
    #: dedicated home for ``ConfigScope.PROJECT``. Empty when the provider
    #: isolates by flags alone and needs no home.
    config_home_files: tuple[str, ...] = ()
    #: The ``NativeMode`` values the provider can enforce through its own
    #: sandbox. Every provider supports ``BYPASS`` (that is how it runs today);
    #: a transport that maps the other modes declares them here, and asking for
    #: one that is missing is a ``ProviderCapabilityError``.
    native_permission_modes: frozenset[NativeMode] = frozenset({NativeMode.BYPASS})
    #: When a session should retire a conversation of this provider; ``None``
    #: means never.
    renewal: RenewalBudget | None = None
    #: Whether a conversation on this provider takes ``steer(text)`` while a
    #: turn runs. A provider that does not (every one-shot transport) rejects
    #: it with ``ProviderCapabilityError`` instead of delaying the message.
    supports_steer: bool = False
