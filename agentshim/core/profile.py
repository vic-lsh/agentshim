"""Declared provider capabilities.

Every optional behaviour is declared here (or on a protocol) so no caller
has to probe a provider object for attributes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING

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

    ``STRICT`` is Codex's ``--output-schema`` subset: every object declares
    its properties and forbids undeclared keys. ``OPEN`` also accepts a
    schema-valued or ``true`` ``additionalProperties``.
    """

    STRICT = "strict"
    OPEN = "open"


def _no_container_env() -> Mapping[str, str]:
    """Empty read-only default: a frozen spec must not carry a mutable one.

    A factory rather than a shared constant because Python 3.10 and 3.11
    reject an unhashable dataclass default, and ``mappingproxy`` is one. See
    ``agentshim.core.mcp._no_env`` for the same pattern.
    """
    return MappingProxyType({})


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
