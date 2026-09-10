"""Declared provider capabilities.

Every optional behaviour is declared here (or on a protocol) so no caller
has to probe a provider object for attributes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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


@dataclass(frozen=True)
class ProviderProfile:
    """Everything a caller needs to know about a provider without running it."""

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
