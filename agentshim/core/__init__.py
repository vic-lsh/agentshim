"""Provider-agnostic core: the types and protocols a provider is written against.

The agent and its session live one layer out, in ``agentshim/agent.py``:
turning a provider name into a provider means importing ``providers``, which
this package may not do.
"""

from __future__ import annotations

from .env import interactive_env
from .errors import (
    AgentShimError,
    CliCheckError,
    CliExitError,
    CliNotFoundError,
    CliTimeoutError,
    McpConfigError,
    ProviderCapabilityError,
    SchemaDialectError,
    SessionResumeError,
)
from .events import (
    AgentEvent,
    AgentEventHandler,
    AssistantText,
    CompositeEventHandler,
    ConsoleEventHandler,
    EventHandlerBase,
    Lifecycle,
    NullEventHandler,
    ProviderError,
    RawOutput,
    Reasoning,
    RunFinished,
    RunStarted,
    SessionStarted,
    Stderr,
    ToolCall,
    ToolResult,
    UsageReport,
    compose_event_handlers,
)
from .mcp import (
    ConfigFileInstallation,
    FlagsInstallation,
    HttpMcpServer,
    McpServer,
    McpTransport,
    NoopInstallation,
    StdioMcpServer,
    install_config_file,
)
from .profile import McpMechanism, OutputSchemaStyle, ProviderProfile, SchemaDialect
from .provider import ArgvContext, McpInstallation, ParsedTurn, Provider, StreamParser
from .schema import compact_json, dialect_problems, materialize, normalize
from .stream import ToolTracker, parse_json_object
from .turn import OutputSchema, TurnRequest, TurnResult
from .usage import ProviderUsage, TokenUsage

__all__ = [
    "AgentEvent",
    "AgentEventHandler",
    "AgentShimError",
    "ArgvContext",
    "AssistantText",
    "CliCheckError",
    "CliExitError",
    "CliNotFoundError",
    "CliTimeoutError",
    "CompositeEventHandler",
    "ConfigFileInstallation",
    "ConsoleEventHandler",
    "EventHandlerBase",
    "FlagsInstallation",
    "HttpMcpServer",
    "Lifecycle",
    "McpConfigError",
    "McpInstallation",
    "McpMechanism",
    "McpServer",
    "McpTransport",
    "NoopInstallation",
    "NullEventHandler",
    "OutputSchema",
    "OutputSchemaStyle",
    "ParsedTurn",
    "Provider",
    "ProviderCapabilityError",
    "ProviderError",
    "ProviderProfile",
    "ProviderUsage",
    "RawOutput",
    "Reasoning",
    "RunFinished",
    "RunStarted",
    "SchemaDialect",
    "SchemaDialectError",
    "SessionResumeError",
    "SessionStarted",
    "Stderr",
    "StdioMcpServer",
    "StreamParser",
    "TokenUsage",
    "ToolCall",
    "ToolResult",
    "ToolTracker",
    "TurnRequest",
    "TurnResult",
    "UsageReport",
    "compact_json",
    "compose_event_handlers",
    "dialect_problems",
    "install_config_file",
    "interactive_env",
    "materialize",
    "normalize",
    "parse_json_object",
]
