from dataclasses import dataclass, field


@dataclass(frozen=True)
class HttpMcpServer:
    """MCP server accessed over HTTP/SSE."""

    name: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)  # pyright: ignore[reportUnknownVariableType]

    def __post_init__(self):
        if not self.name:
            raise ValueError("MCP server name must be non-empty")
        if not self.url:
            raise ValueError(f"MCP server '{self.name}' must have a non-empty url")


@dataclass(frozen=True)
class StdioMcpServer:
    """MCP server launched as a subprocess (stdio transport)."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    env: dict[str, str] = field(default_factory=dict)  # pyright: ignore[reportUnknownVariableType]

    def __post_init__(self):
        if not self.name:
            raise ValueError("MCP server name must be non-empty")
        if not self.command:
            raise ValueError(f"MCP server '{self.name}' must have a non-empty command")


McpServerConfig = HttpMcpServer | StdioMcpServer
