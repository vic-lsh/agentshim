from pydantic import BaseModel, ConfigDict, Field


class HttpMcpServer(BaseModel):
    """MCP server accessed over HTTP/SSE."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    url: str = Field(min_length=1)
    headers: dict[str, str] = Field(default_factory=dict)


class StdioMcpServer(BaseModel):
    """MCP server launched as a subprocess (stdio transport)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    command: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


McpServerConfig = HttpMcpServer | StdioMcpServer
