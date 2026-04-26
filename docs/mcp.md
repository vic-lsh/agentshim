# MCP Servers

Claude Code and Codex can be configured with MCP servers by passing
`HttpMcpServer` and `StdioMcpServer` objects at construction time.

```python
from agentshim import ClaudeCodeCodingAgent, HttpMcpServer, StdioMcpServer

agent = ClaudeCodeCodingAgent(
    model="sonnet",
    mcp_servers=[
        HttpMcpServer(
            name="docs",
            url="http://localhost:9000/sse",
            headers={"Authorization": "Bearer dev-token"},
        ),
        StdioMcpServer(
            name="github",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_TOKEN": "ghp_example"},
        ),
    ],
)

chat = agent.start_session(cwd=".")
print(chat.generate("Use the MCP tools to inspect the repo."))
```

Notes:

- `HttpMcpServer` is for HTTP/SSE-backed MCP servers.
- `StdioMcpServer` is for subprocess-backed MCP servers.
- Gemini and Opencode currently reject `mcp_servers`; use Claude Code or Codex
  if you need MCP.
