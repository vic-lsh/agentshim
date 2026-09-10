# MCP Servers

Ask for MCP servers per turn. agentshim installs them the way the provider
expects, and restores whatever it changed when the turn ends, including when
the turn fails.

```python
from agentshim import CliAgent, StdioMcpServer, TurnRequest

servers = [
    StdioMcpServer(
        name="issues",
        command="python",
        args=["-m", "board.mcp", "issues.json"],
        env={"BOARD_TOKEN": "..."},
    )
]

agent = CliAgent("claude")
agent.start_session(cwd="/workspace").turn(
    TurnRequest(prompt="File a bug for the failing test.", mcp_servers=servers)
)
```

`HttpMcpServer(name=..., url=..., headers=...)` describes an HTTP/SSE server.

## How installation works

`ProviderProfile.mcp` declares the mechanism:

- `CONFIG_FILE`: the servers are merged into the provider's JSON config in the
  workspace (`.mcp.json` for Claude Code). The original bytes are kept and
  written back on restore. If the file changed during the turn, only the
  entries agentshim added are removed: an edit the agent made to one of them,
  a server it added, and unrelated top-level keys all survive. Deleting the
  file during the turn is treated as a workspace edit and is not undone.
- `CLI_FLAGS`: the servers become flags appended to argv; nothing outlives the
  process.
- `NONE`: asking for servers raises `ProviderCapabilityError`.

A `CONFIG_FILE` provider needs a `cwd`, since that is where the config lives.
A config that is a symlink is followed to the file it names, so the link
survives the turn.

## Doing it yourself

`install_config_file(target, server_key=..., servers=..., defaults=...)`
returns an installation whose `restore()` is idempotent, if you need the same
merge semantics outside a turn.
