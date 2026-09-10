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

`HttpMcpServer(name=..., url=..., headers=..., transport=...)` describes a
remote server. `transport` is `"http"` (streamable HTTP, the default) or
`"sse"`; they are different wire protocols, so a CLI told the wrong one
connects and then fails. What each provider does with it:

| provider | `transport="http"` | `transport="sse"` |
| --- | --- | --- |
| claude | `{"type": "http", "url": ...}` | `{"type": "sse", "url": ...}` |
| copilot | `{"type": "http", "url": ...}` | `{"type": "sse", "url": ...}` |
| gemini | `{"httpUrl": ...}` | `{"url": ...}` |
| opencode | `{"type": "remote", "url": ...}` | same |
| codex | `mcp_servers.<name>.url=...` | same |

opencode has one remote server type and codex's `--config` override carries
only the address, so neither can be told which transport to use; both work it
out from the endpoint. Codex additionally cannot carry HTTP headers, and
raises `ProviderCapabilityError` rather than dropping them.

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
merge semantics outside a turn. `restore()` never raises: if the file is no
longer readable as JSON it writes the original bytes back verbatim and returns
a note saying so.

Concurrent installs into one workspace from separate processes are not
supported. The merge is read-modify-write on a plain JSON file with no lock,
so two agents installing into the same `.mcp.json` at the same time will race,
and one restore can undo the other's entries. Give each concurrent turn its
own workspace, or its own `TurnRequest.mcp_workspace`.
