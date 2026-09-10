# agentshim

`agentshim` runs coding-agent CLIs (Claude Code, Codex, Gemini CLI, opencode,
Copilot CLI) as subprocesses and turns their output into typed events and a
typed turn result.

It owns everything that answers "how do I run provider X and understand what
it printed". It does not own application policy: which provider to use, when
to retire a conversation, how to sandbox the host, or how to render events.

## What it includes

- `CliAgent` / `AgentSession`: one turn at a time, resumable, cancellable
- typed events (`AssistantText`, `ToolCall`, `ToolResult`, `UsageReport`, ...)
  delivered on the calling thread
- normalized token accounting where `cached_input_tokens <= input_tokens` on
  every provider
- declared capabilities on `ProviderProfile`, so no caller probes a provider
  for attributes
- injectable `CommandExecutor`s for running the CLI in a container or over a
  remote shell
- MCP server installation and restoration per turn
- native structured output with per-provider schema dialect checks
- `agentshim.testing`: test doubles that emit each provider's real stream
  format

No required runtime dependencies. Python 3.10+.

## Install

```bash
uv add agentshim
```

agentshim does not bundle the agent CLIs. Install and authenticate the
provider tool you want (`claude`, `codex`, `gemini`, `opencode`, or
`copilot`) yourself.

## First turn

```python
from agentshim import CliAgent

agent = CliAgent("claude", model="sonnet")
result = agent.run("Write a short summary of this codebase.", cwd=".")
print(result.text)
print(result.usage.tokens.input_tokens, result.cost_usd)
```

`model` is an opaque provider-specific string; `None` leaves the CLI's own
default. See [Getting Started](getting-started.md) for sessions, per-turn
options, and errors.

## Status

0.6 ships the core, the execution layer, and all five providers on one
protocol: Claude Code, Codex, Gemini CLI, opencode, and Copilot CLI. See
[Architecture](architecture.md) for the layering and the type contracts, and
[Providers](providers.md) for what each CLI supports.

0.6 is not source-compatible with 0.5 and there is no compatibility layer.
The [changelog](https://github.com/vic-lsh/agentshim/blob/main/CHANGELOG.md)
lists every removed and renamed name.
