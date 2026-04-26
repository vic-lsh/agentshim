# agentshim

`agentshim` wraps coding-agent CLIs behind one small Python interface.

It is useful when you want to drive tools like Claude Code, Codex, Gemini, or
Opencode from Python without writing provider-specific subprocess plumbing for
prompting, session resumption, event parsing, or MCP configuration.

## What It Includes

- a shared CLI agent abstraction with a provider registry
- adapters for Claude Code, Codex, Gemini, and Opencode
- stateful chat sessions that automatically resume provider-native threads
- injectable command executors for custom process launching or sandboxing
- MCP server config models for providers that support MCP
- sandbox settings helpers for Claude Code
- a lightweight LiteLLM client and subagent helper

## Install

```bash
uv add agentshim
```

`agentshim` does not bundle the underlying agent CLIs. You still need the
provider tool you want to use installed and authenticated on your machine, for
example `claude`, `codex`, `gemini`, or `opencode`.

## First Request

```python
from agentshim import CodexCodingAgent

agent = CodexCodingAgent(model="gpt-5")
reply = agent.generate("Write a short summary of this codebase.", cwd=".")
print(reply)
```
