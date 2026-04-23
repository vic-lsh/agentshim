# agentshim

`agentshim` wraps coding-agent CLIs behind one small Python interface.

It is useful when you want to drive tools like Claude Code, Codex, Gemini, or
Opencode from Python without writing provider-specific subprocess plumbing for
prompting, session resumption, event parsing, or MCP configuration.

## What It Includes

- a shared CLI agent abstraction with a provider registry
- adapters for Claude Code, Codex, Gemini, and Opencode
- stateful chat sessions that automatically resume provider-native threads
- MCP server config models for providers that support MCP
- sandbox settings helpers for Claude Code
- a lightweight LiteLLM client and subagent helper
- trajectory/usage helpers used by higher-level runtimes

## Install

```bash
uv add agentshim
```

`agentshim` does not bundle the underlying agent CLIs. You still need the
provider tool you want to use installed and authenticated on your machine, for
example `claude`, `codex`, `gemini`, or `opencode`.

## Getting Started

### 1. Use the Generic Agent Interface for Chat and Resume

If you want to choose a provider at runtime, instantiate `CodingAgent`
directly with a provider name.

```python
from agentshim import CodingAgent

agent = CodingAgent(provider="claude", model="sonnet")
chat = agent.start_session(cwd=".")

first_reply = chat.generate("Summarize this repository.")
follow_up = chat.generate("Now list the three highest-risk modules.")

print(first_reply)
print(follow_up)
print(chat.session_id)
```

`start_session()` returns a stateful chat object. On the first `generate(...)`
call, `agentshim` starts a fresh provider conversation. On later calls, it
automatically resumes the same underlying provider session using the session id
captured from the first run.

That corresponds roughly to these native CLI flows:

- Claude Code: first call is like `claude -p ...`, later calls add `claude --resume <session_id> ...`
- Codex: first call is like `codex exec ...`, later calls add `codex exec resume <thread_id> ...`
- Gemini: first call is like `gemini ...`, later calls add `gemini --resume <session_id> ...`
- Opencode: first call is like `opencode run ...`, later calls add `opencode run --session <session_id> ...`

If you only want a one-shot request, use `generate(...)` directly instead of
opening a session:

```python
from agentshim import CodexCodingAgent

agent = CodexCodingAgent(model="gpt-5")
reply = agent.generate("Write a short summary of this codebase.", cwd=".")
print(reply)
```

### 2. Instantiate a Specific Provider Directly

If you already know which backend you want, construct the provider class
yourself.

```python
from agentshim import ClaudeCodeCodingAgent

agent = ClaudeCodeCodingAgent(model="sonnet")
chat = agent.start_session(cwd=".")

print(chat.generate("What does this project do?"))
print(chat.generate("Which files should I read first?"))
```

The bundled provider classes are:

- `ClaudeCodeCodingAgent`
- `CodexCodingAgent`
- `GeminiCodingAgent`
- `OpencodeCodingAgent`

### 3. Configure MCP Servers

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
- Gemini and Opencode currently reject `mcp_servers`; use Claude Code or Codex if you need MCP.

## Extending agentshim

Advanced users can register their own providers.

```python
from agentshim import BaseCodingAgent, CodingAgent, register_provider


@register_provider("my-agent", aliases=("my-agent-dev",))
class MyAgent(BaseCodingAgent):
    def __init__(
        self,
        model: str | None = None,
        recorder=None,
        event_handler=None,
        mcp_servers=None,
        sandbox=False,
    ):
        self.model = model
        self.recorder = recorder
        self.event_handler = event_handler

    def generate(self, prompt: str, cwd=None, timeout=300, silent=False) -> str:
        return f"handled: {prompt}"


agent = CodingAgent(provider="my-agent-dev", model="demo")
print(agent.generate("hello"))
```

Notes:

- Registration is import-driven. Your provider is available only after the module defining it has been imported in the current Python process.
- `list_providers()` returns canonical provider names only. Aliases resolve via `get_provider_class(...)` and `CodingAgent(provider=...)`.
- `register_provider(...)` rejects invalid names, abstract classes, and accidental name collisions unless you pass `overwrite=True`.
- If you want `CodingAgent(...)` to instantiate your provider, its constructor should accept the shared kwargs `model`, `recorder`, `event_handler`, `mcp_servers`, and `sandbox` as needed.

## Development

```bash
uv sync --dev
uv run pytest
```

## Publishing

Build locally with:

```bash
uv build
```

Publish with:

```bash
uv publish
```
