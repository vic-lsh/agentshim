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

### 2. Handle Agent Events

By default, `agentshim` prints provider events to the terminal through a
`ConsoleEventHandler`. That default is used only when you do not provide your
own event handler and `silent=False`.

If you pass `event_handler=...`, you take ownership of event handling. The
built-in console printer is not added implicitly, which avoids surprising
duplicate output.

```python
from agentshim import CodingAgent


class MyHandler:
    def on_thinking(self, text: str) -> None:
        ...

    def on_tool_call(self, tool: str, args=None) -> None:
        ...

    def on_tool_result(
        self,
        tool: str,
        stdout: str = "",
        stderr: str = "",
        exit_code: int | None = None,
        duration: float | None = None,
    ) -> None:
        ...

    def on_usage(self, usage: dict) -> None:
        ...


agent = CodingAgent(provider="claude", event_handler=MyHandler())
agent.generate("Inspect this repository.")
```

To keep the default console output and add your own handler, compose them
explicitly:

```python
from agentshim import CodingAgent, ConsoleEventHandler

agent = CodingAgent(
    provider="claude",
    event_handlers=[
        ConsoleEventHandler(),
        MyHandler(),
    ],
)
agent.generate("Inspect this repository.")
```

You can also build the composition yourself:

```python
from agentshim import CompositeEventHandler, ConsoleEventHandler

handler = CompositeEventHandler([ConsoleEventHandler(), MyHandler()])
agent = CodingAgent(provider="codex", event_handler=handler)
```

Use `silent=True` to suppress the default console handler when you have not
provided any handler:

```python
agent = CodingAgent(provider="claude")
reply = agent.generate("Return only the answer.", silent=True)
```

### 3. Run Commands Through a Custom Executor

Provider classes accept an optional `executor=`. The executor controls binary
lookup, CLI validation, and streaming process execution while `agentshim` keeps
owning provider command construction, stdout/stderr parsing, session state, and
event emission.

This is useful when a caller needs to run the CLI somewhere other than the
current host process, for example through an existing container, remote shell,
or custom sandbox.

```python
from agentshim import (
    CodexCodingAgent,
    CommandHandle,
    CommandRequest,
    CommandResult,
    CommandStreamSink,
)


class MyCommandHandle:
    def terminate(self) -> None:
        ...

    def kill(self) -> None:
        ...


class MyExecutor:
    def find_binary(self, binary_name: str, env: dict[str, str]) -> str:
        # This value becomes request.argv[0]. Container or remote executors can
        # return the binary name if lookup happens in the target runtime.
        return binary_name

    def check_binary(self, binary_path: str, env: dict[str, str], *, timeout: int) -> None:
        # Raise RuntimeError if the target CLI is unavailable. No-op is fine
        # when validation is not cheap or is handled by the runtime.
        return None

    def run(self, request: CommandRequest, sink: CommandStreamSink) -> CommandResult:
        handle = MyCommandHandle()
        sink.started(handle)

        stdout = ""
        stderr = ""

        # Run request.argv in your target runtime, with request.stdin,
        # request.cwd, request.env, and request.timeout. Stream each complete
        # line as it arrives, preserving trailing newlines when present.
        line = "streamed output\n"
        stdout += line
        sink.stdout(line)

        return CommandResult(returncode=0, stdout=stdout, stderr=stderr)


agent = CodexCodingAgent(executor=MyExecutor())
```

The default `HostCommandExecutor` preserves the normal local `subprocess`
behavior. `CodingAgent(provider=..., executor=...)` forwards the same executor
to the selected provider.

Executor contract:

- `CommandRequest.argv` is the complete provider CLI command. `argv[0]` is the
  value returned by `find_binary`.
- `CommandRequest.stdin` is the prompt text to write to the command's standard
  input, then stdin should be closed.
- `CommandRequest.cwd`, `env`, and `timeout` should be honored by the executor.
- Call `sink.started(handle)` once after the command starts. The handle only
  needs `terminate()` and `kill()`.
- Call `sink.stdout(line)` and `sink.stderr(line)` as output is produced. Lines
  should include trailing newlines when the underlying stream provided them.
- Return `CommandResult(returncode, stdout, stderr)` after the command exits.
  The returned text should match what was streamed through the sink.

If you were using the pre-0.5 executor preview, replace
`run_streaming(cmd, ..., on_stdout, on_stderr, on_process_started)` with
`run(request, sink)`. The parser/event APIs remain internal; custom executors
only provide a stable command runtime.

### 4. Instantiate a Specific Provider Directly

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

### 5. Configure MCP Servers

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

Advanced users can register their own providers. `CodingAgent(...)` keeps its
main constructor portable; provider-specific constructor extras should go
through `backend_kwargs`.

```python
from agentshim import BaseCodingAgent, CodingAgent, register_provider


@register_provider("my-agent", aliases=("my-agent-dev",))
class MyAgent(BaseCodingAgent):
    def __init__(
        self,
        model: str | None = None,
        region: str | None = None,
        event_handler=None,
        event_handlers=None,
        mcp_servers=None,
        sandbox=False,
    ):
        self.model = model
        self.region = region
        self.event_handler = event_handler

    def generate(self, prompt: str, cwd=None, timeout=300, silent=False) -> str:
        return f"handled: {prompt}"


agent = CodingAgent(
    provider="my-agent-dev",
    model="demo",
    backend_kwargs={"region": "us-west1"},
)
print(agent.generate("hello"))
```

Notes:

- Registration is import-driven. Your provider is available only after the module defining it has been imported in the current Python process.
- `list_providers()` returns canonical provider names only. Aliases resolve via `get_provider_class(...)` and `CodingAgent(provider=...)`.
- `register_provider(...)` rejects invalid names, abstract classes, and accidental name collisions unless you pass `overwrite=True`.
- If you want `CodingAgent(...)` to instantiate your provider, its constructor should accept the shared kwargs `model`, `event_handler`, `event_handlers`, `mcp_servers`, `sandbox`, and `executor` as needed.
- If your provider needs extra constructor arguments beyond the shared portable set, pass them via `backend_kwargs={...}` when constructing `CodingAgent(...)`.

## Development

```bash
uv sync --dev
bash scripts/format_code.sh --check
bash scripts/check_errors.sh
bash scripts/type_check.sh
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
