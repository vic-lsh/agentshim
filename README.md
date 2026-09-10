# agentshim

`agentshim` runs coding-agent CLIs (Claude Code, Codex, Gemini CLI, opencode,
Copilot CLI) as subprocesses and turns their output into typed events and a
typed turn result.

It owns everything that answers "how do I run provider X and understand what
it printed". It does not own application policy: which provider to use, when
to retire a conversation, how to sandbox the host, or how to render events.

No required runtime dependencies. Python 3.10+.

## What it includes

- `CliAgent` / `AgentSession`: one turn at a time, resumable, cancellable
- typed events delivered on the calling thread
- normalized token accounting where `cached_input_tokens <= input_tokens` on
  every provider
- declared capabilities on `ProviderProfile`, so no caller probes a provider
- injectable `CommandExecutor`s for running the CLI in a container or over a
  remote shell
- MCP server installation and restoration per turn
- native structured output with per-provider schema dialect checks
- `agentshim.testing`: doubles that emit each provider's real stream format

0.6 ships the core, the execution layer, and all five providers: Claude
Code, Codex, Gemini CLI, opencode, and Copilot CLI. It is not
source-compatible with 0.5; see [`CHANGELOG.md`](CHANGELOG.md) for what
changed and how to migrate.

## Install

```bash
uv add agentshim
```

agentshim does not bundle the agent CLIs. Install and authenticate the one
you want (`claude`, `codex`, `gemini`, `opencode`, or `copilot`) yourself.

## One turn

```python
from agentshim import CliAgent

agent = CliAgent("claude", model="sonnet")
result = agent.run("Write a short summary of this codebase.", cwd=".")

print(result.text)
print(result.usage.tokens.input_tokens, result.cost_usd, result.duration_ms)
```

`model` is an opaque provider-specific string, passed through to the CLI
unchanged; `None` leaves the CLI's own default.

Binary lookup and the CLI health check run once, in the constructor, so a
broken install fails immediately rather than halfway through a turn.

## A conversation

```python
session = agent.start_session(cwd=".", timeout=600)

session.turn("What does this project do?")
second = session.turn("Which files should I read first?")

assert second.resumed
print(session.session_id)
```

`adopt(session_id)` continues a conversation you checkpointed earlier and
returns `False` if the provider cannot resume or a turn is in flight;
`forget()` starts fresh on the next turn. `cancel()` is thread-safe: it
terminates the process group, then kills it after a grace period.

## Per-turn options

```python
from pathlib import Path
from agentshim import OutputSchema, StdioMcpServer, TurnRequest

schema = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}

result = session.turn(
    TurnRequest(
        prompt="Summarize the failing test.",
        cwd="/workspace",
        timeout=300,                       # None means no limit
        reasoning_effort="high",
        output_schema=OutputSchema(schema=schema, host_dir=Path("/tmp/schemas")),
        mcp_servers=[StdioMcpServer(name="issues", command="python", args=["-m", "board.mcp"])],
        env={"CI": "1"},
        extra_args=("--append-system-prompt", "Be terse."),
    )
)

print(result.structured_output)
```

MCP servers are installed before the turn and restored after it, including
when the turn fails. Asking for something the provider cannot do raises
`ProviderCapabilityError` before the process starts.

## Events

```python
from agentshim import AssistantText, CliAgent, EventHandlerBase, ToolCall

class Watcher(EventHandlerBase):
    def on_event(self, event):
        if isinstance(event, ToolCall):
            print("tool:", event.tool)
        elif isinstance(event, AssistantText):
            print(event.text, end="")

agent = CliAgent("claude", event_handler=Watcher())
```

`on_event` always runs on the thread that called `turn()`. The executor reads
the CLI's pipes on helper threads but drains them on the calling thread, so a
handler needs no locking of its own.

`ConsoleEventHandler` renders to any text stream; `CompositeEventHandler` fans
out; `NullEventHandler` drops everything.

## Executors

Implement `CommandExecutor` to run a provider CLI somewhere other than the
local host. `TransformingExecutor` covers the common case of rewriting argv,
and applies to the health check too.

```python
from dataclasses import replace
from agentshim import CliAgent, CommandRequest, HostCommandExecutor, TransformingExecutor

def in_container(request: CommandRequest) -> CommandRequest:
    return replace(request, argv=["docker", "exec", "-i", "workspace", *request.argv])

agent = CliAgent("claude", executor=TransformingExecutor(HostCommandExecutor(), in_container))
```

## Testing against agentshim

`agentshim.testing` emits each provider's real stream format, so your tests
never encode a provider's JSON shape.

```python
from agentshim import AssistantText, CliAgent
from agentshim.testing import FakeExecutor, RecordingEventHandler, scripted_turn

events = RecordingEventHandler()
agent = CliAgent(
    "claude",
    executor=FakeExecutor(scripted_turn("claude", text="pong", session_id="s1")),
    event_handler=events,
)

result = agent.run("ping")
assert result.text == "pong"
assert result.session_id == "s1"
assert any(isinstance(event, AssistantText) for event in events.events)
```

Assert on `TurnResult` and the typed events, never on internal attributes.

## Errors

Everything that escapes `turn()` is an `AgentShimError`:

```
AgentShimError
  CliNotFoundError            binary not on PATH
  CliCheckError               binary found but the health check failed
  CliExitError                nonzero exit: argv, returncode, stdout, stderr
    SessionResumeError        the conversation is gone: session_id
  CliTimeoutError             argv, timeout
  ProviderCapabilityError     the provider cannot do what the request asked
    SchemaDialectError        problems: list[str]
  McpConfigError              config file unreadable or not an object
```

## Adding a provider

See [`docs/extending.md`](docs/extending.md) and
[`docs/architecture.md`](docs/architecture.md). A provider package holds only
argv construction, stream parsing, and provider-specific options; everything
shared already lives in `agentshim/core/`.

## Development

```bash
uv sync --dev
uv run pytest
./scripts/format_code.sh --check
./scripts/check_errors.sh
./scripts/type_check.sh
./scripts/check_imports.sh
```

End-to-end tests under `tests/e2e/` run the real CLIs. They are skipped
unless `AGENTSHIM_E2E=1` and the binary is on PATH, so CI never runs them.

```bash
AGENTSHIM_E2E=1 uv run pytest tests/e2e -q
```

Gemini needs a model the account is entitled to, and opencode takes one when
the model in your own opencode config is not the one to test:

```bash
AGENTSHIM_E2E=1 AGENTSHIM_E2E_GEMINI_MODEL=gemini-2.5-flash \
  uv run pytest tests/e2e/test_gemini_e2e.py -q
```

See [`docs/development.md`](docs/development.md) for the full gate list.

```bash
uv build          # package
uv publish        # release

uv run --group docs mkdocs build --strict
```
