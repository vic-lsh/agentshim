# Getting Started

## A single turn

`CliAgent.run` is a one-shot: it opens a throwaway session, runs one turn, and
returns everything the turn produced.

```python
from agentshim import CliAgent

agent = CliAgent("claude", model="sonnet")
result = agent.run("List the top-level packages.", cwd=".")

print(result.text)
print(result.exit_code, result.duration_ms)
```

`model` is an opaque provider-specific string, passed to the CLI unchanged;
`None` leaves the CLI's own default.

Binary lookup and the CLI health check run once, in the constructor, so a
broken install fails immediately rather than halfway through a turn.

## A conversation

A session resumes the provider's own conversation on every turn after the
first.

```python
session = agent.start_session(cwd=".", timeout=600)

session.turn("What does this project do?")
second = session.turn("Which files should I read first?")

assert second.resumed
print(session.session_id)
```

`session.session_id` is readable and writable. `adopt(session_id)` continues a
conversation you checkpointed earlier and returns `False` if the provider
cannot resume or a turn is in flight; `forget()` starts fresh next turn.

## Per-turn options

Anything that varies per turn goes on a `TurnRequest`.

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
        timeout=300,
        reasoning_effort="high",
        output_schema=OutputSchema(schema=schema, host_dir=Path("/tmp/schemas")),
        mcp_servers=[StdioMcpServer(name="issues", command="python", args=["-m", "board.mcp"])],
        env={"CI": "1"},
        extra_args=("--append-system-prompt", "Be terse."),
    )
)

print(result.structured_output)
```

`timeout=None` means no limit. A request field of `None` falls back to the
session default. Only Claude Code and Codex accept an output schema, and only
they accept a reasoning effort; asking any other provider raises
`ProviderCapabilityError` before the process starts. See
[Providers](providers.md).

## Cancelling

`cancel()` is thread-safe and can be called from anywhere while a turn is
running. It terminates the process group, then kills it after the grace
period.

```python
import threading

threading.Timer(30, session.cancel).start()
session.turn("A long-running task.")
```

## Errors

Everything that escapes `turn()` is an `AgentShimError`:

```python
from agentshim import CliExitError, CliTimeoutError, SessionResumeError

try:
    session.turn("...")
except SessionResumeError:
    session.forget()          # the conversation is gone; start over
except CliTimeoutError:
    ...                       # the process group has already been killed
except CliExitError as exc:
    print(exc.returncode, exc.stderr)
```
