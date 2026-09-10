# Events

Every turn emits a stream of frozen event dataclasses. `AgentEvent` is their
union.

| Event | Fields | Meaning |
|---|---|---|
| `RunStarted` | `argv` | the CLI process is about to start |
| `RunFinished` | `exit_code` | the CLI process exited |
| `SessionStarted` | `session_id` | the provider named the conversation |
| `AssistantText` | `text` | assistant-facing message text |
| `Reasoning` | `text` | thinking or reasoning text |
| `ToolCall` | `tool_id`, `tool`, `args` | a tool was invoked |
| `ToolResult` | `tool_id`, `tool`, `stdout`, `stderr`, `exit_code`, `duration_s` | a tool finished |
| `UsageReport` | `usage`, `cost_usd` | provider accounting |
| `Lifecycle` | `kind`, `detail` | provider plumbing |
| `Stderr` | `text` | one stderr line |
| `RawOutput` | `text` | one stdout line that was not a provider event |
| `ProviderError` | `message` | the provider reported an error |

## Handling them

A handler is anything with `on_event(event)`.

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

`event_handlers=[...]` takes several; passing both spellings composes them in
order. `ConsoleEventHandler` renders to any text stream, `NullEventHandler`
drops everything, and `CompositeEventHandler` fans out explicitly.

```python
import sys
from agentshim import CliAgent, ConsoleEventHandler

agent = CliAgent("claude", event_handlers=[ConsoleEventHandler(sys.stderr), Watcher()])
```

## Thread contract

`on_event` always runs on the thread that called `AgentSession.turn()`. The
executor reads the CLI's pipes on helper threads but drains them on the
calling thread, so a handler needs no locking of its own and an exception it
raises propagates out of `turn()` after the process is killed.

## Usage

Every provider emits at least one `UsageReport` per turn when its CLI reports
usage, and the last one matches `TurnResult.usage`. `ProviderUsage.tokens`
obeys `cached_input_tokens <= input_tokens` on every provider: Claude reports
cache tokens disjoint from input tokens, and its parser folds them in.
`ProviderUsage.raw` keeps the CLI's own mapping for diagnostics.
