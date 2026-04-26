# Events

`agentshim` normalizes provider-specific streaming output into a small,
portable event interface. Claude Code, Codex, Copilot, Gemini, and Opencode all
emit different native event streams; the event handler API gives callers one
place to observe the common concepts across those tools.

Use events when you want to render live progress, capture tool activity, record
usage, or connect agent runs to your own logging or tracing system without
parsing each provider's CLI output yourself.

## Portable Events

Custom handlers can implement these common callbacks:

| Event | Called when | Notes |
| --- | --- | --- |
| `on_thinking(text)` | The agent emits assistant text, reasoning/progress text, or other provider status text. | Text may arrive incrementally. Handlers should append or stream it rather than assuming one complete message per call. |
| `on_tool_call(tool, args=None)` | The provider reports that the agent is starting or requesting a tool call. | `tool` is the provider's tool name. `args` is usually a dict, but can be a string or `None` when that is all the provider exposes. |
| `on_tool_result(tool, stdout="", stderr="", exit_code=None, duration=None)` | The provider reports a completed tool result. | Providers vary in how much result metadata they expose, so handlers should tolerate missing `exit_code` and `duration`. |
| `on_usage(usage)` | The provider reports per-turn token or usage information. | The payload is a normalized best-effort dict. Older handlers can omit this method. |

These events are intentionally broad. They are not a lossless copy of every
provider's native event schema; they are the stable interface for common
observability and UI use cases across coding-agent CLIs.

## Optional Runtime Hooks

Some built-in handlers also understand lower-level runtime hooks:

| Hook | Called when |
| --- | --- |
| `on_run_start(command)` | The underlying provider CLI command is about to start. |
| `on_run_end(exit_code=None)` | The underlying provider CLI command has exited. |
| `on_stderr(text)` | The provider process writes stderr that is not parsed as a structured provider event. |

These hooks are useful for terminal renderers and debugging. They are optional:
custom handlers focused on portable agent events can ignore them.

## Default Console Output

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
        print(text, end="")

    def on_tool_call(self, tool: str, args: dict | str | None = None) -> None:
        print(f"tool started: {tool} {args or ''}")

    def on_tool_result(
        self,
        tool: str,
        stdout: str = "",
        stderr: str = "",
        exit_code: int | None = None,
        duration: float | None = None,
    ) -> None:
        output = stdout or stderr
        print(f"tool finished: {tool} exit={exit_code} {output}")

    def on_usage(self, usage: dict) -> None:
        print(f"usage: {usage}")


agent = CodingAgent(provider="claude", event_handler=MyHandler())
agent.generate("Inspect this repository.")
```

You can implement only the events your application needs when the provider path
checks for optional callbacks. For maximum compatibility across providers and
handler composition, define all four portable event methods.

```python
class MinimalHandler:
    def on_thinking(self, text: str) -> None:
        ...

    def on_tool_call(self, tool: str, args: dict | str | None = None) -> None:
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
```

To keep the default console output and add your own handler, compose them
explicitly.

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

You can also build the composition yourself.

```python
from agentshim import CompositeEventHandler, ConsoleEventHandler

handler = CompositeEventHandler([ConsoleEventHandler(), MyHandler()])
agent = CodingAgent(provider="codex", event_handler=handler)
```

Use `silent=True` to suppress the default console handler when you have not
provided any handler.

```python
agent = CodingAgent(provider="claude")
reply = agent.generate("Return only the answer.", silent=True)
```
