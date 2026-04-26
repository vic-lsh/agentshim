# Events

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
