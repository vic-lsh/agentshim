# Extending agentshim

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

- Registration is import-driven. Your provider is available only after the
  module defining it has been imported in the current Python process.
- `list_providers()` returns canonical provider names only. Aliases resolve via
  `get_provider_class(...)` and `CodingAgent(provider=...)`.
- `register_provider(...)` rejects invalid names, abstract classes, and
  accidental name collisions unless you pass `overwrite=True`.
- If you want `CodingAgent(...)` to instantiate your provider, its constructor
  should accept the shared kwargs `model`, `event_handler`, `event_handlers`,
  `mcp_servers`, `sandbox`, and `executor` as needed.
- If your provider needs extra constructor arguments beyond the shared portable
  set, pass them via `backend_kwargs={...}` when constructing
  `CodingAgent(...)`.
