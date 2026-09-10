# Adding a Provider

A provider package holds only argv construction, stream parsing, and
provider-specific options. Everything shared (tool pairing, JSON line
handling, MCP file merge, schema materialization) already lives in `core/`.

Create `agentshim/providers/<name>/` with `provider.py`, `parser.py`,
`events.py` and `scripted.py`, then add one entry to each dict in
`agentshim/providers/__init__.py`.

## The protocol

```python
class Provider(Protocol):
    profile: ProviderProfile
    def build_argv(self, ctx: ArgvContext) -> list[str]: ...
    def new_parser(self, emit: Callable[[AgentEvent], None], *, expect_structured: bool) -> StreamParser: ...
    def install_mcp(self, workspace: Path | None, servers: Sequence[McpServer]) -> McpInstallation: ...
    def classify_exit(self, error: CliExitError, *, resumed: bool) -> AgentShimError: ...
```

`ArgvContext` carries the binary path, model, env, resume id, reasoning
effort, the inline schema or the CLI-visible schema path, the MCP flags, and
the caller's extra args. It deliberately does not carry the prompt: the prompt
always goes on stdin, so an agent's own `pkill -f` cannot match the CLI by
prompt text.

`ProviderProfile` declares every optional behaviour. Fill in all of it,
including `state_dirs`, `auth_env_vars`, `skill_dirs` and
`container_install`: callers read these instead of hardcoding provider names.

## The parser

```python
class StreamParser(Protocol):
    def feed_stdout(self, line: str) -> None: ...
    def feed_stderr(self, line: str) -> None: ...
    def finish(self) -> ParsedTurn: ...
```

Use `parse_json_object(line)`, which returns `None` for blank lines, invalid
JSON, and valid JSON that is not an object; emit `RawOutput` for those rather
than raising. Use `ToolTracker` to pair result frames back to their calls.
Normalize usage so `cached_input_tokens <= input_tokens` holds.

## Test doubles

Expose `scripted_lines(...)` in `scripted.py`, returning the provider's real
stdout for one turn, and register it so `scripted_turn("<name>", ...)` finds
it. Consumers test against that, so it has to round-trip through your own
parser.

## Installing it

```python
from agentshim import CliAgent

agent = CliAgent(MyProvider(), model="...")
```

Providers reached by name go in the dicts in `providers/__init__.py`; there is
no mutable registry and no import side effect.
