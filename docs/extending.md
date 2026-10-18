# Adding a Provider

A provider integration lives in its own package. The shipped providers are
under `agentshim/providers/<name>/`; an application or third party can keep
the same layout in a package outside `agentshim`. Shared behaviour (tool
pairing, JSON line handling, MCP file merge, schema materialization) lives
in `core/`.

Put argv construction in `provider.py`, stream parsing in `parser.py`, and
provider-specific options beside them. Add a `plugin.py` that exports one
object implementing the core `ProviderPlugin` protocol. Every method is
required. `connect` is the single connection entry point; `connection_modes`
declares which modes it accepts. An unsupported mode raises
`ProviderCapabilityError`. The built-in providers use a `StaticProviderPlugin`
adapter for their fixed declarations, but third-party plugins may compute
answers directly. No provider registers itself on import.

`describe` reports agent capabilities without checking the local install;
`probe` checks local readiness using the supplied executor and environment.
`model_info` returns a `ModelSpec` (including a context window and provenance)
or `None`; `pricing_info` returns a `ModelPricing` for a `PricingContext` or
`None`. These methods can calculate answers dynamically. `known_models` and
`known_prices` enumerate facts for versioned table snapshots and may return
empty tuples even if dynamic lookups are available. `test_support` returns
consumer-test adapters or `None`. Methods are never omitted to signal lack of
a capability. A caller may still pass a `ModelTable` or `PricingTable` to
override or pin lookup facts. Identical snapshot entries from multiple plugins
deduplicate; conflicting entries raise an error.

Import `core/` and `execution/` absolutely (`from agentshim.core.events
import ...`) and your own siblings relatively (`from .parser import ...`).

## The protocol

```python
class ProviderPlugin(Protocol):
    @property
    def name(self) -> str: ...
    def describe(self) -> ProviderProfile: ...
    def connection_modes(self) -> frozenset[TransportKind]: ...
    def connect(self, options: ConnectionOptions) -> Provider | Transport: ...
    def probe(self, executor: CommandExecutor, env: Mapping[str, str], timeout: float) -> ProviderStatus: ...
    def model_info(self, model: str, table: ModelTable | None = None) -> ModelSpec | None: ...
    def pricing_info(self, model: str, context: PricingContext, table: PricingTable | None = None) -> ModelPricing | None: ...
    def known_models(self) -> tuple[ModelSpec, ...]: ...
    def known_prices(self) -> tuple[ModelPricing, ...]: ...
    def test_support(self) -> ProviderTestSupport | None: ...
```

The one-shot `Provider` below is a separate, lower-level protocol returned by
`connect(ConnectionOptions(TransportKind.ONE_SHOT))`. A stream connection
returns `Transport`. The union preserves both existing runtimes behind one
plugin method; the requested kind determines which is returned.

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

`install_mcp` returns an `McpInstallation` whose
`restore() -> str | None` must be idempotent and must never raise: it runs
from the session's `finally`, where an exception would destroy the turn's
result or mask the error it failed with. Return a note describing anything
cleanup could not do normally, and `None` otherwise.

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
Normalize usage with `normalized_usage` into the breakdown in
[Usage and Pricing](pricing.md). Report a
failed tool on `ToolResult.stderr` with a nonzero `exit_code`, never on
`stdout`.

## Test doubles

Expose `scripted_lines(...)` in `scripted.py`, returning the provider's real
stdout for one turn. Put it, `resume_failure_lines`, and any probe or MCP
test adapters in the `ProviderTestSupport` returned by `test_support()`. Consumer test helpers
receive the same catalog as the runtime API and round-trip the stream through
the provider's own parser. If the provider has no native output schema, raise
`ValueError` when `structured_output` is passed rather than ignoring it.

## Tests

Put them in `tests/unit/providers/<name>/` as `test_argv.py`,
`test_parser.py`, `test_provider.py` and `test_scripted.py`, grouping cases
in `Test*` classes. `tests/unit/providers/test_conventions.py` is
parametrized over `provider_names()` and will pick the new provider up
automatically: it pins the profile fields, the prompt staying out of argv,
the resume flag, the scripted round trip, the usage invariant, and
`RawOutput` for non-object JSON. Add the new name to its `RESUME_FLAGS` map.
An external plugin can run the equivalent tests in its own repository;
`tests/external_provider/` is an example that works without adding a module
to the `agentshim` namespace.

## Installing it

```python
from agentshim import Agent, ApprovalPolicy, CliAgent, DEFAULT_CATALOG, NativePermissions
from myagent.plugin import PLUGIN

catalog = DEFAULT_CATALOG.with_plugin(PLUGIN)
cli = CliAgent("myagent", catalog=catalog, model="my-model-id")
agent = Agent(
    "myagent", catalog=catalog,
    permissions=NativePermissions.bypass(), approvals=ApprovalPolicy.DENY,
)
```

At least one transport factory is required, so a stream-only integration does
not need to implement the one-shot `Provider` protocol. `ProviderCatalog.with_plugin` returns a new catalog and rejects duplicate
names. Pass it to `Agent`, `CliAgent`, `probe_provider`, `provider_names`, or
the consumer test helpers. The built-in default remains unchanged. A caller
can still pass a `Provider` instance to `CliAgent` or a `Transport` instance
to `Agent` without using a catalog. A plugin may live in a separately
installed package or a local application; publishing to agentshim is not
required.
