# Architecture (0.6)

agentshim runs coding-agent CLIs (Claude Code, Codex, Gemini CLI, opencode,
Copilot CLI) as subprocesses and turns their output into typed events and a
typed turn result. It owns everything that answers "how do I run provider X
and understand what it printed". It does not own application policy: which
provider to use, when to retire a conversation, how to sandbox the host, or
how to render events.

## Layout

```
agentshim/
  __init__.py          public API; everything importable from here is supported
  agent.py             CliAgent, AgentSession: composition over providers/
  oneshot.py           OneShotTransport: CliAgent behind the Transport protocol
  session.py           Session, TurnTicket, Turn: the thin shell around the policy
  runtime.py           Agent: transport + permissions + clock + recovery policy
  core/                provider-agnostic
    conversation.py    ConversationSpec, Conversation, Transport protocols
    session_policy.py  SessionPolicy: the pure recovery state machine, RetryPolicy
    checkpoints.py     Checkpoint, CheckpointStore, InMemoryCheckpointStore
    turn.py            TurnRequest, TurnResult, OutputSchema
    events.py          event dataclasses, AgentEvent union, handlers
    usage.py           TokenUsage, TokenWeights, ProviderUsage
    pricing.py         ModelPricing, PricingTable, price_for, cost_usd
    errors.py          exception hierarchy
    permissions.py     NativeMode, NativePermissions, ApprovalPolicy
    clock.py           StopSignal, Clock, SystemClock
    ids.py             IdAllocator, RandomIds
    profile.py         ProviderProfile and capability enums
    provider.py        Provider and StreamParser protocols, McpInstallation,
                       ArgvContext, ParsedTurn
    stream.py          ToolTracker, parse_json_object
    schema.py          output-schema dialect checks and materialization
    mcp.py             McpServer specs, JSON config-file merge and restore
    env.py             interactive_env(), from `bash -i -c 'env -0'`
    _files.py          private: atomic_write, shared by schema.py and mcp.py
  execution/           process transport
    executor.py        CommandRequest, CommandResult, CommandHandle, sinks, CommandExecutor
    host.py            HostCommandExecutor
    transform.py       TransformingExecutor
    process.py         SpawnRequest, ProcessOutput, Process (long-lived processes)
    confinement.py     Confinement protocol, confine(), PathMap
    docker.py          DockerExecConfinement
  providers/           one folder per CLI, all with the same layout
    __init__.py        get_provider(name), provider_names(), get_scripted_lines(name),
                       get_stream_transport(name, ...)
    claude/            provider.py, parser.py, events.py, scripted.py,
                       sandbox.py, hooks/, stream_transport.py (ClaudeStreamTransport),
                       stream_argv.py
    codex/             provider.py, parser.py, events.py, scripted.py,
                       app_server/ (generated protocol types, see below)
    copilot/           provider.py, parser.py, events.py, scripted.py
    gemini/            provider.py, parser.py, events.py, scripted.py
    opencode/          provider.py, parser.py, events.py, scripted.py
  testing/             test doubles shipped for consumers
    __init__.py        FakeExecutor, FakeRun, RecordingEventHandler, scripted_turn,
                       scripted_resume_failure, installed_mcp_servers
    clock.py           FakeClock, SequentialIds
    process.py         FakeProcess, FakePeer, EchoPeer, SilentPeer, ReplayGates, GateMarker
    claude_stream.py   ClaudeStreamPeers, ClaudeStreamPeer, ClaudePeerTurn, ClaudeRecordedPeer
    confinement.py     FakeConfinement
    contracts.py       ProcessContract, ConfinementContract, ClockContract
```

Rules:

- Imports go one way, innermost last: `testing/` -> `agent.py` ->
  `providers/` -> (`core/`, `execution/`). `core/` and `execution/` never
  import from `providers/` or from `agent.py`; `providers/` imports only from
  `core/` and `execution/`. `execution/` raises the `core/` error types, which
  is the one edge inside the innermost layer.
- `agent.py` is the composition layer, and the only reason it is not in
  `core/`: turning a provider *name* into a provider means importing
  `providers/`, which `core/` may not do.
- The layering is a build gate, not a convention. `[tool.importlinter]` in
  `pyproject.toml` declares it and `scripts/check_imports.sh` runs it.
- A provider folder holds only argv construction, stream parsing, and
  provider-specific options. Shared behaviour (tool pairing, JSON line
  handling, MCP file merge, schema materialization) lives in `core/`.
- Every provider package has the same shape: `provider.py`, `parser.py`,
  `events.py`, `scripted.py`, and an `__init__.py` exporting
  `<Name>Provider`, `PROFILE`, `<Name>StreamParser`, `scripted_lines`,
  `resume_failure_lines`, and `mcp_entry` where the provider has one.
  `fold_usage` is deliberately not on that list: every package has one and no
  two take the same arguments, so a caller could never write code against the
  name. It stays private to its own `parser.py`. Anything beyond that is
  provider-specific (Claude's `sandbox.py` and `hooks/`, Codex's and
  Copilot's `parse_mcp_servers`, kept next to the argv renderer it inverts).
- Imports inside `agentshim/` are absolute across packages
  (`from agentshim.core.events import ...`) and relative between siblings of
  the same package (`from .parser import ...`).
- `tests/unit/providers/<name>/` holds `test_argv.py`, `test_parser.py`,
  `test_provider.py` and `test_scripted.py`, cases grouped in `Test*`
  classes. `tests/unit/providers/test_conventions.py` is parametrized over
  `provider_names()` and pins the rules a per-provider suite cannot see one
  package dropping.
- No module-level mutable registries. `providers/__init__.py` maps names to
  provider factories with a plain dict.
- No required runtime dependencies. Logging goes through a
  `Callable[[str], None]` supplied by the caller.
- Python 3.10 compatible, `from __future__ import annotations` everywhere,
  pyright strict on `agentshim/`, ruff clean.
- No required runtime dependencies means no pydantic either: the MCP server
  specs and every other value type are plain frozen dataclasses.

## Long-lived processes, confinement, clock, permissions

These pieces are additive. `CliAgent` and the providers do not use them yet;
they are the base the session and transport layers are built on.

### Process

`CommandExecutor.run` is one request, one result. `CommandExecutor.spawn(SpawnRequest)`
starts a long-lived `Process` the caller writes to and reads from.

```python
class Process(Protocol):
    def write(self, data: str) -> None: ...        # ProcessClosedError if stdin is closed or the process is gone
    def close_stdin(self) -> None: ...             # idempotent
    def next_output(self, timeout: float | None) -> ProcessOutput | None: ...
    def terminate(self) -> None: ...               # idempotent, signals the process group
    def kill(self) -> None: ...                    # idempotent, signals the process group
    def wait(self, timeout: float | None) -> int | None: ...
```

Output is pulled, never pushed: `next_output` returns `StdoutLine`, `StderrLine`
or `ProcessExited`, or `None` when `timeout` passes with nothing. Lines keep
their trailing newline, as `CommandStreamSink` lines do. `ProcessExited` is
always last (every line written before the exit is delivered first) and is
returned again by every later call. Because the caller decides when output is
consumed, a simulation controls the interleaving and no callback runs on a
foreign thread. `HostCommandExecutor.spawn` starts the process in its own
session; reader threads feed one queue inside `HostProcess`, which is the real
I/O shell. `TransformingExecutor.spawn` applies the same transform as `run`.

### Confinement

A `Confinement` bounds the processes agentshim starts from outside the agent:
`wrap(argv, cwd)` returns the host argv that runs `argv` inside it,
`agent_path(host_path)` maps a host path to the path the agent sees, `env` is
the environment the confined process needs, and `reap()` kills every process of
this kind agentshim left behind, including those of an earlier host process.
`confine(executor, confinement)` applies one to any executor: `run`, `spawn` and
the binary health check go through `wrap`, and `find_binary` trusts the bare
name because the binary lives inside the confinement.

`DockerExecConfinement` runs `docker exec -i` into a running container. The
container id is read on every call. Environment values are passed by name
(`-e KEY`) with the values in the docker client's own environment, which
`confine` merges in, so secrets never appear in the host process table.
Every process carries `AGENTSHIM_CONFINED=1`; `reap` kills exactly the
processes whose environment has it, so a nested daemon in the same container is
left alone, and a missing container counts as nothing to reap.

### Clock and ids

`Clock` (`monotonic()`, `wait(seconds, stop)`) and `IdAllocator`
(`new_id(prefix)`) are injected so waits can be cancelled with a `StopSignal`
and simulations run without real time. `SystemClock` and `RandomIds` are the
real ones; `FakeClock` and `SequentialIds` the test doubles.

### Native permissions

`NativePermissions` says what the agent's own sandbox allows: `bypass()`,
`read_only()` or `workspace_write(writable_roots, network=)`. Roots and network
are only valid with `workspace_write`; roots must be absolute. `BYPASS` turns
the agent's sandbox and approvals off and is only safe inside an outside
`Confinement`. `ApprovalPolicy` (`DENY`, `FAIL_TURN`) says what a transport does
when the agent asks for permission or input: it never waits for a human.
`ProviderProfile.native_permission_modes` lists the modes a provider supports
(default: `BYPASS` only).

### Test doubles and contract suites

`FakeProcess` is driven by a `FakePeer` (`on_start`, `on_stdin`,
`on_stdin_closed`) that answers reactively, so a protocol fake can reply by
request id. `ReplayGates` pause a conversation: a peer emits `GateMarker(name)`
and the process delivers nothing past it while that gate is closed.
`FakeExecutor(peers=...)` spawns them; `FakeConfinement` records wraps and
reaps. `agentshim.testing.contracts` holds `ProcessContract`,
`ConfinementContract` and `ClockContract`: subclass one as `Test<Impl>`, implement
its factories, and pytest runs the inherited tests against your implementation.

`tests/unit/test_provider_name_literals.py` fails when a string constant equal
to a provider name appears outside `providers/` and `testing/`.

## Core types

### Turn

```python
@dataclass(frozen=True)
class OutputSchema:
    schema: Mapping[str, Any]
    host_dir: Path                 # where a schema file may be written (absolute, host)
    cli_dir: str | None = None     # host_dir as the CLI process sees it; defaults to str(host_dir)

@dataclass(frozen=True)
class TurnRequest:
    prompt: str
    cwd: str | None = None                      # None: session default
    timeout: float | None = None                # None: session default
    output_schema: OutputSchema | None = None
    reasoning_effort: str | None = None
    extra_args: Sequence[str] = ()              # appended verbatim to argv
    env: Mapping[str, str] | None = None        # overlay on the agent env for this turn
    mcp_servers: Sequence[McpServer] = ()       # installed before the turn, restored after
    mcp_workspace: Path | None = None           # host dir for config-file MCP installs; defaults to cwd

@dataclass(frozen=True)
class TurnResult:
    text: str                       # final assistant text ("" if none)
    structured_output: Any | None   # schema-conformant payload when output_schema was set
    session_id: str | None          # provider conversation id after this turn
    resumed: bool                   # whether the turn continued an earlier conversation
    usage: ProviderUsage
    cost_usd: float | None
    duration_ms: int                # wall clock, measured by the session
    exit_code: int
```

`OutputSchema` carries two directories because a container-executed CLI
resolves paths inside the container. Claude Code takes the schema inline, so
only `schema` is used. Codex takes a path, so the session writes
`<host_dir>/<sha>.json` and passes `<cli_dir>/<sha>.json`.

### Events

All events are frozen dataclasses. `AgentEvent` is their union.

| Event | Fields | Meaning |
|---|---|---|
| `RunStarted` | `argv: tuple[str, ...]` | the CLI process is about to start |
| `RunFinished` | `exit_code: int \| None` | the CLI process exited |
| `SessionStarted` | `session_id: str` | the provider named the conversation |
| `AssistantText` | `text: str` | assistant-facing message text |
| `Reasoning` | `text: str` | thinking or reasoning text |
| `ToolCall` | `tool_id: str \| None, tool: str, args: Mapping[str, Any] \| str \| None` | a tool was invoked |
| `ToolResult` | `tool_id: str \| None, tool: str, stdout: str, stderr: str, exit_code: int \| None, duration_s: float \| None` | a tool finished |
| `UsageReport` | `usage: ProviderUsage, cost_usd: float \| None` | provider accounting |
| `Lifecycle` | `kind: str, detail: str` | provider plumbing (`thread_started`, `turn_started`, `turn_completed`, ...) |
| `Stderr` | `text: str` | one stderr line |
| `RawOutput` | `text: str` | one stdout line that was not a provider event |
| `ProviderError` | `message: str` | the provider reported an error |

```python
class AgentEventHandler(Protocol):
    def on_event(self, event: AgentEvent) -> None: ...

class EventHandlerBase:          # no-op on_event; subclass and override
class CompositeEventHandler:     # fan-out, in order
class NullEventHandler:
class ConsoleEventHandler:       # human-readable rendering to a text stream
```

Thread contract: `on_event` runs on the thread that called
`AgentSession.turn()`. The executor serializes reader-thread output before
the session sees it.

Usage reporting: every provider emits at least one `UsageReport` per turn
when its CLI reports usage. `ProviderUsage.tokens` obeys
the normalized breakdown in [Usage and Pricing](pricing.md) on every
provider (Claude, opencode and Copilot report cache tokens disjoint from
input tokens and their parsers fold them in). Copilot CLI 1.0.83 reports no token counts at all, so its counts are
zero; the invariant still holds.

Tool results: a tool that failed is reported on `ToolResult.stderr` with a
nonzero `exit_code` and an empty `stdout`, on every provider.

### Usage

`TokenUsage` carries `input_tokens`, `cache_read_input_tokens`,
`cache_write_input_tokens`, `cache_write_1h_input_tokens`, `output_tokens`,
`reasoning_output_tokens`, `turns` and the deprecated `cached_input_tokens`
alias, derives `uncached_input_tokens`, and adds field-wise so per-turn usages
fold into a session total. `pricing.py` holds the static price table and
`cost_usd`.
`ProviderUsage.raw` holds the last raw provider usage mapping for
diagnostics. Each provider package normalizes its CLI's counts in a
`fold_usage` function private to its own `parser.py`; the five signatures
have nothing in common, so the name is not part of the package contract.

### Errors

```
AgentShimError
  CliNotFoundError            binary not on PATH
  CliCheckError               binary found but the health check failed
  TurnFailedError             a turn failed: kind (FailureKind), detail
    CliExitError              nonzero exit: argv, returncode, stdout, stderr
      SessionResumeError      a resumed turn failed because the conversation is gone: session_id
  TurnTimeoutError            a turn overran its budget: timeout
    CliTimeoutError           argv, partial: ParsedTurn | None
  ContinuityError             a turn required a conversation the session does not hold
  SessionStateError           closed session or conversation, a second concurrent turn,
                              a stale or reused ticket
  TurnCancelledError          an interrupt arrived before the turn could start
  ProviderCapabilityError     the provider cannot do what the request asked
    SchemaDialectError        problems: list[str]
  McpConfigError              config file unreadable, unwritable, or not an object
```

Nothing else escapes `turn()`: no bare `RuntimeError`, no
`subprocess.TimeoutExpired`, no `OSError` from a config write, no
`ValueError` from a schema JSON cannot express.
`tests/unit/test_turn_contract.py` pins the paths that once broke this.
The one documented exception is an event handler that raises: it fails the
turn it is watching, with its own exception.

### Profile

```python
class McpMechanism(Enum): NONE, CONFIG_FILE, CLI_FLAGS
class OutputSchemaStyle(Enum): NONE, INLINE_JSON, FILE_PATH
class SchemaDialect(Enum): STRICT, OPEN

@dataclass(frozen=True)
class ProviderProfile:
    name: str                          # "claude"
    display_name: str                  # "Claude Code"
    binary: str                        # "claude"
    supports_resume: bool
    supports_reasoning_effort: bool
    mcp: McpMechanism
    output_schema: OutputSchemaStyle
    schema_dialect: SchemaDialect | None
    state_dirs: tuple[str, ...]        # home-relative, all platforms (".claude", ".claude.json")
    darwin_state_dirs: tuple[str, ...] # home-relative, macOS only
    auth_env_vars: tuple[str, ...]     # ("ANTHROPIC_API_KEY", ...)
    skill_dirs: tuple[str, ...]        # workspace-relative skill discovery dirs
    container_install: tuple[str, ...] # shell commands installing the CLI on Debian
    container_env: Mapping[str, str]   # env a root container run needs beyond auth; default empty
    state_root_env: str | None         # var that relocates state_dirs[0]; None if undocumented
    auth_files: tuple[str, ...]        # home-relative authentication files, inside state_dirs
    credential_files: tuple[str, ...]  # the subset of auth_files that holds refreshable credentials
    resume_state_paths: tuple[str, ...]  # conversation history a resume reads back; empty if unknown
    mcp_config_file: str | None        # workspace-relative MCP config path; set iff mcp is CONFIG_FILE
```

`STRICT` is Codex's `--output-schema` subset. Codex sends the schema
verbatim as a Responses API `json_schema` format with `strict: true`, so the
subset is OpenAI's strict structured outputs: every object (including a
nullable one, `"type": ["object", "null"]`) declares `properties` (`{}` for
none) and sets `additionalProperties: false`, `required` lists exactly the
keys of `properties` (an optional value is declared required and nullable
instead), the root is an object and not an `anyOf`, and every `$ref`
resolves inside the document. The check recurses through `properties`,
`items`, `anyOf`, `$defs` and `definitions`; a `$ref` target is checked where
it is defined. `OPEN` (Claude Code) keeps optional properties and accepts
schema-valued, `true` or absent `additionalProperties`.

The last four fields are additive (0.6.1): every one defaults, so an
existing keyword-built `ProviderProfile` keeps working unchanged.
`container_env` covers what a CLI needs to run as root in a container beyond
credentials (Claude Code's `IS_SANDBOX=1`, required before it accepts
`--dangerously-skip-permissions` as root); it is empty for the other four.
`state_root_env` names the provider's own documented variable for relocating
`state_dirs[0]` (`CLAUDE_CONFIG_DIR`, `CODEX_HOME`, `COPILOT_HOME`); it is
`None` where the CLI documents none, rather than guessed. `auth_files` are
the minimal home-relative files to copy elsewhere so the CLI is already
logged in; every entry lies inside a `state_dirs` entry. `credential_files` is the
subset of them that holds refreshable login credentials rather than settings, for
callers that share those writable so a token refresh is not lost. `mcp_config_file` is
the workspace-relative path a `CONFIG_FILE` provider writes its MCP config
to for a turn (`.mcp.json`, `.gemini/settings.json`, `opencode.json`); the
provider module defines it as the same constant `install_mcp` writes to, so
the two cannot drift, and it is `None` for `CLI_FLAGS`/`NONE` providers.
Codex therefore declares only `.codex/auth.json` as authentication state.
Its user `config.toml` is not required: model, reasoning effort, shell policy,
output schema, and MCP servers are supplied as invocation-scoped flags.

### Provider protocol

```python
class StreamParser(Protocol):
    def feed_stdout(self, line: str) -> None: ...
    def feed_stderr(self, line: str) -> None: ...
    def finish(self) -> ParsedTurn: ...       # text, structured_output, session_id, usage, cost_usd, error

class McpInstallation(Protocol):
    argv: Sequence[str]                       # flags to append (CLI_FLAGS providers), else ()
    def restore(self) -> str | None: ...      # idempotent, never raises; a note to log

class Provider(Protocol):
    profile: ProviderProfile
    def build_argv(self, ctx: ArgvContext) -> list[str]: ...
    def new_parser(self, emit: Callable[[AgentEvent], None], *,
                   expect_structured: bool) -> StreamParser: ...
    def install_mcp(self, workspace: Path | None, servers: Sequence[McpServer]) -> McpInstallation: ...
    def classify_exit(self, error: CliExitError, *, resumed: bool) -> AgentShimError: ...

@dataclass(frozen=True)
class ArgvContext:
    binary_path: str
    model: str | None
    env: Mapping[str, str]
    resume_session_id: str | None
    reasoning_effort: str | None
    schema_inline: str | None        # compact JSON for INLINE_JSON providers
    schema_path: str | None          # CLI-visible path for FILE_PATH providers
    mcp_argv: Sequence[str]
    extra_args: Sequence[str]
    cwd: str | None                  # the turn's cwd, for validating paths; never rendered
```

The prompt is always delivered on stdin and never appears in argv (an
agent's own `pkill -f` must not be able to match the CLI by prompt text).

Custom providers keep the original `new_parser(emit, *, expect_structured)`
signature. A parser that needs invocation context may implement the optional,
runtime-checkable `ContextualStreamParser` protocol with
`configure(context: ParserContext) -> None`. The session calls it before
streaming, passing `ParserContext(previous_usage=..., resumed=...)`. Parsers
without this method continue working unchanged. Codex uses the previous raw
report as a fixed baseline; without one on resume, it marks the increment
unknown and retains the raw total for the next invocation.

### CliAgent and AgentSession (the one-shot path)

`Agent` and `Session` (see [sessions](sessions.md)) are the primary API;
`CliAgent` and `AgentSession` remain for code that drives one CLI process per
turn directly, and are what `OneShotTransport` wraps.

```python
class CliAgent:
    def __init__(self, provider: str | Provider, *, model: str | None = None,
                 executor: CommandExecutor | None = None,
                 env: Mapping[str, str] | None = None,        # None: interactive_env()
                 event_handler: AgentEventHandler | None = None,
                 event_handlers: Sequence[AgentEventHandler] = (),
                 check_timeout: float = 15.0,
                 log: Callable[[str], None] | None = None) -> None
    profile: ProviderProfile
    binary_path: str
    env: dict[str, str]
    def derive(self, *, model, event_handler) -> CliAgent    # same checked install, other model and handler
    def start_session(self, *, cwd: str | None = None, timeout: float | None = None,
                      session_id: str | None = None,
                      previous_usage: ProviderUsage | None = None) -> AgentSession
    def run(self, request: TurnRequest | str, *, cwd=None, timeout=None) -> TurnResult   # one-shot

class AgentSession:
    session_id: str | None              # readable and writable
    last_result: TurnResult | None
    def adopt(self, session_id: str, *, previous_usage: ProviderUsage | None = None) -> bool   # False if unsupported or a conversation is live
    def forget(self) -> bool                   # False if a turn is in flight
    def turn(self, request: TurnRequest | str) -> TurnResult
    def cancel(self, grace_s: float = 5.0) -> None   # thread-safe; terminate then kill
```

`turn()` in order: mark the session busy; resolve cwd, timeout, env; check
capabilities (`reasoning_effort`, `output_schema`) against the profile;
materialize the schema; install MCP servers; build argv; run through the
executor with the parser as the sink; emit `RunFinished` and call
`parser.finish()`; adopt `session_id` from the parser; on nonzero exit build
`CliExitError` and raise `provider.classify_exit(...)`; in `finally` restore
MCP config, clear the process handle and the cancel request, and mark the
session idle. Binary lookup and the health check run once in
`CliAgent.__init__`.

An `AgentShimError` out of the executor (a timeout, a transport failure) takes
the same closing path: `RunFinished(None)`, `parser.finish()`, adopt the
session id, then re-raise. A `CliTimeoutError` carries the partial
`ParsedTurn` on `.partial`.

## Execution

```python
@dataclass(frozen=True)
class CommandRequest: argv, stdin: str | None, cwd: str | None, env: Mapping[str, str], timeout: float | None
@dataclass(frozen=True)
class CommandResult: returncode: int, stdout: str, stderr: str

class CommandHandle(Protocol):
    def terminate(self) -> None: ...
    def kill(self) -> None: ...

class CommandStreamSink(Protocol):
    def started(self, handle: CommandHandle) -> None: ...
    def stdout(self, line: str) -> None: ...
    def stderr(self, line: str) -> None: ...

class CommandExecutor(Protocol):
    def find_binary(self, name: str, env: Mapping[str, str]) -> str: ...      # CliNotFoundError
    def check_binary(self, path: str, env: Mapping[str, str], *, timeout: float) -> None: ...  # CliCheckError
    def run(self, request: CommandRequest, sink: CommandStreamSink) -> CommandResult: ...      # CliTimeoutError
```

`HostCommandExecutor` starts the process in its own session, writes stdin
from a helper thread, reads stdout and stderr on helper threads into a queue,
and drains the queue on the calling thread so sink callbacks are serialized.
On timeout it kills the process group and raises `CliTimeoutError`. Pipes are
closed in `finally`. A sink exception propagates after the process is
killed.

`TransformingExecutor(inner, transform, find_binary=None)` rewrites every
`CommandRequest` before the inner executor sees it. It is how a caller wraps
argv in an OS sandbox or prefixes it with `docker exec`. `check_binary` goes
through `run`, so the transform applies to the health check too.

## Providers

| | claude | codex | gemini | opencode | copilot |
|---|---|---|---|---|---|
| resume | `--resume <id>` | `exec resume <id>` | `--resume <id>` | `run --session <id>` | `--resume <id>` |
| MCP | `.mcp.json` (config file) | `--config mcp_servers.*` flags | `.gemini/settings.json` | `opencode.json` | `--additional-mcp-config` |
| output schema | `--json-schema <inline>`, OPEN | `--output-schema <path>`, STRICT | none | none | none |
| reasoning effort | `--effort` | `--config model_reasoning_effort` | none | none | none |
| stream | `stream-json` | `--json` | `--output-format stream-json` | `run --format json` | `--output-format json` |

Codex always passes `--skip-git-repo-check` and, when the env has a `PATH`,
`--config shell_environment_policy.set.PATH=...` so tools the agent runs see
the launcher's PATH. `codex exec resume <id>` also needs the literal `-`
positional right after the thread id: without it the CLI ignores stdin.

Codex's own sandbox is a `CodexProvider` option. Without one, argv carries
`--dangerously-bypass-approvals-and-sandbox`, as in every earlier release.
With one, it carries `--config` overrides instead (`sandbox_mode`,
`approval_policy`, and for `workspace-write` all four
`sandbox_workspace_write.*` keys), because `exec resume` has no `--sandbox`
flag. `parse_sandbox` inverts that rendering and lives next to it, like
`parse_mcp_servers`. Every value goes through `providers/codex/_toml.py`,
whose encoder escapes control characters: Codex keeps an override that fails
to parse as TOML as a raw string, so a bad literal changes type silently.

A sandboxed turn also passes `--ignore-rules`: an exec-policy `allow` rule
runs its command outside the sandbox, so a rule in the user's
`~/.codex/rules` would otherwise widen it. `excluded_commands` is the one
case that needs rules to load. Codex reads them only from `.rules` files
(`$CODEX_HOME/rules/`, trusted projects), never from `--config`, so
`providers/codex/rules.py` renders them and `install_rules` writes them into a
caller-dedicated `CODEX_HOME`. The provider then omits `--ignore-rules` and
refuses the turn unless `CODEX_HOME` is absolute and outside every directory
the sandbox lets commands write (the turn's `cwd`, `writable_roots`, and the
temp dirs when writable): a writable home would let a command add a rule
exempting itself. That check is why `ArgvContext` carries `cwd`.

Claude's optional settings-file sandbox (`providers/claude/sandbox.py`) is a
provider option, not a portable constructor argument. Its read-confinement
hook is invoked through `sys.executable`. Caller hooks (`ClaudeHook`,
`providers/claude/user_hooks.py`) are the other Claude option; `build_settings`
merges both into the one `--settings` object, agentshim's hook first on each
event, and emits no `--settings` when there is neither.

MCP config-file installation merges servers into the provider's JSON file,
keeps the original bytes, and restores on `restore()`. If the file changed
during the turn, only the entries agentshim added are removed. `restore()`
never raises: it runs from the session's `finally`, where an exception would
destroy a successful `TurnResult` or mask the error the turn failed with. A
file the agent left unreadable as JSON falls back to the original bytes
written verbatim, and the returned note is logged through the agent's `log`.

### providers/codex/app_server

`protocol.py` holds typed, dependency-free Python for the subset of the
`codex app-server` JSON protocol that a transport needs. It is generated by
`scripts/generate_codex_protocol.py` from the JSON Schema that the CLI itself
exports (`codex app-server generate-json-schema`), never edited by hand.
`_wire.py` is the hand-written runtime it calls (decoder combinators,
`CodexProtocolError`, `encode`). `transport.py` (`CodexAppServerTransport`) speaks the protocol, with
`channel.py` (JSON-RPC framing and deadlines), `permissions.py`,
`approvals.py`, `errors.py`, `usage.py`, `mcp.py`, `items.py` and
`profile.py` as its pure pieces. Its test double is
`testing/codex_app_server.py`.

What is generated is decided by `scripts/codex_protocol/allowlist.json`: the
requests the client sends and their responses, the server notifications and
server requests it handles, and a few `opaque` types kept as raw JSON. The
generator keeps the transitive closure of those roots and checks in the
pruned result as `scripts/codex_protocol/schema.json` (with the codex-cli
version), so regeneration needs no CLI. The pruned schema lives outside the
wheel; only `protocol.py` ships.

Shape of the output:

- Objects are frozen, keyword-only dataclasses with snake_case fields.
  `to_wire()` emits the exact JSON the server expects: camelCase keys, `None`
  omitted unless the schema marks the member required (then `null`). Closed
  string sets are `str` enums. Tagged unions are one dataclass per variant
  plus an alias such as `ThreadItem`, decoded by `thread_item_from_wire`.
- Direction decides strictness. A type reachable from anything the server
  sends is tolerant: `from_wire` ignores unknown keys, an unlisted enum value
  stays a plain `str`, and a union gets an `Unknown...` variant that keeps
  the raw JSON and re-encodes it unchanged. A type only the client sends is
  strict: an unlisted enum value or tag raises. A missing or wrongly typed
  required member raises `CodexProtocolError` naming the path
  (`Notification.params.turn.id`) in both.
- Each client request's params class carries `METHOD` and a
  `parse_result(result)` that decodes its response type.
- Envelopes are `Notification`, `ServerRequest`, `Response`, `ErrorResponse`
  and `ClientRequest`/`ClientNotification`. The wire has no `jsonrpc` member:
  `parse_server_message` tells the kinds apart by `method`/`id`/`result`/
  `error`, and an unlisted method keeps its params as `UnknownParams` (a
  server request must still be answered). `reply(id, result)` builds the typed
  reply to a server request; `parse_client_message` is the inverse direction
  for fakes and tests.

The schema has no version number; the CLI version is the version. On a CLI
bump, regenerate (see development.md), read the diff of `protocol.py`, and
re-record `tests/fixtures/codex_app_server/` if the behaviour changed.

## Testing support

`agentshim.testing` is part of the public API.

```python
@dataclass
class FakeRun: stdout: Sequence[str] = (), stderr: Sequence[str] = (), returncode: int = 0, timeout: bool = False

class FakeExecutor:                      # CommandExecutor
    def __init__(self, runs: FakeRun | Sequence[FakeRun] | Callable[[CommandRequest], FakeRun], *, binaries: Mapping[str, str] | None = None)
    requests: list[CommandRequest]
    handles: list[FakeCommandHandle]     # each records terminate()/kill()
    checked: list[str]                   # paths check_binary() was called on

class RecordingEventHandler:             # events: list[AgentEvent]

def scripted_turn(provider: str, *, text: str = "", session_id: str | None = None,
                  usage: TokenUsage | None = None, tool_calls: Sequence[tuple[str, Mapping[str, Any], str]] = (),
                  structured_output: object | None = None, returncode: int = 0) -> FakeRun
```

`scripted_turn` emits the provider's real stream format, so a consumer can
test its integration without knowing any provider's JSON shape. It finds that
format through `providers.get_scripted_lines(name)`, which maps a provider
name to the `scripted_lines(...)` function in `providers/<name>/scripted.py`.
Consumer tests should build agents with `FakeExecutor` and assert on
`TurnResult` and the typed events, never on internal attributes. A provider
with no native output schema raises `ValueError` when `structured_output` is
passed, rather than quietly producing a turn without one.

`FakeCommandHandle` records `terminate()`/`kill()`;
`RecordingEventHandler.of_type(kind)` filters what it recorded.

```python
def scripted_resume_failure(provider: str, *, session_id: str | None = None) -> FakeRun

def installed_mcp_servers(
    provider: str, request: CommandRequest, workspace: Path
) -> dict[str, dict[str, Any]]
```

`scripted_resume_failure` builds a `FakeRun` that fails the way *provider*
recognises a lost resumed conversation, so a consumer can test
`SessionResumeError` handling without knowing any provider's wire format.
Serve it to a session that has already `adopt`-ed a session id: on claude,
codex, gemini and opencode the turn's `resumed` flag plus the nonzero exit is
what `classify_exit` turns into `SessionResumeError`; served to a *fresh*
turn instead, or to Copilot at all, the same run raises a plain
`CliExitError` (Copilot gives no signal that tells a lost session apart from
any other failure). It finds the scripted failure the same way `scripted_turn`
does, through `providers.get_resume_failure_lines(name)`, which maps to
`resume_failure_lines(...)` in `providers/<name>/scripted.py` — a required
entry, so a new provider cannot ship without one. `session_id` is folded into
the scripted message for realism only; what a raised error actually reports
always comes from the resumed turn's own argv.

`installed_mcp_servers` returns the MCP servers one turn installed, keyed by
server name, each a plain dict in a canonical shape regardless of how the
provider itself renders it: `command`, `args` and `env` for a server started
over stdio, `url` and `transport` for one reached over HTTP. For a
CONFIG_FILE provider it reads the config file the provider writes, using that
provider's own filename/path and server-key constants; for a CLI_FLAGS
provider it parses the servers back out of `request.argv` using a
`parse_mcp_servers` function kept next to the argv renderer it inverts
(`providers/codex/provider.py`, `providers/copilot/provider.py`), so the two
cannot drift apart. A CONFIG_FILE provider's config file exists only for the
lifetime of the turn: the library restores it in the session's `finally`, so
`installed_mcp_servers` has to be called from inside a `FakeExecutor` `run`
callback, while the turn is still in flight, not after `agent.run(...)`
returns.

End-to-end tests under `tests/e2e/` run the real CLIs and are skipped unless
`AGENTSHIM_E2E=1` and the binary is on PATH, so CI never runs them.
`AGENTSHIM_E2E_GEMINI_MODEL` and `AGENTSHIM_E2E_OPENCODE_MODEL` name the
model those two suites use. See [development](development.md).

## Decisions

Design choices that are not obvious from the types alone. Each one is pinned
by a test. For what changed from the previous release and how to migrate,
see the
[changelog](https://github.com/vic-lsh/agentshim/blob/main/CHANGELOG.md).

**`ParsedTurn` lives in `core/provider.py`.** The `StreamParser` protocol
referred to it without defining it. It is a frozen dataclass carrying `text`,
`structured_output`, `session_id`, `usage`, `cost_usd`, `error`.

**`core/schema.py` also exports `normalize(schema, dialect)`.** The dialect
check is pure and never mutates its input, which is what makes it safe to run
on a caller's schema. But generators such as Pydantic omit defaulted
properties from `required` and leave `additionalProperties` unset, and a
caller feeding one of those to a `STRICT` provider needs it fixed up.
Splitting the two lets the caller decide: `dialect_problems` reports,
`normalize` rewrites, and the session only ever calls the first.

`normalize` preserves descriptions, titles and examples at every schema node,
so field guidance reaches the model. Real CLI probes on 2026-10-04 confirmed
Codex and Claude accept these annotations. Claude accepts `$ref` annotation
siblings; Codex rejects them, so `STRICT` normalization moves them onto an
`anyOf` wrapper containing the reference. It only drops
metadata rejected by the selected dialect: `$schema` and `$id` under `STRICT`.
The `OPEN` dialect accepts those document identifiers and `oneOf`; `STRICT`
rejects `oneOf`. Dialect policy lives in `core/schema.py` and is shared by
normalization and checking. `normalize` keeps an open map open, since closing
it would change what the schema accepts, so
`dialect_problems(normalize(s, d), d)` checks whether a generated schema can go
native.

**`CliAgent` and `AgentSession` live in `agentshim/agent.py`, above
`providers/`.** `CliAgent("claude")` has to work, so something must turn a
provider name into a provider, and `core/` must not depend on `providers/`.
Hiding that dependency in a function-local import inside `core/` made the
module-level graph look acyclic while the real one was not. Putting the
composition in its own layer above `providers/` makes the dependency an
ordinary top-level import and leaves `core/` genuinely provider-agnostic.
`import-linter` enforces the ordering, and `test_public_api.py` still walks
the AST of every `core/` and `execution/` module as a second check.

**`event_handler` and `event_handlers` are combined, not exclusive.**
Raising when both are passed would add a failure mode to a constructor that
already does I/O, for a call that has one obvious reading.

**A provider's non-portable options are attributes of the concrete provider
class.** Claude's sandbox needs `CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR=1`
in the CLI's environment, and nothing on the `Provider` protocol carries
environment. A caller that wants the sandbox constructs the provider
explicitly anyway, so it reads `ClaudeProvider.sandbox_env` and merges it into
`CliAgent(env=...)`. Adding an env hook to the protocol would make every
provider implement something only one of them needs.

**`expect_structured` makes the Claude parser fall back to `result`.** Older
Claude Code builds put the schema-conformant payload in the `result` text
instead of a dedicated `structured_output` field. When a schema was requested
and the field is absent, the parser tries to decode `result` as JSON, and
leaves `structured_output` as `None` when that fails. Prose is never forced
into a structured payload.

**Final text: the terminal `result` frame wins; otherwise the assistant text
blocks are joined with a newline.** A provider that summarizes its own turn
is more accurate than reassembling the stream, and joining is only the
fallback for providers that print no terminal frame.

**`HostCommandExecutor.check_binary` goes through `run`.** The spec required
this of `TransformingExecutor`; doing it in the host executor too means the
health-check probe is a normal command with a pipe for stdin, so it can never
inherit a TTY and become a stopped process that deadlocks the parent.

**`CommandResult.returncode` is `int`, never `None`.** The executor always
waits, and a killed process still has a code.

**A failed tool result goes on `ToolResult.stderr` with a nonzero
`exit_code`, on every provider.** The event has separate streams; putting a
failure on `stdout` would make a renderer show it as success. What counts as
failure is per-CLI (Claude's `is_error`, Codex's nonzero `exit_code` or
`status: failed`, Gemini's and opencode's error status, Copilot's
`success: false`), but the reporting is not: `stdout` is empty, the message
is on `stderr`, and `exit_code` is the CLI's own code where it reports one
and `1` where it only reports a boolean.
`tests/unit/providers/test_conventions.py` pins this and the other rules
across all five providers.

**Each provider's parser classifies a failed turn as a `FailureKind`.** The
parser sees the structured frames a CLI writes about a failed request, so it
sets `ParsedTurn.error_kind`, and the session puts that and the stream's
error text on `CliExitError.kind` and `.detail`. Claude reads the assistant
frame's `error` kind, the result frame's `api_error_status`, and, for older
builds, `API Error: <status>` in the result text, and reports `SCHEMA` for the
`error_max_structured_output_retries` subtype with the last rejected
`StructuredOutput` call's validation errors as the detail; Codex reads the text of its
last `error`/`turn.failed` event, and never reports `SCHEMA` because
`--output-schema` is sent as a strict response format that constrains decoding
(a `maxLength` or `minItems` is met by truncating or padding the output, exit 0). Each reads stderr only when the stream
reported nothing, and never tool output. The other providers report `OTHER`.
The error text goes into the message because Claude reports its failures in
the stream and leaves stderr empty.

**Claude, Gemini and opencode map an unclassified nonzero exit on a resumed
turn to `SessionResumeError`.** A Claude failure the parser classified as
`TRANSIENT`, `USAGE_LIMIT`, `AUTH` or `SCHEMA` happened inside a conversation that
resumed, so it keeps its kind and the session keeps the conversation. None of those CLIs gives a distinguishable exit for
a missing transcript, and a resumed turn that failed is unusable either way:
the caller has to start a fresh conversation, and without the typed error a
caller holding a checkpoint would offer the same dead id forever. The session
id is recovered from argv. Codex is the exception: it names a lost rollout on
stderr, so only that case is classified.

**`adopt` and `forget` both refuse while a turn is in flight**, tracked by
the session's idle flag rather than by the presence of a process handle: the
handle only appears once the executor has started the process, which is too
late. Both return `bool` rather than raising, because racing a turn is a
scheduling question the caller can retry, not a programming error.

**`cancel` records the request when there is no handle yet.** The same
too-late window (installing MCP servers, building argv, spawning) would
otherwise make `cancel()` a silent no-op on a turn that has already begun. The
flag is set only while the session is busy and is cleared in `turn()`'s
`finally`, so it can never carry over to a turn that starts later, and
`_set_handle` terminates the process as soon as one exists.

**A nonzero exit adopts the session id before it raises.** A provider that
named the conversation and then failed leaves something the caller can resume;
raising first strands it. The exception is `SessionResumeError`, where the
provider is saying the conversation is gone.

**Truncation in `ConsoleEventHandler` has no per-tool exceptions.** Giving
named tools a larger budget is caller policy; a caller who wants it writes
their own handler.
