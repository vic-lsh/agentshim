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
  core/                provider-agnostic
    turn.py            TurnRequest, TurnResult, OutputSchema
    events.py          event dataclasses, AgentEvent union, handlers
    usage.py           TokenUsage, ProviderUsage
    errors.py          exception hierarchy
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
  providers/           one folder per CLI, all with the same layout
    __init__.py        get_provider(name), provider_names(), get_scripted_lines(name)
    claude/            provider.py, parser.py, events.py, scripted.py,
                       sandbox.py, hooks/
    codex/             provider.py, parser.py, events.py, scripted.py
    copilot/           provider.py, parser.py, events.py, scripted.py
    gemini/            provider.py, parser.py, events.py, scripted.py
    opencode/          provider.py, parser.py, events.py, scripted.py
  testing/             test doubles shipped for consumers
    __init__.py        FakeExecutor, FakeRun, RecordingEventHandler, scripted_turn
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
  `<Name>Provider`, `PROFILE`, `<Name>StreamParser`, `scripted_lines`, and
  `mcp_entry` where the provider has one. `fold_usage` is deliberately not on
  that list: every package has one and no two take the same arguments, so a
  caller could never write code against the name. It stays private to its own
  `parser.py`. Anything beyond that is provider-specific (Claude's
  `sandbox.py` and `hooks/`).
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
`cached_input_tokens <= input_tokens` on every provider (Claude and Copilot
report cache tokens disjoint from input tokens and their parsers fold them
in). Copilot CLI 1.0.83 reports no token counts at all, so its counts are
zero; the invariant still holds.

Tool results: a tool that failed is reported on `ToolResult.stderr` with a
nonzero `exit_code` and an empty `stdout`, on every provider.

### Usage

`TokenUsage` carries `input_tokens`, `output_tokens`, `cached_input_tokens`,
`cache_write_input_tokens`, `reasoning_output_tokens` and `turns`, and adds
field-wise so per-turn usages fold into a session total.
`ProviderUsage.raw` holds the last raw provider usage mapping for
diagnostics. Each provider package normalizes its CLI's counts in a
`fold_usage` function private to its own `parser.py`; the five signatures
have nothing in common, so the name is not part of the package contract.

### Errors

```
AgentShimError
  CliNotFoundError            binary not on PATH
  CliCheckError               binary found but the health check failed
  CliExitError                nonzero exit: argv, returncode, stdout, stderr
    SessionResumeError        a resumed turn failed because the conversation is gone: session_id
  CliTimeoutError             argv, timeout, partial: ParsedTurn | None
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
    auth_files: tuple[str, ...]        # home-relative credential/config files, inside state_dirs
    mcp_config_file: str | None        # workspace-relative MCP config path; set iff mcp is CONFIG_FILE
```

`STRICT` is Codex's `--output-schema` subset: every object declares its
properties and forbids undeclared keys. `OPEN` accepts schema-valued or
`true` `additionalProperties`.

The last four fields are additive (0.6.1): every one defaults, so an
existing keyword-built `ProviderProfile` keeps working unchanged.
`container_env` covers what a CLI needs to run as root in a container beyond
credentials (Claude Code's `IS_SANDBOX=1`, required before it accepts
`--dangerously-skip-permissions` as root); it is empty for the other four.
`state_root_env` names the provider's own documented variable for relocating
`state_dirs[0]` (`CLAUDE_CONFIG_DIR`, `CODEX_HOME`, `COPILOT_HOME`); it is
`None` where the CLI documents none, rather than guessed. `auth_files` are
the minimal home-relative files to copy elsewhere so the CLI is already
logged in; every entry lies inside a `state_dirs` entry. `mcp_config_file` is
the workspace-relative path a `CONFIG_FILE` provider writes its MCP config
to for a turn (`.mcp.json`, `.gemini/settings.json`, `opencode.json`); the
provider module defines it as the same constant `install_mcp` writes to, so
the two cannot drift, and it is `None` for `CLI_FLAGS`/`NONE` providers.

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
    def new_parser(self, emit: Callable[[AgentEvent], None], *, expect_structured: bool) -> StreamParser: ...
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
```

The prompt is always delivered on stdin and never appears in argv (an
agent's own `pkill -f` must not be able to match the CLI by prompt text).

### Agent and session

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
    def start_session(self, *, cwd: str | None = None, timeout: float | None = None,
                      session_id: str | None = None) -> AgentSession
    def run(self, request: TurnRequest | str, *, cwd=None, timeout=None) -> TurnResult   # one-shot

class AgentSession:
    session_id: str | None              # readable and writable
    last_result: TurnResult | None
    def adopt(self, session_id: str) -> bool   # False if unsupported or a conversation is live
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

Claude's optional settings-file sandbox (`providers/claude/sandbox.py`) is a
provider option, not a portable constructor argument. Its read-confinement
hook is invoked through `sys.executable`.

MCP config-file installation merges servers into the provider's JSON file,
keeps the original bytes, and restores on `restore()`. If the file changed
during the turn, only the entries agentshim added are removed. `restore()`
never raises: it runs from the session's `finally`, where an exception would
destroy a successful `TurnResult` or mask the error the turn failed with. A
file the agent left unreadable as JSON falls back to the original bytes
written verbatim, and the returned note is logged through the agent's `log`.

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

`normalize` also drops the document metadata (`$schema`, `$id`, `title`,
`description`, `examples`) that a generator emits and Codex's subset refuses,
so a normalized schema passes `dialect_problems` in either dialect. The
`OPEN` dialect does not report `$schema` or `$id` at all: a CLI that accepts
open-ended schemas ignores them.

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

**Claude, Gemini and opencode map *any* nonzero exit on a resumed turn to
`SessionResumeError`.** None of those CLIs gives a distinguishable exit for
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

