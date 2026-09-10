# Changelog

## 0.6.1 (unreleased)

Additive only: every new field defaults, so a 0.6.0 consumer's existing
`ProviderProfile` construction and every shipped provider keep working
unchanged.

`ProviderProfile` gains four fields:

- `container_env: Mapping[str, str]` (default empty): environment a CLI
  needs to run as root in a container, beyond auth. Claude Code sets
  `IS_SANDBOX=1`, which it requires before accepting
  `--dangerously-skip-permissions` as root; the other four providers need
  nothing extra.
- `state_root_env: str | None` (default `None`): the CLI's own documented
  variable for relocating its primary state directory (`state_dirs[0]`).
  `CLAUDE_CONFIG_DIR` for Claude, `CODEX_HOME` for Codex, `COPILOT_HOME` for
  Copilot; `None` for Gemini and opencode, which document no such variable.
- `auth_files: tuple[str, ...]` (default empty): home-relative paths to the
  files that hold credentials and user configuration, the minimal set to
  copy into another environment so the CLI is already logged in. Every entry
  lies inside a `state_dirs` entry.
- `mcp_config_file: str | None` (default `None`): for a `CONFIG_FILE`
  provider, the workspace-relative path its MCP config is written to for a
  turn (`.mcp.json`, `.gemini/settings.json`, `opencode.json`). The provider
  module defines it as the same constant `install_mcp` writes to, so the two
  cannot drift. `None` for `CLI_FLAGS`/`NONE` providers.

`tests/unit/providers/test_conventions.py` pins these across every provider:
`auth_files` entries lie inside `state_dirs`, `mcp_config_file` is set
exactly when `mcp` is `CONFIG_FILE` and matches where `install_mcp` actually
writes, `state_root_env` is uppercase when set, and `container_env` keys are
uppercase.

## 0.6.0

A restructure, not an upgrade. 0.6 replaces the whole public API of 0.5 and
ships **no compatibility layer**: a 0.5 consumer will not import, let alone
run, until it is migrated. The sections below list every removed and renamed
name so a migration can be done mechanically.

### Restructure

The package is now four layers that import strictly inward:
`agentshim.testing` -> `agentshim/agent.py` -> `agentshim/providers/` ->
(`agentshim/core/`, `agentshim/execution/`). `agentshim/agent.py` is the
composition layer, and the only thing that turns a provider name into a
provider. `import-linter` enforces the ordering as a build gate
(`scripts/check_imports.sh`).

- `agentshim/core/` holds the provider-agnostic types: `TurnRequest`,
  `TurnResult`, `OutputSchema`, the event dataclasses, `TokenUsage`,
  `ProviderUsage`, the error hierarchy, `ProviderProfile`, the `Provider`
  and `StreamParser` protocols, `ParsedTurn`, MCP specs, schema dialect
  checks, and `interactive_env()`.
- `agentshim/execution/` holds the process transport: `CommandExecutor`,
  `HostCommandExecutor`, `TransformingExecutor`.
- `agentshim/providers/<name>/` holds one CLI each, all with the same
  layout: `provider.py`, `parser.py`, `events.py`, `scripted.py`, and an
  `__init__.py` exporting `<Name>Provider`, `PROFILE`, `scripted_lines`.
  All five ship: claude, codex, copilot, gemini, opencode. `fold_usage` is
  not part of that contract: every package has one and no two take the same
  arguments, so it stays private to its own `parser.py`.
- `agentshim/testing/` is new and part of the public API.

`ProviderProfile` declares every optional behaviour, so no caller probes a
provider object: resume, reasoning effort, MCP mechanism, output-schema
style and dialect, state dirs, auth env vars, skill dirs, container install.

### Breaking changes from 0.5

**Removed from the public API.**

| 0.5 | Replacement |
| --- | --- |
| `CodingAgent` | `CliAgent` |
| `BaseCodingAgent` | the `Provider` protocol, in `agentshim/core/provider.py` |
| `BaseAgentSession` | `AgentSession` |
| `ClaudeCodeCodingAgent` | `CliAgent("claude")`, or `agentshim.providers.claude.ClaudeProvider` |
| `CodexCodingAgent` | `CliAgent("codex")` |
| `CopilotCodingAgent` | `CliAgent("copilot")` |
| `GeminiCodingAgent` | `CliAgent("gemini")` |
| `OpencodeCodingAgent` | `CliAgent("opencode")` |
| `get_provider_class(name)` | `get_provider(name)`, which returns an instance |
| `list_providers()` | `provider_names()` |
| `register_provider(...)` | none: pass a `Provider` instance to `CliAgent` |
| `McpServerConfig` | `McpServer` |
| `SandboxConfig` (top level) | `agentshim.providers.claude.SandboxConfig` |

Whole modules are gone: `agentshim.base`, `agentshim.cli_agent`,
`agentshim.events`, `agentshim.executor`, `agentshim.mcp_config`,
`agentshim.sandbox`, `agentshim.usage`, `agentshim.utils`,
`agentshim.subagent`, `agentshim.llm_client`, and the five
`agentshim.<name>_events` compatibility shims. `subagent.py` and
`llm_client.py` made direct LLM API calls, which is not what a CLI shim is
for; the display helpers in `utils.py` are now private to
`ConsoleEventHandler`.

**Renamed.**

| 0.5 | 0.6 |
| --- | --- |
| `session.generate(prompt) -> str` | `session.turn(TurnRequest \| str) -> TurnResult` |
| `get_interactive_env()` | `interactive_env(*, refresh=False)`, now cached per process |
| `build_claude_sandbox_settings()` | `build_settings()` |
| `<Provider>CodingAgent` | `<Provider>Provider` |

**Event handling is typed.** 0.5 dispatched seven optional callbacks
(`on_run_start`, `on_run_end`, `on_thinking`, `on_tool_call`,
`on_tool_result`, `on_usage`, `on_stderr`), three of which were not even on
the protocol and were reached through `getattr`. 0.6 has one method,
`on_event(event)`, taking a frozen dataclass from the `AgentEvent` union:
`RunStarted`, `RunFinished`, `SessionStarted`, `AssistantText`, `Reasoning`,
`ToolCall`, `ToolResult`, `UsageReport`, `Lifecycle`, `Stderr`, `RawOutput`,
`ProviderError`. `on_thinking` covered assistant text and reasoning
together; those are now separate events. `SessionStarted`, `Lifecycle`,
`RawOutput` and `ProviderError` have no 0.5 counterpart.

`event_handler=` and `event_handlers=` are now combined rather than
mutually exclusive; passing both no longer raises.

**Errors are typed.** 0.5 raised bare `RuntimeError` for every CLI failure.
0.6 raises only `AgentShimError` subclasses, and nothing else escapes
`turn()`:

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

`SessionResumeError` is new; there was no 0.5 name for it.

**The prompt is no longer in argv.** 0.5 wrote the prompt to stdin *and*
appended it to argv on three providers: claude as a bare positional, copilot
as `-p <prompt>`, opencode as a positional wrapped in literal double quotes
(`f'"{prompt}"'`, which reached the CLI with the quotes in the string,
since argv is not shell-parsed). 0.6 delivers the prompt on stdin only, on
every provider, so an agent's own `pkill -f` cannot match the CLI by prompt
text and the prompt does not appear in the process table. `ArgvContext` has
no prompt field.

**The opencode default model is gone.** 0.5 substituted
`google-vertex/gemini-3-pro-preview` whenever no model was given. 0.6 omits
`--model` entirely, so opencode uses the model from the user's own config.
Callers who relied on the implicit default must pass `model=` explicitly.

**Dependencies removed.** 0.5 declared `litellm>=1.0.0` and `loguru>=0.7.2`,
and used pydantic transitively through litellm for the MCP server specs.
0.6 declares `dependencies = []`. litellm went with `subagent.py` and
`llm_client.py`; the MCP specs are frozen dataclasses; and logging is a
`Callable[[str], None]` the caller supplies, with `ConsoleEventHandler`
writing to a `TextIO` given at construction instead of to a loguru logger.
`bind_event_handler_context` and `default_event_handler` are gone with it.

**Usage gained two fields.** `TokenUsage` adds `cache_write_input_tokens`
and `reasoning_output_tokens`, so `to_dict()` now emits six keys rather than
four. `ProviderUsage` adds `raw`, which holds the CLI's own usage mapping
for diagnostics and is deliberately excluded from `to_dict()`. Every
provider now normalizes to the invariant `cached_input_tokens <=
input_tokens`.

**Structured output is a first-class request.** `TurnRequest.output_schema`
takes an `OutputSchema`, `ProviderProfile.output_schema` and
`schema_dialect` declare what a provider accepts, and a schema the provider
cannot express raises `SchemaDialectError` before the process starts.

**MCP servers are per-turn, and Claude installs them differently.** 0.5 took
`mcp_servers` on the agent constructor and, for Claude, rendered them into
`--mcp-config <json> --strict-mcp-config`. 0.6 takes them on
`TurnRequest.mcp_servers`, and `ProviderProfile.mcp` declares the mechanism:
Claude, Gemini and opencode merge into a workspace config file
(`.mcp.json`, `.gemini/settings.json`, `opencode.json`), keeping the
original bytes and restoring them when the turn ends, including when it
raised; Codex and Copilot render flags. A config-file provider therefore
needs a `cwd`, and raises `ProviderCapabilityError` without one.

### Fixed

- **Codex tool results.** A failed tool was reported on `ToolResult.stdout`
  with an empty `stderr`, so a renderer showed a failure as success.
  Failures now go on `stderr` with a nonzero `exit_code`, which is the rule
  on all five providers.
- **Non-object JSON lines.** A stdout line that was valid JSON but not an
  object (a bare `42`, a top-level array) is no longer treated as a frame.
  `parse_json_object` returns `None` for blank lines, invalid JSON and
  non-objects alike, and the parser emits `RawOutput` so the line stays
  observable instead of crashing the turn or being dropped.
- **`ConsoleEventHandler` painted failed tool results green.** It chose the
  colour from whether there was any output rather than from `exit_code`, so a
  failure read as a success. Failures are red now, and a failed result with no
  output says so instead of "ran successfully".
- **`AgentSession.forget()` could race a running turn.** It cleared
  `session_id` without the session lock, so an id the in-flight turn was about
  to write survived the forget, or an id nobody else had was dropped. It now
  follows `adopt`'s rule, refusing under the lock while a turn is in flight,
  and returns `bool` to say which happened.
- **`interactive_env()` truncated multi-line variables.** The probe parsed
  `env` output by splitting on newlines, so a value containing one (a key, an
  exported shell function) was cut short and its remaining lines became junk
  keys. The probe asks for `env -0` and splits on NUL, falling back to plain
  `env` where `-0` is not supported.
- **Claude's read-confinement hook quoted paths by hand.** The command was
  assembled by wrapping each part in literal double quotes, which leaves `$`,
  backticks and `"` inside a path live for the shell Claude runs it in. It uses
  `shlex.join` now.
- **Claude tool results rendered as Python dict reprs.** A `tool_result`
  whose content is a list of blocks, which is how Claude sends anything but
  plain text, was flattened with `str()`, so `ToolResult.stdout` carried
  `{'type': 'text', 'text': 'hi'}` instead of `hi`. Text blocks now
  contribute their text and every other block is serialized as JSON.
- **`HttpMcpServer` meant a different transport on every provider.** The same
  spec became SSE on claude and copilot but streamable HTTP on gemini, codex
  and opencode, so a server reachable on one provider silently failed on
  another. `HttpMcpServer` now takes
  `transport: Literal["http", "sse"] = "http"`, and each provider renders it
  the way its CLI names it (`{"type": ...}` on claude and copilot,
  `httpUrl`/`url` on gemini). opencode's single `remote` type and Codex's
  `--config` URL override cannot express the choice, so both work it out from
  the endpoint; `docs/mcp.md` has the table. The default changes claude and
  copilot from SSE to streamable HTTP.
- **Three non-`AgentShimError` exceptions escaped `turn()`**, against the
  documented contract that catching `AgentShimError` is enough. Installing MCP
  servers into a read-only workspace raised `PermissionError` from the config
  write, and now raises `McpConfigError`. A schema carrying a `NaN` or an
  infinity raised `ValueError` from `materialize`, and now raises
  `ProviderCapabilityError`. `TransformingExecutor.check_binary` raised
  `FileNotFoundError` when the transform's own prefix binary (`docker`, a
  sandbox wrapper) was missing, and now raises `CliCheckError`.
  `tests/unit/test_turn_contract.py` pins all three.
- **Atomic writes replaced symlinks.** `atomic_write` renamed its temporary
  file onto the path it was given, so a `.mcp.json` (or `.gemini/settings.json`,
  or `opencode.json`) that was a symlink into a dotfiles checkout became a
  regular file, and the file it pointed at went stale. Symlinks are now
  followed to the file they name. `install_config_file` resolves the same way,
  so the backup, the merged write and the restore all address one inode, and
  the installation reports the resolved path.
- **`$schema` and `$id` were rejected everywhere, and unfixable.** Both
  dialects reported them as unsupported keywords, so a schema straight out of
  a generator failed before the process started, and `normalize()` could not
  repair it because it did not strip them. `dialect_problems` now reports them
  under `STRICT` only, which is where they are genuinely refused (Codex's
  `--output-schema` subset); `OPEN` ignores them the way the CLI does. And
  `normalize()` drops the document metadata, `$schema`, `$id`, `title`,
  `description` and `examples`, so a normalized schema is accepted in either
  dialect. A property named `title` is untouched: `properties` is traversed as
  a map of subschemas, not as keywords.
- **A cancel before the CLI spawned was dropped.** `cancel()` looked only at
  the process handle, which the executor publishes after it has started the
  process. A cancel arriving while the turn was installing MCP servers,
  building argv or spawning therefore did nothing at all. The request is now
  recorded under the session lock and paid out the moment the handle appears.
  It is only ever recorded while a turn is in flight and is cleared when that
  turn ends, so cancelling an idle session still leaves the next turn alone.
- **A timed-out turn emitted no `RunFinished` and never finished its
  parser.** An `AgentShimError` from the executor skipped the run's closing
  path entirely, so a handler saw a run that started and never ended, and
  whatever the parser had read, the session id included, was thrown away. The
  run now closes the same way it does on a clean exit: `RunFinished(None)`,
  `parser.finish()`, adopt the session id, re-raise. `CliTimeoutError` gained
  `partial: ParsedTurn | None` carrying that reading.
- **A turn that named its conversation and then failed was unresumable.** The
  nonzero-exit check raised before `parsed.session_id` was adopted, so the id
  the provider had already printed was lost. Adoption now happens first, and
  is skipped only when `classify_exit` reported `SessionResumeError`, which is
  the provider saying the conversation is gone.
- **MCP restore could destroy the turn.** `ConfigFileInstallation.restore()`
  marked itself done before doing any work and raised `McpConfigError` when
  the agent had left the config file non-JSON or non-object. Raised from the
  session's `finally`, that discarded a successful `TurnResult` or masked the
  real `CliExitError`, and left agentshim's injected entries in the file
  forever. `restore()` now never raises: anything the precise unmerge cannot
  handle falls back to writing the file's original bytes verbatim (removing
  the file when there were none), and it reports what it did as a note the
  session logs through the agent's `log`. `McpInstallation.restore()`
  therefore returns `str | None` rather than `None`.
- **Undecodable CLI output.** The child was read in text mode with the
  platform default codec and no error handler, and the reader thread
  swallowed the resulting `UnicodeDecodeError`. One byte that is not valid
  UTF-8 therefore discarded the rest of the turn's output and returned exit 0
  with an empty transcript, or, with a large output and no timeout, hung the
  turn forever because nobody was draining the child's pipe. The streams are
  now decoded as UTF-8 with `errors="replace"`, the reader closes its end of
  the pipe when it stops for any reason, and a reader that did fail raises
  `CliExitError` instead of presenting a truncated stream as a clean EOF.
- **Stdin deadlock.** 0.5 wrote the whole prompt to the child's stdin on the
  calling thread. A prompt larger than the pipe buffer, sent to a CLI that
  was not reading stdin, blocked the turn forever. `HostCommandExecutor`
  now writes stdin from a helper thread and closes it in `finally`.
- **Callback threading.** 0.5 called the sink directly from two reader
  threads, so an event handler ran concurrently on both and had to do its
  own locking. The executor now drains a queue on the thread that called
  `turn()`, so every callback is serialized there. This is a documented
  contract, not an implementation detail.
- **Claude read-confinement hook.** The hook was registered as
  `"<path>/confine_reads.py" <roots>`, which needs the file to be
  executable and its shebang to name a usable interpreter. It now runs
  through `sys.executable`, so it uses the interpreter agentshim is running
  under.
- **opencode tool status.** Tool parts were reported only when their status
  was `"success"` or `"error"`. opencode's terminal status is `"completed"`,
  so every successful tool call was silently dropped. All three terminal
  states are now paired into `ToolCall` and `ToolResult`, with the part's own
  `time` block supplying the duration.
- **Error frames.** Codex `turn.failed` and top-level `error` frames,
  Gemini `error` frames, opencode `error` frames and Copilot
  `session.error` frames now become `ProviderError` events and set
  `ParsedTurn.error`, instead of being ignored or folded into the turn
  text. Codex stderr lines are `Stderr` events rather than prose in the
  result.

### Added

- `agentshim.testing`: `FakeExecutor`, `FakeRun`, `FakeCommandHandle`,
  `RecordingEventHandler` and `scripted_turn(provider, ...)`. There was no
  0.5 equivalent; a consumer had to write its own `CommandExecutor` fake and
  hand-roll each provider's JSON. `scripted_turn` emits the provider's real
  stream format, produced by `providers/<name>/scripted.py` and round-tripped
  through that provider's real parser, so a consumer's tests never encode a
  provider's wire format.
- `TransformingExecutor`, for running the CLI under an OS sandbox or inside
  a container by rewriting argv. `HostCommandExecutor.check_binary` now goes
  through `run`, so the health check is transformed too and a
  container-executed CLI can actually be probed; the probe is a normal
  command with a pipe for stdin, so it can never inherit a TTY.
- `AgentSession.adopt()`, `forget()`, and a thread-safe `cancel()` that
  terminates the process group and kills it after a grace period.
- Gemini and opencode classify any nonzero exit of a resumed turn as
  `SessionResumeError`, the rule Claude already followed: neither CLI can
  tell a lost session apart from another failure, and a caller holding a
  checkpoint would otherwise offer the same dead id forever. Codex keeps
  matching its explicit "no rollout found" message.
- The Claude parser only reports `structured_output` for a turn that asked
  for a schema, so an unrequested field can never replace the prose answer.
- `TurnRequest.mcp_workspace`: the host directory that receives config-file
  MCP installs when the turn has no host `cwd`, which is the container case
  (the CLI runs at the container path while the config file belongs on the
  bind-mounted host workspace). Defaults to `cwd`.
- `ClaudeProvider`, `CodexProvider`, `CopilotProvider`, `GeminiProvider`,
  `OpencodeProvider` and `SandboxConfig` are exported from `agentshim`
  directly. The docs already told callers to construct them, so they were
  reaching into `agentshim.providers.<name>` for something the public API
  should have carried.
- `scripts/check_imports.sh` and the `[tool.importlinter]` contract.
- `__version__` on the package.

### Known limitations

- **Copilot CLI reports no token usage.** Verified on Copilot CLI 1.0.83: a
  run prints no `assistant.usage` frame and no per-message `outputTokens`,
  so `TurnResult.usage.tokens` is all zeros for this provider. The
  `session.usage_checkpoint` frame it does print carries
  `totalPremiumRequests` and `totalNanoAiu`, which are billing units rather
  than tokens, plus prompt-cache diagnostics (`prompt_tokens`,
  `frontier_tokens`, `tool_tokens`, per-segment `tokens`) that describe how
  the prompt was assembled and not what the turn was charged. Folding those
  into `input_tokens` would report a number that is not the turn's usage, so
  the parser leaves the frame alone.
  `tests/fixtures/copilot/usage_checkpoint_1_0_83.jsonl` is a recording of
  such a run and pins the behaviour. Copilot also reports no cost.
- **Codex and Gemini report no cost.** `TurnResult.cost_usd` is `None` on
  both; only Claude Code and opencode report one.
- **Only Claude Code and Codex accept a native output schema**, and only
  they accept a reasoning effort; asking Gemini, opencode or Copilot for
  either raises `ProviderCapabilityError`. Copilot has an `--effort` flag,
  but its levels are model-specific and unvalidated, so the provider does
  not expose it rather than mistranslating a portable one.
- **Codex cannot carry HTTP headers on an MCP server.** It configures
  servers through `--config` overrides, which have nowhere to put them, so
  an `HttpMcpServer` with headers raises `ProviderCapabilityError` rather
  than having them dropped silently. The other four pass headers through.
- **Resume diagnosis is provider-dependent.** Only Codex reports a lost
  conversation distinguishably, on stderr. Claude maps any nonzero exit on a
  resumed turn to `SessionResumeError`, because `claude --resume` gives no
  distinguishable exit code and a failed resumed turn is unusable either
  way. Gemini, opencode and Copilot give no signal at all, so a failed
  resumed turn raises a plain `CliExitError` there.
