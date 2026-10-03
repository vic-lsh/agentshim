# Changelog

## Unreleased

A session can leave out the user's own CLI configuration. Additive.

### Added

- `ConfigScope` (`ALL`, `PROJECT`) as a session option on `start_session` and
  `run`, `ProviderProfile.config_scopes`, and `ArgvContext.config_scope`.
  `PROJECT` keeps the user's settings, hooks, global instructions, notify
  commands, profiles and memory out of every turn. Claude Code:
  `--setting-sources project,local` and `autoMemoryEnabled=false`. Codex: a
  dedicated `CODEX_HOME` plus `--ignore-user-config --disable memories`;
  `build_argv` refuses the scope in the user's own home. Copilot, Gemini and
  opencode refuse it.
- `ProviderProfile.config_home_files` and `prepare_config_home`, which build
  that dedicated home holding only the login and return the environment that
  points the CLI at it.

## 0.12.0 (2026-10-03)

A turn whose output never matched its schema says so. Additive except that
Claude's `error_max_structured_output_retries` is no longer `OTHER`.

### Added

- `FailureKind.SCHEMA`: the provider gave up producing output that matches the
  requested schema. Claude Code reports it for the
  `error_max_structured_output_retries` result, with the validation errors of
  the last rejected `StructuredOutput` call on `CliExitError.detail`. A resumed
  turn that fails this way keeps its conversation. Codex never reports it:
  `--output-schema` constrains decoding, so its output always parses against
  the schema.
- `scripted_failure("claude", FailureKind.SCHEMA)`.

### Changed

- `scripted_failure(provider, FailureKind.OTHER)` scripts a failure that is
  not about the schema (Claude `error_during_execution`, Codex running out of
  context). `scripted_failure("codex", FailureKind.SCHEMA)` raises
  `ValueError`.

## 0.11.0 (2026-10-03)

A failed turn says why it failed. Additive except that a classified failure of
a resumed Claude turn is no longer a `SessionResumeError`.

### Added

- `FailureKind` (`TRANSIENT`, `USAGE_LIMIT`, `AUTH`, `OTHER`) on
  `CliExitError.kind`, and the stream's error text on `CliExitError.detail`.
  Claude Code classifies from the assistant frame's `error` kind, the result
  frame's `api_error_status`, and `API Error: <status>` text; Codex from its
  `error` and `turn.failed` messages. Copilot, Gemini and opencode report
  `OTHER`.
- `ParsedTurn.error_kind`, set by a provider's parser.
- `agentshim.testing.scripted_failure(provider, kind)` and
  `providers.get_failure_lines`: a real failed Claude or Codex run of each kind.

### Changed

- A resumed Claude turn that failed with a classified kind (an overload, a
  usage limit, a login problem) raises `CliExitError` with that kind instead of
  `SessionResumeError`, and the session keeps the conversation.

### Fixed

- `CliExitError`'s message was empty when Claude reported the failure only in
  its stream (for example `error_max_structured_output_retries`): the message
  now carries the stream's error text as well as stderr.

## 0.10.1 (2026-10-03)

### Fixed

- Codex `McpScope.SESSION` failed the whole turn with `invalid transport`
  when the CLI could not see a config file agentshim had scanned (a confined
  or containerized home): `enabled=false` alone defines an entry with no
  transport. Each disabled server now restates its own `command` or `url`, so
  the override is a complete, disabled entry whether or not the file is seen.

## 0.10.0 (2026-10-03)

MCP isolation. Additive: the default scope keeps today's behaviour. Adds a
runtime dependency on `tomli` for Python 3.10 only (Codex config parsing).

### Added

- `McpScope` (`ALL`, `SESSION`) and `start_session(mcp_scope=...)` /
  `run(mcp_scope=...)`. `SESSION` connects the CLI only to the servers in
  `TurnRequest.mcp_servers`; a turn given none sees no MCP server. It is a
  separate option from `SkillScope` because a provider may enforce one and not
  the other; each is refused independently.
- `ProviderProfile.mcp_scopes` declares the scopes a provider enforces;
  asking for another raises `ProviderCapabilityError` from `start_session`.
- `ArgvContext.mcp_scope` and `ArgvContext.mcp_servers` carry the scope and
  the turn's servers to `build_argv`.
- Claude Code: `SESSION` passes `--strict-mcp-config` plus the turn's servers
  inline as `--mcp-config`. This drops user and project (`.mcp.json`) servers,
  plugin servers and claude.ai account connectors.
- Codex: `SESSION` sets `enabled = false` on every `mcp_servers` entry found
  in `$CODEX_HOME/config.toml`, `/etc/codex/config.toml` and
  `<cwd>/.codex/config.toml` that the turn was not given, and sets
  `features.plugins=false` and `features.apps=false` (plugin servers and the
  account's ChatGPT apps).
- Gemini, opencode and Copilot support `ALL` only.
- Live e2e: a server seeded into the CLI's default configuration is reachable
  under `ALL` and not under `SESSION` (`tests/e2e/test_mcp_scope_e2e.py`).

### Changed

- `parse_mcp_servers` ignores `enabled=false` overrides: they disable a
  configured server and do not define one.

## 0.9.0 (2026-10-03)

Skill isolation. Additive: the default scope keeps today's behaviour.

### Added

- `SkillScope` (`ALL`, `PROJECT`) and `start_session(skill_scope=...)` /
  `run(skill_scope=...)`. `PROJECT` offers only the workspace's skills and
  the CLI's built-in ones, not the user's personal skills or plugins.
- `ProviderProfile.skill_scopes` declares the scopes a provider enforces;
  asking for another raises `ProviderCapabilityError` from `start_session`.
- `ArgvContext.skill_scope` carries the session's scope to `build_argv`.
- Claude Code: `PROJECT` passes `--setting-sources project,local`, which also
  skips user settings and `~/.claude/CLAUDE.md`; credentials still work.
- Codex: `PROJECT` sets `features.plugins=false` and disables every user
  `SKILL.md` (`$CODEX_HOME/skills` except `.system`, `~/.agents/skills`)
  through `skills.config`.
- Gemini, opencode and Copilot support `ALL` only.
- Live e2e: `SkillScope.PROJECT` hides a skill seeded in a relocated user
  home while the workspace skill stays offered (`tests/e2e/test_skills_e2e.py`).

## 0.8.0 (2026-10-03)

Skill observability. Additive: a caller that ignores the new events and the
new `TurnResult` field sees no change, but an exhaustive match over
`AgentEvent` gains two members.

### Added

- `SkillsDiscovered(names)` and `SkillInvoked(name, source_path, tool_id)`
  events. `SkillInvoked` sits directly after the `ToolCall` that loaded the
  skill.
- `TurnResult.skills`: a `SkillSummary` (`discovered`, `invocations`,
  `invoked`, `invocation_count`) folded from those events by `SkillTracker`,
  which callers can also register as a handler to cover several turns.
  `None` means unknown, never zero.
- `ProviderProfile.skill_discovery` and `skill_invocation`, each a
  `SkillSignal` (`NONE`, `STRUCTURED`, `INFERRED`).
- Claude Code: discovery from the `system/init` `skills` list; a load is a
  `Skill` tool call or a `Read` of a `SKILL.md`.
- Codex: a load is inferred from a shell command that names a `SKILL.md`
  under a `skills/` directory. `codex exec --json` does not list offered
  skills, so discovery is unknown.
- Gemini, opencode and Copilot report unknown for both.
- `scripted_turn(..., skills_offered=[...], skills_invoked=[...])` scripts the
  offered list (Claude) and skill loads (Claude, Codex) in each provider's own
  format; a provider whose stream cannot carry them raises `ValueError`.
- Recorded skill streams under `tests/fixtures/{claude,codex}/` and a live
  e2e suite, `tests/e2e/test_skills_e2e.py`, with a positive and a negative
  turn per provider.

### Fixed

- `normalize(schema, SchemaDialect.STRICT)` no longer closes an open map
  (`additionalProperties` a schema or `true`), which silently turned
  `dict[str, float]` into an object that can only be empty. The map is kept
  and `dialect_problems` on the normalized schema reports it, so
  `dialect_problems(normalize(s, d), d) == []` is the check for whether a
  generated schema can go native.

## 0.7.0 (2026-09-27)

First-class cache accounting and a static pricing table. Additive except one
change of meaning: `cached_input_tokens` now counts cache reads only (see
Changed).

### Added

- `TokenUsage.cache_read_input_tokens` (input served from the prompt cache),
  `cache_write_1h_input_tokens` (the one-hour-TTL part of cache writes, from
  Claude's `cache_creation` breakdown) and the derived
  `uncached_input_tokens` (`input - cache_read - cache_write`). `to_dict()`
  gains the three keys; no key was removed.
- Codex turns now report `cache_write_input_tokens` and
  `reasoning_output_tokens` from `turn.completed` (both used to be dropped).
- `TokenUsage` checks its invariants on construction:
  `cache_read + cache_write <= input`, `cache_write_1h <= cache_write`,
  `reasoning <= output`, no negative counts. Parsers build it through the new
  `normalized_usage`, which clamps a CLI's inconsistent counts instead.
- `TokenUsage.from_dict` reads a `to_dict()` mapping back, including one
  written before 0.7.0.
- `TokenWeights` and `TokenUsage.weighted_total(weights)`: a total weighted by
  token class (uncached input, cache reads, 5m and 1h cache writes, output,
  reasoning), each token weighted once.
- A static, versioned pricing table (`agentshim.core.pricing`):
  `ModelPricing`, `PricingTable`, `default_pricing()`,
  `price_for(provider, model, table=None)` (`None` for an unknown model, never
  zero), `cost_usd(usage, pricing)` and `ModelPricing.relative_weights()`.
  It covers OpenAI GPT-6 (astra, sol, luna), GPT-5.6, GPT-5.5 and earlier
  GPT-5 models, and Claude Fable 5.1, Fable 5, Opus 5.5, Opus 5, Opus 4.8,
  Sonnet 5, Sonnet 4.6 and Haiku 4.5, each with its official source URL and
  check date (all 2026-09-27). Callers override or extend it with
  `PricingTable.with_entries` or `PricingTable.from_dict`. Holding prices in
  agentshim is a user-directed decision; see `docs/pricing.md`.
- Recorded Codex, Claude (5m and 1h cache writes), opencode and Gemini
  streams under `tests/fixtures/`, with tests pinning each provider's mapping.
  The Claude Haiku 4.5 recording's `total_cost_usd` is reproduced exactly by
  the table.

### Changed

- `cached_input_tokens` is a deprecated alias of `cache_read_input_tokens`.
  On Claude, opencode and Copilot it used to be reads plus writes; it is now
  reads only, since a cache write is billed above base input and a read far
  below it. Codex and Gemini report no writes in their cached count, so their
  value is unchanged. Constructing with `cached_input_tokens=` still works.

### Documented

- Claude's `result.usage` covers the main conversation only; a Task
  subagent's tokens appear only in `modelUsage` and `total_cost_usd`.

## 0.6.8 (2026-09-27)

A tightening of the Codex (`STRICT`) schema check: schemas it now rejects
were already rejected by the API, after a full model turn.

### Fixed

- A Codex output schema is checked against OpenAI's strict structured-output
  rules before the CLI starts. `turn()` raises `SchemaDialectError`, naming
  each offending node by JSON pointer, when an object leaves a property out
  of `required` (declare it required and nullable instead), lists a
  `required` name missing from `properties`, omits `properties` (`{}` is
  fine) or `additionalProperties: false` (including on a nullable
  `["object", "null"]` object), uses `anyOf` at
  the root, or has a `$ref` that resolves to nothing. Codex used to fail such
  a turn with `invalid_json_schema` only after the model ran. The Claude
  (`OPEN`) dialect is unchanged.
- A `$ref` of `"#"` (root recursion) is accepted as local in both dialects.
- Problem pointers escape `/` and `~` in property names (RFC 6901), and
  `enum`, `const` and `required` values are no longer inspected as schemas.
- `normalize` closes a nullable object node too, and gives an object with
  no `properties` an empty one, so its output passes the `STRICT` check.

## 0.6.7 (2026-09-27)

Additive, with one tightening: a sandboxed Codex turn no longer applies the
user's exec-policy rules (see Fixed).

### Added

- `CodexSandboxConfig(excluded_commands=[...])` lets named commands run
  outside Codex's sandbox while everything else stays confined, the Codex
  counterpart of Claude's `excludedCommands`. Each entry is shell words
  matched as a command prefix. Codex supports this only as exec-policy
  `prefix_rule(decision="allow")` rules read from `$CODEX_HOME/rules/`, so
  `agentshim.providers.codex.install_rules(home, config)` writes them into a
  dedicated Codex home and the turn runs with `CODEX_HOME` set to it. The
  provider refuses a missing or relative `CODEX_HOME`, or one the sandbox lets
  commands write. `render_rules` and `parse_rules` expose the rules file.
- `ArgvContext.cwd` (default `None`): the turn's cwd, for a provider to
  validate paths against. It is never rendered into argv.

### Fixed

- Sandboxed Codex turns pass `--ignore-rules`. An `allow` rule in the user's
  `~/.codex/rules/` (which the TUI writes when a command is approved) used to
  run its command outside a sandbox agentshim had asked for.

### Documented

- Codex often omits commands its sandbox denied from `--json` output, and
  exposes them nowhere else in structured form, so no `ToolCall` event is
  emitted for them.

## 0.6.6 (2026-09-27)

Additive: every new option defaults to earlier behaviour.

### Added

- `CodexProvider(sandbox=CodexSandboxConfig(...))` keeps Codex's own OS
  sandbox on instead of bypassing it. `mode` is `read-only`,
  `workspace-write` (default) or `danger-full-access`; `workspace-write` also
  takes absolute `writable_roots`, `network_access` and `writable_tmp`. The
  config is rendered as `--config` overrides so resumed turns keep it, pins
  every `workspace-write` key so user config cannot widen it, and pins
  `approval_policy="never"`. Invalid combinations raise on construction.
  Without a config the argv is unchanged.
- `agentshim.providers.codex.parse_sandbox(argv)` inverts that rendering.
- `ClaudeProvider(hooks=[ClaudeHook(event, command, matcher, timeout_s)])`
  adds caller hooks to the turn's inline settings, after agentshim's own
  read-confinement hook. `command` is an argv quoted with `shlex.join`. Hooks
  work with or without a sandbox. `build_settings` now takes
  `SandboxConfig | None` and a keyword `hooks`; existing calls are unchanged.
- Hypothesis property tests, with a `fuzz` profile (`HYPOTHESIS_PROFILE=fuzz`).
- Cheap-model e2e knobs `AGENTSHIM_E2E_CLAUDE_MODEL` and
  `AGENTSHIM_E2E_CODEX_MODEL`, and a credential-free Codex sandbox
  enforcement matrix that runs whenever `codex` is installed.

### Fixed

- Codex `--config` string values escape control characters. A newline or
  other control character in an MCP command, argument, env value or `PATH`
  used to produce invalid TOML, which Codex silently keeps as a raw string.

## 0.6.5 (2026-09-27)

- Mark invocation-scoped Codex MCP servers as required so slow servers remain
  in the initial tool catalog and honor their configured startup timeout.

## 0.6.4 (2026-09-27)

- Add optional positive finite `startup_timeout_s` to stdio and HTTP MCP
  server specs. Codex renders it as an invocation-scoped
  `startup_timeout_sec` override, so slow-starting servers can initialize
  without modifying the user's config file.

## 0.6.3 (2026-09-26)

- Narrow Codex `auth_files` to `.codex/auth.json`. User `config.toml` is not
  authentication state and need not be copied or mounted into an isolated
  agent process; per-turn settings continue to use CLI config overrides.

## 0.6.2 (2026-09-26)

- Add optional positive finite `tool_timeout_s` to stdio and HTTP MCP server
  specs. Codex renders it as an invocation-scoped `tool_timeout_sec` config
  override; providers that cannot represent it reject the request explicitly.

## 0.6.1 (2026-09-10)

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

### Added

- `agentshim.testing.scripted_resume_failure(provider, *, session_id=None)`:
  a `FakeRun` that fails the way *provider* recognises a lost resumed
  conversation, so a consumer can test `SessionResumeError` handling without
  knowing any provider's wire format. Backed by a required
  `resume_failure_lines(...)` entry in every `providers/<name>/scripted.py`,
  found through `providers.get_resume_failure_lines(name)` the same way
  `scripted_turn` finds `scripted_lines`, so a new provider cannot ship
  without one.
- `agentshim.testing.installed_mcp_servers(provider, request, workspace)`:
  the MCP servers one turn installed, keyed by server name, each normalized
  to `command`/`args`/`env` (stdio) or `url`/`transport` (HTTP) regardless of
  how the provider itself renders it. Reads the config file for a
  CONFIG_FILE provider and parses `request.argv` for a CLI_FLAGS one, via a
  `parse_mcp_servers` kept next to the renderer it inverts
  (`providers/codex/provider.py`, `providers/copilot/provider.py`). Meant to
  be called from inside a `FakeExecutor` `run` callback: a CONFIG_FILE
  provider's config exists only for the lifetime of the turn.

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
