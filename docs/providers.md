# Providers

`get_provider(name)` resolves a name to a provider with its default options;
`provider_names()` lists what is available.

```python
from agentshim import CliAgent, get_provider, provider_names

print(provider_names())
# ['claude', 'codex', 'copilot', 'gemini', 'opencode']
agent = CliAgent("claude")
```

`model` is an opaque provider-specific string. agentshim passes it to the CLI
unchanged and never validates it, so `"sonnet"` for Claude Code and
`"anthropic/claude-sonnet-4-5"` for opencode are both just strings. `None`
leaves the CLI's own default.

Pass a provider instance instead of a name when you need a provider-specific
option:

```python
from agentshim import ClaudeProvider, CliAgent, SandboxConfig, interactive_env

provider = ClaudeProvider(sandbox=SandboxConfig(allowed_domains=["github.com"]))
agent = CliAgent(provider, env={**interactive_env(), **provider.sandbox_env})
```

## Sandboxes

The two sandboxing providers confine different things, so each has its own
option rather than a shared abstraction.

**Codex** sandboxes every command the model runs at the OS level. By default
agentshim bypasses it (`--dangerously-bypass-approvals-and-sandbox`), for a
caller that already isolates the whole process. `CodexSandboxConfig` keeps it
on:

```python
from agentshim import CliAgent, CodexProvider, CodexSandboxConfig

# Read the repository, write nothing, no network.
reviewer = CliAgent(CodexProvider(sandbox=CodexSandboxConfig(mode="read-only")))

# Write the workspace and one cache directory; reach a local Docker socket.
builder = CliAgent(
    CodexProvider(
        sandbox=CodexSandboxConfig(
            mode="workspace-write",
            writable_roots=["/home/me/.cache/go-build"],
            network_access=True,
            writable_tmp=False,
        )
    )
)
```

| mode | workspace | `writable_roots` | `/tmp`, `$TMPDIR` | network |
|---|---|---|---|---|
| `read-only` | read | n/a | read | no |
| `workspace-write` | write | write | write unless `writable_tmp=False` | only with `network_access=True` |
| `danger-full-access` | write | n/a | write | yes |

Every setting is passed as a `--config` override, which `codex exec resume`
accepts where it rejects `--sandbox`, so a resumed turn keeps the sandbox.
`workspace-write` renders every key even at its default, so the user's
`~/.codex/config.toml` cannot widen it, and every sandboxed turn pins
`approval_policy="never"` so a refused command fails instead of waiting for
an approval no one will give. The config is validated on construction:
`writable_roots` must be absolute, and options that only apply to
`workspace-write` are rejected on the other modes rather than ignored.
`parse_sandbox(argv)` in `agentshim.providers.codex` recovers the config a
turn ran with, for tests.

A sandboxed turn also passes `--ignore-rules`. Codex runs any command that an
exec-policy `prefix_rule(..., decision="allow")` matches outside its sandbox,
and without the flag it would load such rules from the user's
`~/.codex/rules/` (where approving a command in the TUI saves them) and from
trusted projects.

### Exempting one command

Some commands need access the sandbox withholds from everything else, such
as a gateway that talks to the Docker socket. `excluded_commands` is the
Codex counterpart of Claude Code's `excludedCommands`:

```python
from pathlib import Path

from agentshim import CliAgent, CodexProvider, CodexSandboxConfig, interactive_env
from agentshim.providers.codex import install_rules

sandbox = CodexSandboxConfig(
    mode="workspace-write",
    excluded_commands=["sdo detector check"],
)
home = Path("/var/lib/myapp/codex-home")  # dedicated; not in the workspace or /tmp
install_rules(home, sandbox)              # writes home/rules/agentshim.rules
# Codex also needs credentials in that home: copy auth.json (see
# PROFILE.auth_files) or pass OPENAI_API_KEY.
agent = CliAgent(
    CodexProvider(sandbox=sandbox),
    env={**interactive_env(), "CODEX_HOME": str(home)},
)
```

Each entry is shell words. A command the model runs is exempt when its words
start with an entry's words: `sdo detector check --all` is exempt,
`sdo detector` and `/usr/bin/sdo detector check` are not. Codex matches the
command line the model wrote, and anything else on that line (`&&`, `|`,
`;`, `$(...)`) keeps the whole line in the sandbox, exempt part included.
Every other command stays confined exactly as the rest of the config says.
Exemptions apply to `read-only` and `workspace-write`; `danger-full-access`
rejects them.

How it works, and what it costs:

- Codex has no per-command sandbox setting and no `--config` key for rules:
  exemptions exist only as exec-policy `allow` rules in `.rules` files, and
  an `allow` rule runs the command with **no sandbox at all**. An exempt
  command can write anywhere and reach any socket or host the user can, and
  the model chooses its remaining arguments. Exempt only commands that are
  safe with any arguments. Codex offers no narrower mechanism: network access
  (`network_access`) and write access (`writable_roots`) apply to every
  command in the session.
- The rules come from `$CODEX_HOME/rules/`, so the turn needs a dedicated
  Codex home. `install_rules` writes `rules/agentshim.rules` atomically and
  refuses a home whose `rules/` holds any other `.rules` file. Every Codex
  process that uses the home and does not pass `--ignore-rules` applies the
  exemption, and resumed turns must use the same home, since it also holds
  the session rollouts.
- The provider refuses the turn (`ProviderCapabilityError`) unless
  `CODEX_HOME` is set, absolute, and outside every directory the sandbox lets
  commands write: the turn's `cwd`, `writable_roots`, and `/tmp` and
  `$TMPDIR` when `writable_tmp` is on. A command able to write the home could
  add a rule exempting itself.
- Keep the home out of the system temp dir even with `writable_tmp=False`:
  Codex refuses to create its sandbox helper under it, and then every
  sandboxed command fails to start.
- Trusted projects' `.codex/rules/` still load. A fresh home trusts no
  project; do not mark the workspace trusted in its `config.toml`.

### Denied commands in the event stream

Codex 0.157 does not always report a command its sandbox refused. When the
sandbox denies a write or a socket, `codex exec --json` often emits no
`command_execution` item for it, so no `ToolCall` or `ToolResult` event
reaches agentshim; a command that merely fails for another reason is
reported. Nothing else carries the denial in structured form: stderr is
silent, and the session rollout under `$CODEX_HOME/sessions/` holds only the
model's own tool call (for current models, JavaScript source passed to a
code-mode `exec` tool) and its free-text output, with no command record and
no denial flag. agentshim therefore cannot surface these attempts. Do not
treat the absence of a `ToolCall` as proof that a command was not attempted;
to audit, check the effect (the file or socket the command would have
touched), as the e2e suite does.

**Claude Code** sandboxes only the Bash tool's subprocesses; see
`SandboxConfig` above.

## Claude Code hooks

A policy the sandbox cannot express, such as refusing one executable in
Bash, goes in a hook. `ClaudeProvider(hooks=...)` adds them to the settings
agentshim already passes with `--settings`, after its own read-confinement
hook, so neither replaces the other. Passing a second `--settings` through
`extra_args` would compete with agentshim's.

```python
import sys

from agentshim import ClaudeHook, ClaudeProvider, CliAgent

deny_go = ClaudeHook(
    event="PreToolUse",
    matcher="Bash",
    command=[sys.executable, "/opt/hooks/deny_executables.py", "go"],
    timeout_s=10,
)
agent = CliAgent(ClaudeProvider(hooks=[deny_go]))
```

`command` is an argv, not a shell string. Claude Code runs hooks through a
shell, and agentshim quotes each element with `shlex.join`, so every
argument arrives literally. Use an absolute executable: the agent's PATH is
not yours. Invalid hooks (a non-PascalCase event, an empty or bare-string
command, a NUL byte, a non-positive timeout) raise on construction. Hooks
apply to every turn, including resumed ones.

## Capabilities

Everything a caller needs to know before running a turn is declared on
`agent.profile`, so nothing has to probe a provider object:

```python
profile = agent.profile
profile.supports_resume            # bool
profile.supports_reasoning_effort  # bool
profile.mcp                        # McpMechanism
profile.output_schema              # OutputSchemaStyle
profile.schema_dialect             # SchemaDialect | None
profile.state_dirs                 # home-relative provider state
profile.darwin_state_dirs          # extra state dirs on macOS only
profile.auth_env_vars              # credential variables to forward
profile.skill_dirs                 # workspace-relative skill discovery dirs
profile.skill_discovery            # SkillSignal: does the stream list offered skills?
profile.skill_invocation           # SkillSignal: does the stream reveal skill loads?
profile.skill_scopes               # frozenset[SkillScope] a session may request
profile.mcp_scopes                 # frozenset[McpScope] a session may request
profile.config_scopes              # frozenset[ConfigScope] a session may request
profile.config_home_files          # state-root files a ConfigScope.PROJECT home carries over
profile.container_install          # shell commands installing the CLI
```

Asking for something a provider cannot do raises `ProviderCapabilityError`
before the process starts.

## Per-provider behaviour

| | claude | codex | gemini | opencode | copilot |
|---|---|---|---|---|---|
| resume | `--resume <id>` | `exec resume <id>` | `--resume <id>` | `run --session <id>` | `--resume <id>` |
| MCP | `.mcp.json` | `--config mcp_servers.*` | `.gemini/settings.json` | `opencode.json` | `--additional-mcp-config` |
| output schema | `--json-schema`, OPEN | `--output-schema`, STRICT | none | none | none |
| reasoning effort | `--effort` | `--config model_reasoning_effort` | none | none | none |
| stream | `stream-json` | `--json` | `stream-json` | `run --format json` | `--output-format json` |
| token usage | yes | yes | yes | yes | no, see below |
| cost | yes | no | no | yes | no |
| skills offered | `system/init` `skills` (STRUCTURED) | unknown | unknown | unknown | unknown |
| skill loads | `Skill` tool call, or `Read` of a `SKILL.md` (STRUCTURED) | shell read of a `SKILL.md` (INFERRED) | unknown | unknown | unknown |
| `SkillScope.PROJECT` | `--setting-sources project,local` | `features.plugins=false`, user skills off in `skills.config` | refused | refused | refused |
| `McpScope.SESSION` | `--strict-mcp-config` and the turn's servers as inline `--mcp-config` | `enabled=false` on configured servers, `features.plugins=false`, `features.apps=false` | refused | refused | refused |
| `ConfigScope.PROJECT` | `--setting-sources project,local`, `autoMemoryEnabled=false` | dedicated `CODEX_HOME` from `prepare_config_home`, `--ignore-user-config --disable memories` | refused | refused | refused |

`start_session(skill_scope=SkillScope.PROJECT)` (also on `run`) offers the
agent only the workspace's skills (`profile.skill_dirs`) and the CLI's
built-in ones, not the user's personal skills or installed plugins, so who
runs agentshim does not change what the agent sees. A provider that cannot
enforce it raises `ProviderCapabilityError` from `start_session`. Side
effects beyond skills:

- Claude Code: the `user` settings source is skipped, so `~/.claude/skills`,
  user plugins (their skills, hooks and MCP servers), `~/.claude/CLAUDE.md`
  and user settings (default model, permissions, hooks, `env`,
  `apiKeyHelper`) are not loaded. Credentials (`.credentials.json`,
  `auth_env_vars`) still work; auth configured only through user-settings
  `env` or `apiKeyHelper` does not, so pass it as environment instead.
  Workspace settings, `--settings`, managed policy and claude.ai account
  connectors still apply.
- Codex: plugins are switched off as a feature, and every `SKILL.md` under
  `$CODEX_HOME/skills` (except Codex's bundled `.system`) and
  `~/.agents/skills` is disabled by path; the scan runs on the host
  agentshim runs on. `config.toml`, auth and session storage are untouched,
  so resume works as before.

`start_session(mcp_scope=McpScope.SESSION)` (also on `run`) connects the CLI
only to the servers in `TurnRequest.mcp_servers`; a turn given none sees no
MCP server. It is independent of `skill_scope`. A provider that cannot
enforce it raises `ProviderCapabilityError` from `start_session`.

- Claude Code: `--strict-mcp-config` makes `--mcp-config` the only source, so
  user and project (`.mcp.json`) servers, plugin servers and claude.ai
  account connectors are all dropped. The turn's servers travel inline in
  argv (as Codex's do), not through the workspace `.mcp.json`.
- Codex: every `mcp_servers` entry in `$CODEX_HOME/config.toml`,
  `/etc/codex/config.toml` and `<cwd>/.codex/config.toml` that the turn was
  not given is disabled by name, and plugins and apps (the account's ChatGPT
  connectors) are switched off as features. The scan runs where agentshim
  runs. Give session servers names the user's config does not use: Codex
  merges same-named entries field by field.

`start_session(config_scope=ConfigScope.PROJECT)` (also on `run`) loads none
of the user's own CLI configuration: settings, hooks, global instructions,
notify commands, profiles and memory. The workspace's own instructions and
settings, the turn's flags and credentials keep working. It is independent of
the skill and MCP scopes. Verified against Claude Code 2.1.288 and Codex
0.156.1 by seeding each source with a code word or a hook that touches a file.

| Source | Claude Code | Codex |
|---|---|---|
| user hooks | `--setting-sources` (`settings.json` hooks) | dedicated home (`hooks.json`) |
| global instructions | `--setting-sources` (`~/.claude/CLAUDE.md`) | dedicated home (`AGENTS.md`) |
| user settings, `env`, profiles, notify | `--setting-sources` | `--ignore-user-config`, dedicated home |
| memory | `autoMemoryEnabled=false` | dedicated home, `--disable memories` |
| user skills, plugins | `--setting-sources` | dedicated home |
| exec-policy rules | n/a | dedicated home |

- Claude Code isolates by flags alone and needs no home: `prepare_config_home`
  returns `{}` for it. Built-in skills stay.
- Codex reads `AGENTS.md`, `hooks.json` and `memories/` from `$CODEX_HOME`
  with no flag to skip them, so the scope needs a state root of its own.
  `prepare_config_home(profile, home, env)` links `profile.config_home_files`
  (`auth.json`) from the state root `env` selects into `home` as symlinks and
  returns `{"CODEX_HOME": home}` to merge into the agent's environment. A link,
  not a copy: Codex rotates the OAuth refresh token on every refresh and
  rejects a reused one, and it saves `auth.json` by truncating and writing in
  place, which follows the link. Both homes therefore always hold the current
  login, whichever refreshed it. A sandbox around the CLI must expose the
  link's target read-write. The home also keeps
  Codex's conversations, so reuse it for every session that must resume
  another. `build_argv` refuses the scope when `CODEX_HOME` is unset,
  relative, or the user's own `~/.codex`. `--ignore-user-config` also means a
  workspace `.codex/config.toml` is not loaded; pass settings as turn flags.

The prompt is never in argv on any provider: it always goes on stdin, so an
agent's own `pkill -f` cannot match the CLI by prompt text.

## Credentials

`profile.auth_env_vars` names the variables a provider reads, so a caller
forwarding credentials does not hardcode provider names:

| provider | variables |
|---|---|
| claude | `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, `ANTHROPIC_CUSTOM_HEADERS` |
| codex | `OPENAI_API_KEY`, `OPENAI_BASE_URL` |
| copilot | `COPILOT_GITHUB_TOKEN`, `GH_TOKEN`, `GITHUB_TOKEN` |
| gemini | `GEMINI_API_KEY`, `GOOGLE_API_KEY` |
| opencode | none: it authenticates through `opencode auth` |

`CliAgent(env=...)` replaces the environment rather than extending it. The
default is `interactive_env()`, which captures what a login shell would give,
because provider CLIs are usually installed by a shell rc file that a
non-interactive process never sources. Extend it explicitly:

```python
from agentshim import CliAgent, interactive_env

agent = CliAgent("claude", env={**interactive_env(), "ANTHROPIC_API_KEY": "..."})
```

## Readiness probe

`probe_provider(name)` (or `Agent.probe()`) reports whether a provider could
start a turn, without running one and without calling a model. It returns a
`ProviderStatus`: `binary_found`, `path`, `version` (from `--version`), and
`auth` as an `AuthState` with an `auth_detail` that names the fix.

```python
from agentshim import AuthState, probe_provider

status = probe_provider("codex")          # local host; pass executor=/confinement= to probe elsewhere
if not status.binary_found:
    raise SystemExit(status.auth_detail)  # "codex was not found on PATH"
if status.auth is AuthState.FAILED:
    raise SystemExit(status.auth_detail)  # "codex is not logged in; run `codex login`"
```

`UNKNOWN` is not `FAILED`: it means the CLI has no cheap way to tell, or the
check could not run, and a caller must not stop on it. A missing binary is a
result, not an exception, so one pass can report every provider. The commands
run through the executor and environment a turn would use, so a confinement or
container is probed where the agent would run. `Agent.probe()` needs an agent
built from a provider name (constructing it already requires the binary).

| provider | version | authentication mechanism | states |
|---|---|---|---|
| claude | `claude --version` | `claude auth status` (JSON `loggedIn`; also honours `ANTHROPIC_API_KEY` and OAuth-token variables) | `KNOWN_OK`, `FAILED`, `UNKNOWN` for unrecognised output |
| codex | `codex --version` | `codex login status` (`Logged in ...` / `Not logged in`); with `CODEX_API_KEY` or `OPENAI_API_KEY` set, "not logged in" is `UNKNOWN` because the command ignores them | `KNOWN_OK`, `FAILED`, `UNKNOWN` |
| gemini | `gemini --version` | none | always `UNKNOWN` |
| copilot | `copilot --version` | none (`copilot login` is interactive only) | always `UNKNOWN` |
| opencode | `opencode --version` | none that proves a usable model (`opencode auth list` shows stored keys only) | always `UNKNOWN` |

`agentshim.testing.probe_executor(provider, version=..., auth=..., installed=...)`
is a `FakeExecutor` that answers these commands the way each CLI does.

## Token usage

Every provider normalizes its counts into `TokenUsage`; see
[Usage and Pricing](pricing.md) for the breakdown and how each CLI's fields
map onto it.

**Codex exec JSON reports cumulative thread totals.** On 0.144.4 and 0.160.0,
`turn.completed.usage` copies `input_tokens`, `cached_input_tokens`,
`cache_write_input_tokens`, `output_tokens` and `reasoning_output_tokens` from
the thread's `total`, including on `codex exec resume`. There is no
per-invocation usage field in this stream. See the
[Codex JSON event processor](https://github.com/openai/codex/blob/rust-v0.160.0/codex-rs/exec/src/event_processor_with_jsonl_output.rs).

agentshim subtracts the previous raw thread report before normalizing these
counts. `TurnResult.usage.tokens` and `UsageReport` contain the invocation's
increment; `ProviderUsage.raw` and the completion lifecycle detail retain the
CLI's cumulative values. A fresh conversation starts from zero. Sessions
retain baselines by conversation id across `forget()` and `adopt()`.

When adopting a thread from outside this session, pass its last report:

```python
session = agent.start_session(session_id=saved_id, previous_usage=saved_usage)
# Or: session.adopt(saved_id, previous_usage=saved_usage)
```

Checkpoint `saved_usage.provider` and `saved_usage.raw` along with the id.
`ProviderUsage.to_dict()` omits `raw`, so its output alone is insufficient.
Reconstruct a checkpoint with `ProviderUsage(provider="codex", raw=saved_raw)`.
If a resumed invocation has no baseline, it returns its result normally with
`ProviderUsage.increment_known=False` on both `TurnResult.usage` and
`UsageReport`. Token counts are zero placeholders, while `tokens.turns` still
counts completion frames. Check `increment_known` before pricing or budgeting;
these placeholders do not mean the invocation was free. The cumulative total
remains in `raw`, never in the increment. The session retains that total as the
next invocation's baseline, so subsequent increments are known. The missing
increment cannot be recovered from that frame alone. CLI failures retain their
original classification. Changes to the same
thread outside the session must be accompanied by an updated baseline.
Missing reports cannot account for tokens consumed before a failed invocation
ends; those tokens appear in the next observed total. Inconsistent or decreasing
counts are clamped by `normalized_usage` after subtraction.

One other gap:
**Copilot CLI reports no token counts.** Verified on 1.0.83, a run prints no
`assistant.usage` frame and no per-message `outputTokens`, so
`TurnResult.usage.tokens` is all zeros. Its `session.usage_checkpoint` frame
carries premium-request and AI-unit billing counters plus prompt-cache
diagnostics, none of which is the turn's billed token usage, so the parser
does not read them into `TokenUsage`.

All five providers ship on the same protocol.
