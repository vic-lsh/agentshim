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

## Token usage

Every provider normalizes its counts into `TokenUsage`, and
`cached_input_tokens <= input_tokens` holds on all of them. One gap:
**Copilot CLI reports no token counts.** Verified on 1.0.83, a run prints no
`assistant.usage` frame and no per-message `outputTokens`, so
`TurnResult.usage.tokens` is all zeros. Its `session.usage_checkpoint` frame
carries premium-request and AI-unit billing counters plus prompt-cache
diagnostics, none of which is the turn's billed token usage, so the parser
does not read them into `TokenUsage`.

All five providers ship on the same protocol.
