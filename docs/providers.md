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
