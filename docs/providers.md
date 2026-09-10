# Providers

`get_provider(name)` resolves a name to a provider with its default options;
`provider_names()` lists what is available.

```python
from agentshim import CliAgent, get_provider, provider_names

print(provider_names())
agent = CliAgent("claude")
```

Pass a provider instance instead of a name when you need a provider-specific
option:

```python
from agentshim import CliAgent, interactive_env
from agentshim.providers.claude import ClaudeProvider, SandboxConfig

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
profile.auth_env_vars              # credential variables to forward
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

All five providers ship on the same protocol.
