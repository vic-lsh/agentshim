# agentshim

`agentshim` is a small Python package that wraps coding-agent CLIs behind a
stable interface for higher-level runtimes.

It currently includes:

- a `CodingAgent` base class and provider registry
- CLI adapters for Claude Code, Codex, Gemini, and Opencode
- MCP server config dataclasses
- sandbox settings helpers for Claude Code
- a lightweight litellm client and isolated subagent helper
- a trajectory protocol with a no-op implementation

## Install

```bash
pip install agentshim
```

For development:

```bash
uv sync --dev
uv run pytest
```

## Quick Example

```python
from agentshim.claude import ClaudeCodeCodingAgent

agent = ClaudeCodeCodingAgent(model="sonnet")
result = agent.generate("Summarize the repository layout.", cwd=".")
print(result)
```

## Publishing

Build locally with:

```bash
uv build
```

Publish with:

```bash
uv publish
```
