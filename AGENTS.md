# agentshim

A Python library that drives agent CLIs (Claude Code, Codex) through one typed
turn API. See [docs/development.md](docs/development.md) for quality gates,
property tests and the release checklist.

## Agent conventions

- Cap fuzz and property-test runs at 5 minutes of wall-clock time
  (`HYPOTHESIS_PROFILE=fuzz timeout 300 uv run pytest tests/unit -q`). Run
  longer only when the user explicitly asks for a long run.
- The e2e suite spends real model tokens; run it with cheap models
  (`haiku`, `gpt-6-luna`) as described in the release checklist.
