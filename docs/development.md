# Development

Install development dependencies and run the test suite with `uv`.

```bash
uv sync --dev
uv run pytest
```

## Quality Gates

CI runs five checks. Four have a script so they can be run the same way
locally; the fifth is `pytest`. A change is ready when all five pass.

| Gate | Command | Checks |
| --- | --- | --- |
| Format | `bash scripts/format_code.sh --check` | `ruff format` is a no-op (drop `--check` to reformat) |
| Lint | `bash scripts/check_errors.sh` | `ruff check` (pass `--fix` to autofix) |
| Types | `bash scripts/type_check.sh` | `pyright`, strict on `agentshim/` |
| Layers | `bash scripts/check_imports.sh` | `import-linter` layering contracts |
| Tests | `uv run pytest` | the unit suite; the e2e suite is skipped by default |

The layering contract is declared under `[tool.importlinter]` in
`pyproject.toml`. It fails the build when a module imports outward, for example
`core/` reaching into `providers/`. See
[architecture](architecture.md) for what each layer owns.

## End-to-End Tests

`tests/e2e/` runs the real provider CLIs. Each suite is skipped unless
`AGENTSHIM_E2E=1` and its binary is on PATH, so a default `pytest` run needs
no credentials and makes no network calls. CI never sets the variable, so the
e2e suite is a local gate only. The `e2e` marker selects or excludes them:
`-m e2e`, `-m "not e2e"`.

```bash
AGENTSHIM_E2E=1 uv run pytest tests/e2e -q
```

Install and authenticate each CLI first. Two providers take a model from the
environment, because the CLI default is not usable on every account:

| Variable | Effect |
| --- | --- |
| `AGENTSHIM_E2E_GEMINI_MODEL` | Gemini model to use. Without it the CLI default is used, which fails with `ModelNotFoundError` on an account that has no access to it. |
| `AGENTSHIM_E2E_OPENCODE_MODEL` | opencode `provider/model`. Without it the model from your own opencode config is used. |

```bash
AGENTSHIM_E2E=1 AGENTSHIM_E2E_GEMINI_MODEL=gemini-2.5-flash \
  uv run pytest tests/e2e/test_gemini_e2e.py -q
```

The e2e conftest unsets `CLAUDECODE` for the session. `interactive_env()`
captures the environment through `bash -i`, which inherits it from an
enclosing Claude Code session, and a nested run is not the code path these
tests cover.

## Build Package

```bash
uv build
```

The wheel carries `agentshim/` whole: all five provider packages, the Claude
read-confinement hook under `providers/claude/hooks/`, and `py.typed`.

## Publish Package

```bash
uv publish
```

## Build Docs Locally

```bash
uv run --group docs mkdocs build --strict
```
