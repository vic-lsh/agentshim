# Development

Install development dependencies and run the test suite with `uv`.

```bash
uv sync --dev
uv run pytest
```

## Quality Gates

CI runs these five checks, and each has a script so it can be run the same way
locally. A change is ready when all five pass.

| Gate | Command | Checks |
| --- | --- | --- |
| Format | `bash scripts/format_code.sh --check` | `ruff format` is a no-op (drop `--check` to reformat) |
| Lint | `bash scripts/check_errors.sh` | `ruff check` (pass `--fix` to autofix) |
| Types | `bash scripts/type_check.sh` | `pyright`, strict on `agentshim/` |
| Layers | `bash scripts/check_imports.sh` | `import-linter` layering contracts |
| Tests | `uv run pytest` | the unit suite; end-to-end tests need `AGENTSHIM_E2E=1` |

The layering contract is declared under `[tool.importlinter]` in
`pyproject.toml`. It fails the build when a module imports outward, for example
`core/` reaching into `providers/`. See
[architecture](architecture.md) for what each layer owns.

## Build Package

```bash
uv build
```

## Publish Package

```bash
uv publish
```

## Build Docs Locally

```bash
uv sync --group docs
uv run mkdocs build --strict
```
