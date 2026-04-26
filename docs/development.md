# Development

Install development dependencies and run the test suite with `uv`.

```bash
uv sync --dev
uv run pytest
```

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
