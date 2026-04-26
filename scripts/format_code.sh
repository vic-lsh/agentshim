#!/bin/bash
# Format Python code using ruff format.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

if [ "$1" == "--check" ]; then
    echo "Checking code formatting (dry-run)..."
    uv run ruff format --check .
else
    echo "Formatting code..."
    uv run ruff format .
fi

echo "Formatting check complete."
