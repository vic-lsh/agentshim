#!/bin/bash
# Run static type checking with pyright.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Running type checks with pyright..."
cd "$PROJECT_ROOT"

if command -v uv >/dev/null 2>&1; then
    uv run pyright
else
    echo "Error: 'uv' is not installed. Please install uv to run this script."
    exit 1
fi
