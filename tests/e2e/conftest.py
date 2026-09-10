"""End-to-end tests run the real provider CLIs.

They are skipped unless ``AGENTSHIM_E2E=1`` and the binary is on PATH, so a
normal ``pytest`` run needs no credentials and makes no network calls.

Two providers take their model from the environment, because the CLI default
is not usable on every account:

- ``AGENTSHIM_E2E_GEMINI_MODEL`` picks a Gemini model the account is
  entitled to. Without it the CLI's own default is used, which fails with
  ``ModelNotFoundError`` on an account that has no access to it.
- ``AGENTSHIM_E2E_OPENCODE_MODEL`` picks an opencode ``provider/model``.
  Without it the model from the user's own opencode config is used.

Both are optional; unset means "let the CLI choose".
"""

from __future__ import annotations

import os
import shutil

import pytest

#: Set by an enclosing Claude Code session. The CLI under test must not
#: inherit it: a nested run is a different code path from the one these
#: tests cover, so it is dropped before any environment is captured.
_NESTED_CLAUDE_VAR = "CLAUDECODE"

GEMINI_MODEL_VAR = "AGENTSHIM_E2E_GEMINI_MODEL"
OPENCODE_MODEL_VAR = "AGENTSHIM_E2E_OPENCODE_MODEL"


def requires_cli(binary: str) -> pytest.MarkDecorator:
    """Skip the test unless e2e is enabled and *binary* is installed."""
    enabled = os.environ.get("AGENTSHIM_E2E") == "1"
    reason = "set AGENTSHIM_E2E=1" if not enabled else f"{binary} is not on PATH"
    return pytest.mark.skipif(not enabled or shutil.which(binary) is None, reason=reason)


def model_from_env(variable: str) -> str | None:
    """Return the model named by *variable*, or ``None`` for the CLI default."""
    return os.environ.get(variable) or None


@pytest.fixture(scope="session", autouse=True)
def _drop_nested_claude_marker() -> None:
    """Unset ``CLAUDECODE`` for the whole e2e session.

    ``interactive_env()`` captures the environment through ``bash -i``, which
    inherits this process's variables, so the marker has to leave
    ``os.environ`` before the first ``CliAgent`` is built. It is removed once
    per session rather than per test because that capture is cached.
    """
    os.environ.pop(_NESTED_CLAUDE_VAR, None)
