"""One real limits read per provider. No prompt and no model call; spending is never exercised.

Claude's read starts the CLI with no prompt. Set ``AGENTSHIM_E2E_CLAUDE_MODEL`` to pin the model the
CLI would use (it is passed as ``ANTHROPIC_MODEL``); no model call is made either way.
"""

from __future__ import annotations

import pytest
from agentshim import WindowKind, interactive_env, read_limits

from .conftest import CLAUDE_MODEL_VAR, model_from_env, requires_cli

pytestmark = pytest.mark.e2e


def _env(model: str | None) -> dict[str, str]:
    env = dict(interactive_env())
    if model is not None:
        env["ANTHROPIC_MODEL"] = model
    return env


def _check(provider: str, env: dict[str, str]) -> None:
    limits = read_limits(provider, env=env, timeout=45.0)
    assert limits.unavailable is None, limits.unavailable
    assert limits.windows
    assert limits.account is not None
    assert "@" not in limits.account
    ids = [w.id for w in limits.windows]
    assert len(ids) == len(set(ids))
    for window in limits.windows:
        assert window.kind in WindowKind
        assert window.used_fraction >= 0
        assert window.resets_at is None or window.resets_at.tzinfo is not None


@requires_cli("claude")
def test_claude_reads_its_limits() -> None:
    _check("claude", _env(model_from_env(CLAUDE_MODEL_VAR)))


@requires_cli("codex")
def test_codex_reads_its_limits() -> None:
    _check("codex", _env(None))
