"""``probe_provider`` against each installed CLI.

No model is called and no tokens are spent: the probe runs ``--version`` and,
where the CLI has one, its own authentication status command. Skipped unless
``AGENTSHIM_E2E=1`` and the binary is on PATH.
"""

from __future__ import annotations

import re
import subprocess
from typing import TYPE_CHECKING

import pytest
from agentshim import AuthState, interactive_env, probe_provider

from tests.e2e.conftest import requires_cli

if TYPE_CHECKING:
    from pathlib import Path

#: provider name -> (binary, has an authentication status command)
PROVIDERS = {
    "claude": ("claude", True),
    "codex": ("codex", True),
    "gemini": ("gemini", False),
    "copilot": ("copilot", False),
    "opencode": ("opencode", False),
}

#: Variables that authenticate a CLI without a stored login.
_CREDENTIAL_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "OPENAI_API_KEY",
    "CODEX_API_KEY",
)


def _cases() -> list[object]:
    return [
        pytest.param(name, has_status, marks=[pytest.mark.e2e, requires_cli(binary)])
        for name, (binary, has_status) in PROVIDERS.items()
    ]


@pytest.mark.parametrize(("provider", "has_status"), _cases())
def test_the_installed_cli_reports_its_version_and_authentication(
    provider: str, *, has_status: bool
) -> None:
    status = probe_provider(provider)
    assert status.binary_found
    assert status.path
    assert status.version is not None
    assert re.fullmatch(r"\d+(\.\d+)+(-[0-9A-Za-z.]+)?", status.version)
    assert status.auth_detail
    if has_status:
        # The account's state is not ours to know, but the CLI can always say.
        assert status.auth in (AuthState.KNOWN_OK, AuthState.FAILED)
    else:
        assert status.auth is AuthState.UNKNOWN


@pytest.mark.parametrize(
    "provider",
    [pytest.param(p, marks=[pytest.mark.e2e, requires_cli(p)]) for p in ("claude", "codex")],
)
def test_a_cli_with_no_stored_login_is_failed_not_unknown(provider: str, tmp_path: Path) -> None:
    env = {k: v for k, v in interactive_env().items() if k not in _CREDENTIAL_VARS}
    env["CLAUDE_CONFIG_DIR" if provider == "claude" else "CODEX_HOME"] = str(tmp_path)
    status = probe_provider(provider, env=env)
    assert status.auth is AuthState.FAILED
    assert f"{provider}" in status.auth_detail


@pytest.mark.e2e
@requires_cli("claude")
def test_the_probe_agrees_with_the_cli_it_wraps() -> None:
    status = probe_provider("claude")
    printed = subprocess.run(  # noqa: S603 - the installed CLI, fixed arguments
        [status.path or "claude", "--version"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert status.version is not None
    assert status.version in printed
