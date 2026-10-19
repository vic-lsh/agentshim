"""A provider's login command is declared data: how a person signs its CLI in."""

from __future__ import annotations

import pytest
from agentshim import LoginSpec, get_provider, provider_names
from hypothesis import given
from hypothesis import strategies as st

_ARGS = st.lists(st.text(min_size=1, max_size=8), min_size=1, max_size=4).map(tuple)

#: Providers whose CLI has no sign-in command of its own: Gemini CLI signs in on its first
#: interactive run, so there is nothing to start.
_NO_LOGIN = {"gemini"}


@pytest.mark.parametrize("provider", provider_names())
def test_a_provider_declares_a_login_command_unless_its_cli_has_none(provider: str) -> None:
    profile = get_provider(provider).profile
    assert (profile.login is None) == (provider in _NO_LOGIN)


@pytest.mark.parametrize(
    ("provider", "command", "device"),
    [
        ("claude", ("claude", "auth", "login"), None),
        ("codex", ("codex", "login"), ("codex", "login", "--device-auth")),
        ("copilot", ("copilot", "login"), None),
        ("opencode", ("opencode", "auth", "login"), None),
    ],
)
def test_the_commands_are_each_clis_own(
    provider: str, command: tuple[str, ...], device: tuple[str, ...] | None
) -> None:
    profile = get_provider(provider).profile
    assert profile.login is not None
    assert profile.login.command(profile.binary) == command
    if device is None:
        with pytest.raises(ValueError, match="no device login"):
            profile.login.command(profile.binary, device=True)
    else:
        assert profile.login.command(profile.binary, device=True) == device


@given(binary=st.text(min_size=1, max_size=8), argv=_ARGS, device=st.none() | _ARGS)
def test_a_command_is_the_binary_then_its_arguments(
    binary: str, argv: tuple[str, ...], device: tuple[str, ...] | None
) -> None:
    spec = LoginSpec(argv=argv, device_argv=device)
    assert spec.command(binary) == (binary, *argv)
    if device is not None:
        assert spec.command(binary, device=True) == (binary, *device)


@pytest.mark.parametrize(("argv", "device"), [((), None), (("login",), ()), ((), ("login",))])
def test_an_empty_command_is_rejected(
    argv: tuple[str, ...], device: tuple[str, ...] | None
) -> None:
    with pytest.raises(ValueError, match="at least one argument"):
        LoginSpec(argv=argv, device_argv=device)
