"""``read_limits``, ``limit_window`` and ``use_reset_credit`` across providers."""

from __future__ import annotations

import agentshim
import pytest
from agentshim import (
    LimitsUnavailableReason,
    ProviderCapabilityError,
    RateLimitStatus,
    limit_providers,
    limit_window,
    provider_names,
    read_limits,
    use_reset_credit,
)
from agentshim.testing import FakeExecutor
from agentshim.testing.claude_stream import ClaudeStreamPeers
from agentshim.testing.clock import FakeClock
from agentshim.testing.confinement import FakeConfinement
from agentshim.testing.limits import claude_account, claude_usage
from agentshim.testing.process import SilentPeer

ENV = {"PATH": "/bin"}


def _silent() -> FakeExecutor:
    return FakeExecutor([], peers=lambda _request: SilentPeer())


def test_only_the_providers_that_can_say_are_readable() -> None:
    assert limit_providers() == ("claude", "codex")
    assert set(limit_providers()) < set(provider_names())


@pytest.mark.parametrize("provider", sorted(set(provider_names()) - {"claude", "codex"}))
def test_a_provider_with_no_read_says_unsupported_and_starts_nothing(provider: str) -> None:
    executor = _silent()
    limits = read_limits(provider, executor=executor, env=ENV, clock=FakeClock())
    assert limits.provider == provider
    assert limits.windows == ()
    assert limits.unavailable is not None
    assert limits.unavailable.reason is LimitsUnavailableReason.UNSUPPORTED
    assert executor.spawns == []


@pytest.mark.parametrize("provider", sorted(set(provider_names()) - {"codex"}))
def test_only_codex_has_credits_to_spend(provider: str) -> None:
    executor = _silent()
    with pytest.raises(ProviderCapabilityError):
        use_reset_credit(provider, idempotency_key="k", executor=executor, env=ENV)
    assert executor.spawns == []


@pytest.mark.parametrize("provider", sorted(set(provider_names()) - {"claude", "codex"}))
def test_a_provider_with_no_limits_has_no_window_for_a_status(provider: str) -> None:
    status = RateLimitStatus(window="primary", used_fraction=0.5, resets_at=None)
    assert limit_window(provider, status) is None


def test_an_unknown_provider_is_a_caller_error() -> None:
    with pytest.raises(ValueError, match="unknown provider"):
        read_limits("nope", executor=_silent(), env=ENV)
    with pytest.raises(ValueError, match="unknown provider"):
        use_reset_credit("nope", idempotency_key="k", executor=_silent(), env=ENV)
    with pytest.raises(ValueError, match="unknown provider"):
        limit_window("nope", RateLimitStatus(window="x", used_fraction=0.1, resets_at=None))


def test_a_timeout_must_be_positive() -> None:
    with pytest.raises(ValueError, match="timeout"):
        read_limits("claude", executor=_silent(), env=ENV, timeout=0)


def test_a_confinement_supplies_the_environment_so_env_may_not_also_be_given() -> None:
    with pytest.raises(ValueError, match="env conflicts with confinement"):
        read_limits("claude", executor=_silent(), confinement=FakeConfinement(), env=ENV)


def test_a_read_runs_where_a_turn_would_run_inside_the_confinement() -> None:
    inner = FakeExecutor(
        [], peers=ClaudeStreamPeers(account=claude_account(), usage=claude_usage()).build
    )
    confinement = FakeConfinement(env={"PATH": "/confined/bin", "HOME": "/confined"})
    limits = read_limits("claude", executor=inner, confinement=confinement, clock=FakeClock())
    assert limits.unavailable is None
    (spawn,) = inner.spawns
    assert spawn.argv[0] == "fake-confine"
    assert spawn.env["HOME"] == "/confined"


def test_the_limits_api_is_public() -> None:
    for name in (
        "AccountLimits",
        "LimitWindow",
        "WindowKind",
        "ResetCredit",
        "ResetCredits",
        "ResetCreditResult",
        "ResetOutcome",
        "LimitsUnavailable",
        "LimitsUnavailableReason",
        "LimitsReader",
        "LimitsSession",
        "LimitsSource",
        "account_digest",
        "limit_providers",
        "limit_window",
        "read_limits",
        "use_reset_credit",
    ):
        assert name in agentshim.__all__, name
