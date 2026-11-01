"""Codex's limits read and reset: the requests it makes, what it makes of the answer."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pytest
from agentshim import (
    AccountLimits,
    LimitsUnavailableReason,
    ResetCreditResult,
    ResetOutcome,
    TurnFailedError,
    WindowKind,
    account_digest,
    limit_window,
    read_limits,
    use_reset_credit,
)
from agentshim.execution.process import ProcessExited, StderrLine, StdoutLine
from agentshim.providers.codex.app_server import ClientRequest
from agentshim.providers.codex.app_server.protocol import (
    GetAccountRateLimitsResponse,
    RateLimitSnapshot,
)
from agentshim.providers.codex.app_server.rate_limits import rate_limit_statuses
from agentshim.providers.codex.limits import parse_rate_limits
from agentshim.testing import FakeExecutor
from agentshim.testing.clock import FakeClock
from agentshim.testing.codex_app_server import CodexScript
from agentshim.testing.limits import CodexAccount, CodexWindow
from agentshim.testing.process import SilentPeer
from hypothesis import given
from hypothesis import strategies as st

FIXTURE = Path(__file__).parents[3] / "fixtures" / "codex_limits" / "rate_limits_read.json"
ENV = {"PATH": "/bin"}
CONSUME = "account/rateLimitResetCredit/consume"
READ = "account/rateLimits/read"


def _methods(script: CodexScript) -> list[str]:
    return [m.params.METHOD for m in script.received if isinstance(m, ClientRequest)]


def _read(script: CodexScript) -> AccountLimits:
    executor = FakeExecutor([], peers=script.peer)
    return read_limits("codex", executor=executor, env=ENV, clock=FakeClock())


def _spend(script: CodexScript, key: str, credit_id: str | None = None) -> ResetCreditResult:
    executor = FakeExecutor([], peers=script.peer)
    return use_reset_credit(
        "codex",
        idempotency_key=key,
        credit_id=credit_id,
        executor=executor,
        env=ENV,
        clock=FakeClock(),
    )


class TestRecordedReply:
    """What codex-cli 0.160.0 answered."""

    def test_a_pro_plan_has_one_weekly_window_and_no_credits(self) -> None:
        wire = json.loads(FIXTURE.read_text())["result"]
        limits = parse_rate_limits(GetAccountRateLimitsResponse.from_wire(wire))
        assert limits.unavailable is None
        (window,) = limits.windows
        assert (window.id, window.kind, window.label) == (
            "codex:primary",
            WindowKind.WEEKLY,
            "Weekly",
        )
        assert window.used_fraction == pytest.approx(0.03)
        assert window.window_minutes == 10_080
        assert window.resets_at == dt.datetime.fromtimestamp(1_794_082_513, tz=dt.timezone.utc)
        assert limits.plan == "pro"
        assert limits.account == account_digest("codex", "account-1")
        assert limits.reset_credits is not None
        assert (limits.reset_credits.available, limits.reset_credits.credits) == (0, ())
        assert limits.usage_url == "https://chatgpt.com/codex/settings/usage"


class TestTheRead:
    def test_it_says_hello_and_asks_for_the_limits_and_nothing_else(self) -> None:
        script = CodexScript(account=CodexAccount())
        limits = _read(script)
        assert limits.unavailable is None
        assert _methods(script) == ["initialize", READ]
        assert script.requests(CONSUME) == []
        assert script.violations == []
        assert script.thread_ids() == []

    def test_a_read_reaches_the_server_with_the_parameters_the_schema_names(self) -> None:
        script = CodexScript(account=CodexAccount(credits=("c1",)))
        _read(script)
        (request,) = script.requests(READ)
        assert request.params.to_wire() == {}
        result = script.account
        assert result is not None
        assert [m for m in script.received if not isinstance(m, ClientRequest)]  # `initialized`

    def test_windows_are_told_apart_by_length_not_position(self) -> None:
        account = CodexAccount(
            plan="free",
            buckets={
                "codex": (
                    CodexWindow(10, 300, 1_794_000_000),
                    CodexWindow(20, 10_080, 1_794_100_000),
                ),
                "codex_spark": (CodexWindow(5, 43_200, 1_794_200_000), None),
            },
        )
        windows = {w.id: w for w in _read(CodexScript(account=account)).windows}
        assert windows["codex:primary"].kind is WindowKind.SESSION
        assert windows["codex:secondary"].kind is WindowKind.WEEKLY
        assert windows["codex_spark:primary"].kind is WindowKind.MONTHLY
        assert windows["codex_spark:primary"].label == "Monthly · Codex_Spark"
        assert windows["codex:primary"].label == "Session"

    def test_credits_are_listed_with_their_ids(self) -> None:
        limits = _read(CodexScript(account=CodexAccount(credits=("c1", "c2"))))
        held = limits.reset_credits
        assert held is not None
        assert held.available == 2
        assert [c.id for c in held.credits] == ["c1", "c2"]
        assert all(c.granted_at.tzinfo is not None for c in held.credits)

    def test_a_server_that_errors_is_a_failed_read_that_says_so(self) -> None:
        limits = _read(CodexScript(account=CodexAccount(read_error="not signed in")))
        assert limits.windows == ()
        assert limits.unavailable is not None
        assert limits.unavailable.reason is LimitsUnavailableReason.PROBE_FAILED
        assert "not signed in" in limits.unavailable.detail

    def test_a_server_without_the_method_is_a_failed_read(self) -> None:
        limits = _read(CodexScript())
        assert limits.unavailable is not None
        assert f"unsupported {READ}" in limits.unavailable.detail

    def test_a_server_that_never_answers_times_out_on_the_clock_it_was_given(self) -> None:
        clock = FakeClock()
        executor = FakeExecutor([], peers=lambda _request: SilentPeer())
        limits = read_limits("codex", executor=executor, env=ENV, timeout=3.0, clock=clock)
        assert limits.unavailable is not None
        assert "did not answer initialize within 3.0s" in limits.unavailable.detail
        assert executor.processes[0].stdin_closed

    def test_a_server_that_dies_at_start_is_a_failed_read_with_its_last_words(self) -> None:
        class Dies:
            def on_start(self) -> list[Any]:
                return [StderrLine("not logged in\n"), ProcessExited(1)]

            def on_stdin(self, data: str) -> list[Any]:
                del data
                return []

            def on_stdin_closed(self) -> list[Any]:
                return []

        executor = FakeExecutor([], peers=lambda _request: Dies())
        limits = read_limits("codex", executor=executor, env=ENV, clock=FakeClock())
        assert limits.unavailable is not None
        assert "not logged in" in limits.unavailable.detail

    @pytest.mark.parametrize(
        "method", ["turn/futureThing", "item/futureThing", "thread/futureThing", "turn/started"]
    )
    def test_any_sign_of_a_thread_stops_the_read(self, method: str) -> None:
        script = CodexScript(account=CodexAccount())

        class Speaks:
            """Answers like the real server, but announces a thread before the answer."""

            def __init__(self) -> None:
                self.inner = script.peer()

            def on_start(self) -> list[Any]:
                return list(self.inner.on_start())

            def on_stdin(self, data: str) -> list[Any]:
                out = list(self.inner.on_stdin(data))
                if READ in data:
                    note = json.dumps({"method": method, "params": {}}) + "\n"
                    return [StdoutLine(note), *out]
                return out

            def on_stdin_closed(self) -> list[Any]:
                return list(self.inner.on_stdin_closed())

        executor = FakeExecutor([], peers=lambda _request: Speaks())
        limits = read_limits("codex", executor=executor, env=ENV, clock=FakeClock())
        assert limits.unavailable is not None
        assert limits.unavailable.reason is LimitsUnavailableReason.PROBE_FAILED

    @given(
        st.dictionaries(
            st.sampled_from(["codex", "codex_a", "codex_b"]),
            st.tuples(
                st.none()
                | st.builds(
                    CodexWindow,
                    st.integers(0, 400),
                    st.sampled_from([300, 10_080, 43_200, None, 77]),
                    st.none() | st.integers(1, 4_000_000_000),
                ),
                st.none()
                | st.builds(
                    CodexWindow,
                    st.integers(0, 400),
                    st.sampled_from([300, 10_080, 43_200, None, 77]),
                    st.none() | st.integers(1, 4_000_000_000),
                ),
            ),
            min_size=1,
        ),
        st.lists(st.sampled_from(["c1", "c2", "c3"]), unique=True, max_size=3),
    )
    def test_a_read_is_a_read_for_any_account(
        self,
        buckets: dict[str, tuple[CodexWindow | None, CodexWindow | None]],
        held: list[str],
    ) -> None:
        script = CodexScript(account=CodexAccount(buckets=buckets, credits=tuple(held)))
        limits = _read(script)
        assert limits.unavailable is None
        expected = sum(w is not None for pair in buckets.values() for w in pair)
        assert len(limits.windows) == expected
        assert limits.reset_credits is not None
        assert limits.reset_credits.available == len(held)
        assert _methods(script) == ["initialize", READ]  # never a thread, a turn or a spend
        assert script.violations == []


class TestSameWindowsFromTurns:
    """An ``account/rateLimits/updated`` notification and a read name a window the same way."""

    @given(
        st.sampled_from(["codex", "codex_a"]),
        st.integers(0, 300),
        st.sampled_from([300, 10_080, 43_200, None]),
        st.integers(1, 4_000_000_000),
    )
    def test_the_window_from_a_turn_is_the_window_from_a_read(
        self, limit: str, used: int, minutes: int | None, resets: int
    ) -> None:
        account = CodexAccount(buckets={limit: (CodexWindow(used, minutes, resets), None)})
        (read,) = _read(CodexScript(account=account)).windows
        wire = account.rate_limits()["rateLimitsByLimitId"][limit]
        (status,) = rate_limit_statuses(RateLimitSnapshot.from_wire(wire))
        turn = limit_window("codex", status)
        assert turn is not None
        assert (turn.id, turn.kind, turn.used_fraction, turn.resets_at, turn.window_minutes) == (
            read.id,
            read.kind,
            read.used_fraction,
            read.resets_at,
            read.window_minutes,
        )

    def test_a_status_that_only_says_a_limit_was_reached_is_no_window(self) -> None:
        wire = {"limitId": "codex", "rateLimitReachedType": "rate_limit_reached"}
        (status,) = rate_limit_statuses(RateLimitSnapshot.from_wire(wire))
        assert limit_window("codex", status) is None


class TestSpendingACredit:
    def test_it_spends_the_named_credit_and_reads_again(self) -> None:
        script = CodexScript(account=CodexAccount(credits=("c1", "c2")))
        result = _spend(script, "k1", "c2")
        assert result.outcome is ResetOutcome.RESET
        assert script.account is not None
        assert script.account.spent == ["c2"]
        assert result.limits.reset_credits is not None
        assert result.limits.reset_credits.available == 1
        assert [w.used_fraction for w in result.limits.windows] == [0.0]
        (spend,) = script.requests(CONSUME)
        assert spend.params.to_wire() == {"idempotencyKey": "k1", "creditId": "c2"}

    def test_without_a_credit_id_the_server_picks_one(self) -> None:
        script = CodexScript(account=CodexAccount(credits=("c1",)))
        assert _spend(script, "k1").outcome is ResetOutcome.RESET
        (spend,) = script.requests(CONSUME)
        assert spend.params.to_wire() == {"idempotencyKey": "k1"}

    def test_with_no_credit_nothing_is_spent_and_it_says_so(self) -> None:
        script = CodexScript(account=CodexAccount())
        assert _spend(script, "k1").outcome is ResetOutcome.NO_CREDIT

    def test_a_key_is_needed(self) -> None:
        script = CodexScript(account=CodexAccount(credits=("c1",)))
        with pytest.raises(ValueError, match="idempotency_key"):
            _spend(script, "  ")
        assert script.spawned == 0

    def test_an_outcome_it_does_not_know_is_an_error_not_a_guess(self) -> None:
        class Strange(CodexAccount):
            def consume(self, idempotency_key: str, credit_id: str | None) -> str:
                del idempotency_key, credit_id
                return "somethingNew"

        with pytest.raises(TurnFailedError, match="may or may not be spent"):
            _spend(CodexScript(account=Strange(credits=("c1",))), "k1")

    def test_a_server_that_refuses_is_an_error_the_caller_sees(self) -> None:
        with pytest.raises(TurnFailedError, match="unsupported"):
            _spend(CodexScript(), "k1")

    @given(
        st.integers(0, 3),
        st.lists(
            st.tuples(
                st.sampled_from(["a", "b", "c"]), st.none() | st.sampled_from(["c0", "c1", "c9"])
            ),
            max_size=8,
        ),
    )
    def test_a_retry_with_the_same_key_never_spends_another_credit(
        self, held: int, attempts: list[tuple[str, str | None]]
    ) -> None:
        account = CodexAccount(credits=tuple(f"c{n}" for n in range(held)))
        script = CodexScript(account=account)
        answered: dict[str, ResetOutcome] = {}
        for key, credit in attempts:
            before = list(account.spent)
            outcome = _spend(script, key, credit).outcome
            if key in answered:
                assert outcome is answered[key]
                assert account.spent == before
            answered.setdefault(key, outcome)
            assert len(set(account.spent)) == len(account.spent)
        assert len(account.spent) <= min(held, len({key for key, _ in attempts}))
        assert len(account.spent) == sum(o is ResetOutcome.RESET for o in answered.values())
