"""Claude's limits read: the wire it sends, what it makes of the answer, and how it fails."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pytest
from agentshim import (
    AccountLimits,
    LimitsUnavailableReason,
    ProviderCapabilityError,
    RateLimitStatus,
    WindowKind,
    account_digest,
    limit_window,
    read_limits,
    use_reset_credit,
)
from agentshim.execution.process import ProcessExited, StderrLine, StdoutLine
from agentshim.providers.claude.limits import parse_usage
from agentshim.providers.claude.rate_limits import parse_rate_limit
from agentshim.testing import FakeExecutor
from agentshim.testing.claude_stream import ClaudeStreamPeers
from agentshim.testing.clock import FakeClock
from agentshim.testing.limits import claude_account, claude_usage
from agentshim.testing.process import PeerOutput, SilentPeer
from hypothesis import given
from hypothesis import strategies as st

FIXTURES = Path(__file__).parents[3] / "fixtures" / "claude_limits"
FIVE_HOUR = "2026-11-01T05:30:00+00:00"
WEEK = "2026-11-01T18:00:00.123456+00:00"
ENV = {"PATH": "/bin"}


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text())["response"]


class _Says:
    """A CLI that writes *start* the moment it starts and then ignores everything."""

    def __init__(self, *start: PeerOutput) -> None:
        self._start = list(start)

    def on_start(self) -> list[PeerOutput]:
        return list(self._start)

    def on_stdin(self, data: str) -> list[PeerOutput]:
        del data
        return []

    def on_stdin_closed(self) -> list[PeerOutput]:
        return []


def _read(peers: ClaudeStreamPeers) -> tuple[AccountLimits, FakeExecutor]:
    executor = FakeExecutor([], peers=peers.build)
    return read_limits("claude", executor=executor, env=ENV, clock=FakeClock()), executor


class TestRecordedReply:
    """What the real CLI (2.1.316; the shape is the same on 2.1.304) sent."""

    def test_the_windows_the_account_has(self) -> None:
        limits = parse_usage(_fixture("initialize.json")["account"], _fixture("get_usage.json"))
        assert limits.unavailable is None
        by_id = {w.id: w for w in limits.windows}
        assert list(by_id) == ["five_hour", "seven_day", "seven_day_fable"]
        session = by_id["five_hour"]
        assert (session.kind, session.label, session.window_minutes) == (
            WindowKind.SESSION,
            "Session",
            300,
        )
        assert session.used_fraction == pytest.approx(0.10)
        assert session.resets_at == dt.datetime(
            2026, 11, 1, 5, 29, 59, 527557, tzinfo=dt.timezone.utc
        )
        weekly = by_id["seven_day"]
        assert (weekly.kind, weekly.label, weekly.window_minutes) == (
            WindowKind.WEEKLY,
            "Weekly",
            10_080,
        )
        scoped = by_id["seven_day_fable"]
        assert (scoped.label, scoped.used_fraction) == ("Weekly · Fable", 0.0)

    def test_the_account_is_a_digest_and_the_plan_a_word(self) -> None:
        limits = parse_usage(_fixture("initialize.json")["account"], _fixture("get_usage.json"))
        assert limits.account == account_digest("claude", "person@example.com")
        assert "person" not in repr(limits)
        assert limits.plan == "team"
        assert limits.usage_url == "https://claude.ai/settings/usage"
        assert limits.reset_credits is None


class TestParse:
    def test_model_windows_take_the_models_display_name(self) -> None:
        usage = claude_usage(model_scoped=[("Opus 5", 40, WEEK), ("Claude Sonnet", 10, None)])
        windows = {w.id: w for w in parse_usage(None, usage).windows}
        assert windows["seven_day_opus_5"].label == "Weekly · Opus 5"
        assert windows["seven_day_claude_sonnet"].resets_at is None

    def test_the_older_model_keys_fill_in_only_when_model_scoped_lacks_them(self) -> None:
        old = claude_usage(seven_day_opus=(50, WEEK), seven_day_sonnet=(20, WEEK))
        assert {w.id for w in parse_usage(None, old).windows} == {
            "seven_day_opus",
            "seven_day_sonnet",
        }
        both = claude_usage(seven_day_opus=(50, WEEK), model_scoped=[("Opus", 70, WEEK)])
        windows = {w.id: w for w in parse_usage(None, both).windows}
        assert windows["seven_day_opus"].used_fraction == pytest.approx(0.70)

    def test_a_window_that_states_no_share_is_left_out_not_made_up(self) -> None:
        usage = claude_usage(five_hour=(None, FIVE_HOUR), seven_day=(12.5, None))
        windows = {w.id: w for w in parse_usage(None, usage).windows}
        assert list(windows) == ["seven_day"]
        assert windows["seven_day"].used_fraction == pytest.approx(0.125)

    @pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), "40", True, None])
    def test_a_share_that_is_not_a_usable_number_is_left_out(self, bad: object) -> None:
        usage = claude_usage(seven_day=(40, WEEK))
        usage["rate_limits"]["five_hour"] = {"utilization": bad, "resets_at": FIVE_HOUR}
        assert [w.id for w in parse_usage(None, usage).windows] == ["seven_day"]

    def test_a_login_with_no_plan_says_so_and_has_no_windows(self) -> None:
        usage = claude_usage(available=False)
        limits = parse_usage(claude_account(api_provider="firstParty"), usage)
        assert limits.windows == ()
        assert limits.unavailable is not None
        assert limits.unavailable.reason is LimitsUnavailableReason.API_KEY
        assert "firstParty" in limits.unavailable.detail
        assert limits.plan is None

    def test_no_email_is_no_account(self) -> None:
        assert parse_usage(claude_account(email=None), claude_usage()).account is None

    @given(
        st.recursive(
            st.none() | st.booleans() | st.integers() | st.floats() | st.text(max_size=8),
            lambda inner: (
                st.lists(inner, max_size=3)
                | st.dictionaries(st.text(max_size=12), inner, max_size=4)
            ),
            max_leaves=20,
        )
    )
    def test_no_reply_makes_the_reader_raise(self, junk: object) -> None:
        """Whatever the CLI answers in a shape it may change, the read returns limits."""
        shapes: list[dict[str, Any]] = [
            {
                "rate_limits_available": True,
                "rate_limits": {"five_hour": junk, "model_scoped": junk},
            },
            {"rate_limits": {"model_scoped": [junk], "seven_day": {"utilization": junk}}},
            {"rate_limits": junk, "subscription_type": junk},
            junk if isinstance(junk, dict) else {},  # pyright: ignore[reportUnknownVariableType]
        ]
        account = {"email": junk, "apiProvider": junk}
        for usage in shapes:
            for window in parse_usage(account, usage).windows:
                assert window.used_fraction >= 0
                assert window.resets_at is None or window.resets_at.tzinfo is not None


class TestTheRead:
    def test_it_asks_for_the_usage_and_nothing_else(self) -> None:
        peers = ClaudeStreamPeers(
            account=claude_account(), usage=claude_usage(five_hour=(39, FIVE_HOUR))
        )
        limits, _ = _read(peers)
        assert [w.id for w in limits.windows] == ["five_hour"]
        assert limits.unavailable is None
        (peer,) = peers.peers
        requests = [m["request"] for m in peer.received if m["type"] == "control_request"]
        assert requests == [
            {"subtype": "initialize"},
            {"subtype": "get_usage", "skip_behaviors": True},
        ]
        assert peer.prompts == []  # no user message, so no model call
        assert [m["type"] for m in peer.received] == ["control_request", "control_request"]

    def test_the_process_starts_with_nothing_that_could_act(self) -> None:
        peers = ClaudeStreamPeers(account=claude_account(), usage=claude_usage())
        _, executor = _read(peers)
        argv = executor.spawns[0].argv
        for flag in ("--no-session-persistence", "--strict-mcp-config"):
            assert flag in argv
        assert "--resume" not in " ".join(argv)
        assert "--model" not in argv  # the caller's environment picks the model; none is run
        assert argv[argv.index("--setting-sources") + 1] == "user"
        assert json.loads(argv[argv.index("--settings") + 1]) == {"disableAllHooks": True}
        assert json.loads(argv[argv.index("--mcp-config") + 1]) == {"mcpServers": {}}

    def test_the_process_is_stopped_after_the_read(self) -> None:
        _, executor = _read(ClaudeStreamPeers(account=claude_account(), usage=claude_usage()))
        (process,) = executor.processes
        assert process.stdin_closed
        assert process.wait(0) is not None

    def test_a_nested_session_marker_is_not_passed_on(self) -> None:
        executor = FakeExecutor([], peers=ClaudeStreamPeers(usage=claude_usage()).build)
        read_limits(
            "claude", executor=executor, env={"PATH": "/bin", "CLAUDECODE": "1"}, clock=FakeClock()
        )
        assert "CLAUDECODE" not in executor.spawns[0].env

    def test_a_cli_that_refuses_the_request_is_a_failed_read_that_says_so(self) -> None:
        limits, _ = _read(ClaudeStreamPeers(account=claude_account()))  # no get_usage support
        assert limits.windows == ()
        assert limits.unavailable is not None
        assert limits.unavailable.reason is LimitsUnavailableReason.PROBE_FAILED
        assert "refused get_usage" in limits.unavailable.detail

    def test_a_cli_that_never_answers_times_out_on_the_clock_it_was_given(self) -> None:
        clock = FakeClock()
        executor = FakeExecutor([], peers=lambda _request: SilentPeer())
        limits = read_limits("claude", executor=executor, env=ENV, timeout=2.0, clock=clock)
        assert limits.unavailable is not None
        assert "did not answer initialize within 2.0s" in limits.unavailable.detail
        assert 2.0 <= clock.monotonic() < 4.0
        assert executor.processes[0].terminate_calls == 1  # it ignored end of input

    def test_a_cli_that_is_not_installed_is_a_failed_read(self) -> None:
        executor = FakeExecutor([], binaries={"codex": "/bin/codex"}, peers=lambda _r: SilentPeer())
        limits = read_limits("claude", executor=executor, env=ENV, clock=FakeClock())
        assert limits.unavailable is not None
        assert limits.unavailable.reason is LimitsUnavailableReason.PROBE_FAILED
        assert "claude" in limits.unavailable.detail
        assert executor.spawns == []

    def test_a_cli_that_exits_before_answering_is_a_failed_read_with_its_last_words(self) -> None:
        dies = _Says(StderrLine("not logged in\n"), ProcessExited(1))
        executor = FakeExecutor([], peers=lambda _request: dies)
        limits = read_limits("claude", executor=executor, env=ENV, clock=FakeClock())
        assert limits.unavailable is not None
        assert "not logged in" in limits.unavailable.detail

    @pytest.mark.parametrize("frame", ["assistant", "stream_event", "result"])
    def test_any_sign_of_a_model_turn_stops_the_read(self, frame: str) -> None:
        speaks = _Says(StdoutLine(json.dumps({"type": frame}) + "\n"))
        executor = FakeExecutor([], peers=lambda _request: speaks)
        limits = read_limits("claude", executor=executor, env=ENV, clock=FakeClock())
        assert limits.unavailable is not None
        assert "model turn" in limits.unavailable.detail
        assert executor.processes[0].stdin_closed

    @given(
        st.lists(
            st.tuples(
                st.sampled_from(["five_hour", "seven_day"]),
                st.floats(min_value=0, max_value=500, allow_nan=False),
            ),
            max_size=2,
            unique_by=lambda pair: pair[0],
        ),
        st.lists(
            st.tuples(st.text("abcdefgh ", min_size=1, max_size=6), st.floats(0, 100)),
            max_size=3,
            unique_by=lambda pair: pair[0].strip().lower(),
        ),
    )
    def test_every_stated_window_comes_back_once_and_none_is_invented(
        self, fixed: list[tuple[str, float]], scoped: list[tuple[str, float]]
    ) -> None:
        usage = claude_usage(
            five_hour=next(((u, FIVE_HOUR) for k, u in fixed if k == "five_hour"), None),
            seven_day=next(((u, WEEK) for k, u in fixed if k == "seven_day"), None),
            model_scoped=[(name, used, WEEK) for name, used in scoped if name.strip()],
        )
        peers = ClaudeStreamPeers(account=claude_account(), usage=usage)
        limits, _ = _read(peers)
        stated = len(fixed) + len({n.strip().lower() for n, _ in scoped if n.strip()})
        assert len(limits.windows) <= stated  # names can merge into one slug; none is invented
        for window in limits.windows:
            assert window.used_fraction >= 0
            assert window.window_minutes in {300, 10_080}


class TestNoResetCredits:
    def test_claude_has_none_to_spend(self) -> None:
        with pytest.raises(ProviderCapabilityError, match="no reset credits"):
            use_reset_credit(
                "claude",
                idempotency_key="k",
                executor=FakeExecutor([], peers=ClaudeStreamPeers().build),
                env=ENV,
                clock=FakeClock(),
            )

    def test_and_nothing_is_started_to_find_out(self) -> None:
        executor = FakeExecutor([], peers=ClaudeStreamPeers().build)
        with pytest.raises(ProviderCapabilityError):
            use_reset_credit("claude", idempotency_key="k", executor=executor, env=ENV)
        assert executor.spawns == []


class TestSameWindowsFromTurns:
    """A turn's ``rate_limit_event`` and a read name the same window the same way."""

    def _statuses(self) -> list[RateLimitStatus]:
        frame = parse_rate_limit(
            {
                "rate_limit_info": {
                    "status": "allowed",
                    "unifiedWindows": {
                        "five_hour": {"utilization": 0.39, "resetsAt": 1_793_500_200},
                        "seven_day": {"utilization": 0.28, "resetsAt": 1_794_000_000},
                    },
                }
            }
        )
        assert frame is not None
        return list(frame.statuses)

    def test_ids_kinds_and_labels_match_the_read(self) -> None:
        read = parse_usage(
            None, claude_usage(five_hour=(39, FIVE_HOUR), seven_day=(28, WEEK))
        ).windows
        turns = [limit_window("claude", status) for status in self._statuses()]
        assert [(w.id, w.kind, w.label, w.window_minutes) for w in read] == [
            (w.id, w.kind, w.label, w.window_minutes) for w in turns if w is not None
        ]
        assert [w.used_fraction for w in turns if w is not None] == pytest.approx([0.39, 0.28])
        assert all(w.resets_at is not None for w in turns if w is not None)

    def test_a_limit_that_is_reached_without_a_share_is_a_full_window(self) -> None:
        status = RateLimitStatus(
            window="five_hour", used_fraction=None, resets_at=None, exhausted=True
        )
        window = limit_window("claude", status)
        assert window is not None
        assert window.used_fraction == 1.0

    @pytest.mark.parametrize(
        "status",
        [
            RateLimitStatus(window=None, used_fraction=0.5, resets_at=None),
            RateLimitStatus(window="five_hour", used_fraction=None, resets_at=None),
            RateLimitStatus(window="five_hour", used_fraction=-0.5, resets_at=None),
            RateLimitStatus(window="five_hour", used_fraction=float("nan"), resets_at=None),
        ],
    )
    def test_a_status_with_nothing_to_draw_is_no_window(self, status: RateLimitStatus) -> None:
        assert limit_window("claude", status) is None

    @given(
        st.text(min_size=1, max_size=20).filter(lambda s: s == s.strip()),
        st.floats(min_value=0, max_value=5, allow_nan=False),
        st.none() | st.integers(min_value=1, max_value=100_000),
        st.none() | st.floats(min_value=0, max_value=4e9),
    )
    def test_any_named_window_with_a_share_is_a_window_with_that_name_as_its_id(
        self, name: str, used: float, minutes: int | None, resets: float | None
    ) -> None:
        status = RateLimitStatus(
            window=name, used_fraction=used, resets_at=resets, window_minutes=minutes
        )
        window = limit_window("claude", status)
        assert window is not None
        assert window.id == name
        assert window.used_fraction == used
        assert (window.resets_at is None) == (resets is None)
