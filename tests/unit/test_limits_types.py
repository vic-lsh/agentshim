"""The limits value types: what they accept, and the words they produce."""

from __future__ import annotations

import datetime as dt
import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from agentshim import (
    AccountLimits,
    LimitsUnavailableReason,
    LimitWindow,
    ResetCredits,
    WindowKind,
    account_digest,
)
from agentshim.core.limits import (
    SESSION_MINUTES,
    WEEKLY_MINUTES,
    kind_of_minutes,
    label_of_kind,
)

UTC = dt.timezone.utc


def _window(
    *,
    id: str = "w",  # noqa: A002 - the field's own name
    label: str = "Weekly",
    used_fraction: float = 0.5,
    resets_at: dt.datetime | None = None,
    window_minutes: int | None = None,
) -> LimitWindow:
    return LimitWindow(
        id=id,
        kind=WindowKind.WEEKLY,
        label=label,
        used_fraction=used_fraction,
        resets_at=resets_at,
        window_minutes=window_minutes,
    )


class TestLimitWindow:
    @given(st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False))
    def test_any_finite_share_of_zero_or_more_is_a_window(self, used: float) -> None:
        window = _window(used_fraction=used)
        assert window.remaining_fraction == max(0.0, 1.0 - used)
        assert 0.0 <= window.remaining_fraction <= 1.0

    @pytest.mark.parametrize("used", [-0.01, math.nan, math.inf, -math.inf])
    def test_a_share_that_is_negative_or_not_finite_is_refused(self, used: float) -> None:
        with pytest.raises(ValueError, match="used_fraction"):
            _window(used_fraction=used)

    def test_a_naive_reset_time_is_refused(self) -> None:
        with pytest.raises(ValueError, match="timezone aware"):
            _window(resets_at=dt.datetime(2026, 11, 1))  # noqa: DTZ001 - the point of the test
        assert _window(resets_at=dt.datetime(2026, 11, 1, tzinfo=UTC)).resets_at is not None

    def test_an_empty_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="id"):
            _window(id="")
        with pytest.raises(ValueError, match="label"):
            _window(label="")

    @pytest.mark.parametrize("minutes", [0, -5])
    def test_a_window_length_must_be_positive(self, minutes: int) -> None:
        with pytest.raises(ValueError, match="window_minutes"):
            _window(window_minutes=minutes)


class TestKinds:
    def test_the_lengths_the_providers_report(self) -> None:
        assert kind_of_minutes(SESSION_MINUTES) is WindowKind.SESSION
        assert kind_of_minutes(WEEKLY_MINUTES) is WindowKind.WEEKLY
        assert kind_of_minutes(None) is WindowKind.OTHER

    @given(st.integers(min_value=28 * 24 * 60, max_value=31 * 24 * 60))
    def test_any_length_of_28_to_31_days_is_monthly(self, minutes: int) -> None:
        assert kind_of_minutes(minutes) is WindowKind.MONTHLY

    @given(st.integers().filter(lambda m: m not in {SESSION_MINUTES, WEEKLY_MINUTES}))
    def test_a_kind_is_monthly_only_for_a_month(self, minutes: int) -> None:
        in_a_month = 28 * 24 * 60 <= minutes <= 31 * 24 * 60
        assert (kind_of_minutes(minutes) is WindowKind.MONTHLY) == in_a_month

    def test_labels(self) -> None:
        assert label_of_kind(WindowKind.SESSION) == "Session"
        assert label_of_kind(WindowKind.WEEKLY, "Opus") == "Weekly · Opus"
        assert label_of_kind(WindowKind.OTHER) == "Other"
        assert label_of_kind(WindowKind.OTHER, "Spark") == "Spark"


class TestAccountDigest:
    @given(st.text(min_size=1), st.text(min_size=1))
    def test_the_digest_is_stable_and_hides_what_it_names(self, provider: str, who: str) -> None:
        digest = account_digest(provider, who)
        assert digest == account_digest(provider, who)
        assert len(digest) == 16
        assert who.strip().lower() not in digest or len(who.strip()) <= 2

    @given(st.emails())
    def test_case_and_padding_do_not_make_a_different_account(self, email: str) -> None:
        assert account_digest("claude", email) == account_digest("claude", f"  {email.upper()} ")

    @given(st.emails())
    def test_the_same_name_on_two_providers_is_two_accounts(self, email: str) -> None:
        assert account_digest("claude", email) != account_digest("codex", email)


class TestAccountLimits:
    def test_windows_must_have_distinct_ids(self) -> None:
        with pytest.raises(ValueError, match="repeat an id"):
            AccountLimits(provider="p", windows=(_window(), _window()))

    def test_a_failed_read_carries_no_windows(self) -> None:
        with pytest.raises(ValueError, match="unavailable"):
            AccountLimits(
                provider="p",
                windows=(_window(),),
                unavailable=AccountLimits.failed(
                    "p", LimitsUnavailableReason.PROBE_FAILED
                ).unavailable,
            )

    def test_failed_names_its_reason(self) -> None:
        failed = AccountLimits.failed("p", LimitsUnavailableReason.UNSUPPORTED, "no read")
        assert failed.windows == ()
        assert failed.unavailable is not None
        assert failed.unavailable.reason is LimitsUnavailableReason.UNSUPPORTED
        assert failed.unavailable.detail == "no read"

    def test_a_count_of_credits_is_never_negative(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            ResetCredits(available=-1)
