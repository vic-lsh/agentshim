"""An account's plan limits as a caller sees them, and the contract for reading them.

A subscription login has limits that are not a model's: a session window of a
few hours, a weekly one, perhaps one per model, each with a share used and a
time it refills. A provider reports them two ways. While a turn runs it emits
``RateLimitStatus`` events, one window at a time. On request, a provider CLI can
also say what they are right now without running a turn. This module is the
second shape: :class:`AccountLimits`, what a read returns.

The two shapes name windows the same way. :func:`~agentshim.limits.limit_window`
turns a ``RateLimitStatus`` into the :class:`LimitWindow` a read would have
returned for it, so a caller that keeps readings from turns and also reads on
demand can merge them by ``LimitWindow.id`` and never learns a provider's own
window names (``five_hour``, ``primary``).

Nothing here identifies a person. :attr:`AccountLimits.account` is a digest, so
a caller can tell that two machines hold the same account without either
learning an email address or an account id.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import datetime as dt
    from collections.abc import Mapping

    from agentshim.execution.executor import CommandExecutor

    from .clock import Clock
    from .events import RateLimitStatus

#: Window lengths (minutes) that name a kind. A monthly window is any length of 28 to 31 days.
SESSION_MINUTES = 300
WEEKLY_MINUTES = 10_080
_MONTHLY_MINUTES = range(28 * 24 * 60, 31 * 24 * 60 + 1)

_DIGEST_HEX = 16


class WindowKind(str, Enum):
    """What a window is, in words every provider shares."""

    SESSION = "session"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    OTHER = "other"


def kind_of_minutes(minutes: int | None) -> WindowKind:
    """The kind of a window that lasts *minutes*; ``OTHER`` when unknown or unusual."""
    if minutes == SESSION_MINUTES:
        return WindowKind.SESSION
    if minutes == WEEKLY_MINUTES:
        return WindowKind.WEEKLY
    if minutes is not None and minutes in _MONTHLY_MINUTES:
        return WindowKind.MONTHLY
    return WindowKind.OTHER


def label_of_kind(kind: WindowKind, scope: str | None = None) -> str:
    """The display label of a window: ``Session``, ``Weekly``, ``Monthly``, ``Weekly · Opus``.

    ``OTHER`` windows are labelled by *scope* alone, and ``Other`` when they have none.
    """
    base = {
        WindowKind.SESSION: "Session",
        WindowKind.WEEKLY: "Weekly",
        WindowKind.MONTHLY: "Monthly",
        WindowKind.OTHER: "",
    }[kind]
    if not scope:
        return base or "Other"
    return f"{base} · {scope}" if base else scope


def account_digest(provider: str, identity: str) -> str:
    """An opaque, stable key for the account *identity* names on *provider*.

    Equal inputs give equal digests on every machine, so a caller can pool
    windows of one account seen from several. It is a hash of a name, not a
    secret: it keeps an email or an account id out of what is stored and
    shown, and does not hide it from someone who already knows it.
    """
    text = f"{provider}\0{identity.strip().lower()}"
    return hashlib.sha256(text.encode()).hexdigest()[:_DIGEST_HEX]


@dataclass(frozen=True)
class LimitWindow:
    """One limit of an account: a share of an allowance and when it refills.

    ``id`` is stable for the same window across reads and across the two ways a
    provider reports it, so readings merge by it. ``used_fraction`` is the share
    used, ``0.0`` to ``1.0`` (it may exceed 1 when the provider reports
    over-use). ``resets_at`` is when the window refills, timezone aware, or
    ``None`` when the provider did not say. ``window_minutes`` is the window's
    length when the provider says it, or when the window's name fixes it.
    """

    id: str
    kind: WindowKind
    label: str
    used_fraction: float
    resets_at: dt.datetime | None = None
    window_minutes: int | None = None

    def __post_init__(self) -> None:
        """Reject a window nobody could draw or merge."""
        if not self.id or self.id != self.id.strip():
            msg = "limit window id must be non-empty and unpadded"
            raise ValueError(msg)
        if not self.label:
            msg = f"limit window {self.id!r} needs a label"
            raise ValueError(msg)
        if not math.isfinite(self.used_fraction) or self.used_fraction < 0:
            msg = f"limit window {self.id!r} used_fraction must be a finite share of 0 or more"
            raise ValueError(msg)
        if self.resets_at is not None and self.resets_at.tzinfo is None:
            msg = f"limit window {self.id!r} resets_at must be timezone aware"
            raise ValueError(msg)
        if self.window_minutes is not None and self.window_minutes <= 0:
            msg = f"limit window {self.id!r} window_minutes must be positive"
            raise ValueError(msg)

    @property
    def remaining_fraction(self) -> float:
        """The share still available, never below zero."""
        return max(0.0, 1.0 - self.used_fraction)


@dataclass(frozen=True)
class ResetCredit:
    """One credit that refills the account's limits at once when spent.

    ``id`` is the provider's opaque handle for it, the value
    :func:`~agentshim.limits.use_reset_credit` takes to spend this credit and
    no other.
    """

    id: str
    granted_at: dt.datetime
    expires_at: dt.datetime | None = None
    title: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        """Require an id, and aware times."""
        if not self.id:
            msg = "reset credit id must be non-empty"
            raise ValueError(msg)
        for name, value in (("granted_at", self.granted_at), ("expires_at", self.expires_at)):
            if value is not None and value.tzinfo is None:
                msg = f"reset credit {name} must be timezone aware"
                raise ValueError(msg)


@dataclass(frozen=True)
class ResetCredits:
    """The reset credits an account holds.

    ``available`` is the count the provider states. ``credits`` lists the ones
    it described, which can be fewer (the provider may cap the list, or skip
    the lookup): never infer the count from its length.
    """

    available: int
    credits: tuple[ResetCredit, ...] = ()

    def __post_init__(self) -> None:
        """Reject a negative count."""
        if self.available < 0:
            msg = "reset credits available must not be negative"
            raise ValueError(msg)


class LimitsUnavailableReason(str, Enum):
    """Why a read has no windows.

    ``UNSUPPORTED``: this provider has no way to read limits. ``API_KEY``: the
    CLI is billed by token, so there is no plan to have limits. ``PROBE_FAILED``:
    the read could not be done now (no binary, not logged in, timed out, an error
    reply); ``LimitsUnavailable.detail`` says which.
    """

    UNSUPPORTED = "unsupported"
    API_KEY = "api_key"
    PROBE_FAILED = "probe_failed"


@dataclass(frozen=True)
class LimitsUnavailable:
    """A read that produced no limits, and why."""

    reason: LimitsUnavailableReason
    detail: str = ""


@dataclass(frozen=True)
class AccountLimits:
    """What one read of a provider's limits returned.

    A read never raises for a CLI problem: it returns ``unavailable`` and no
    windows, so a caller keeps the last good reading and says why it is old.
    ``account`` is a digest (see :func:`account_digest`), ``None`` when the
    provider named no account. ``plan`` is the provider's own plan word
    (``team``, ``pro``). ``usage_url`` is where a person sees these limits.
    """

    provider: str
    windows: tuple[LimitWindow, ...] = ()
    account: str | None = None
    plan: str | None = None
    reset_credits: ResetCredits | None = None
    usage_url: str | None = None
    unavailable: LimitsUnavailable | None = None

    def __post_init__(self) -> None:
        """Require windows to be unique, and absent when the read failed."""
        ids = [window.id for window in self.windows]
        if len(set(ids)) != len(ids):
            msg = f"limit windows of {self.provider!r} repeat an id: {sorted(ids)}"
            raise ValueError(msg)
        if self.unavailable is not None and (self.windows or self.reset_credits is not None):
            msg = "an unavailable read carries no windows or reset credits"
            raise ValueError(msg)

    @classmethod
    def failed(
        cls, provider: str, reason: LimitsUnavailableReason, detail: str = ""
    ) -> AccountLimits:
        """A read that produced nothing, for *reason*."""
        return cls(provider=provider, unavailable=LimitsUnavailable(reason, detail))


class ResetOutcome(str, Enum):
    """How the provider answered a request to spend a reset credit."""

    RESET = "reset"
    NOTHING_TO_RESET = "nothing_to_reset"
    NO_CREDIT = "no_credit"
    ALREADY_REDEEMED = "already_redeemed"


@dataclass(frozen=True)
class ResetCreditResult:
    """What spending a reset credit did, and the limits read after it."""

    outcome: ResetOutcome
    limits: AccountLimits


@dataclass(frozen=True)
class LimitsSession:
    """Where and how one limits read runs: the same inputs a readiness probe takes.

    ``timeout`` bounds the whole read, from starting the CLI to its answer.
    ``clock`` measures it, so a simulation controls it.
    """

    executor: CommandExecutor
    env: Mapping[str, str]
    timeout: float
    clock: Clock


class LimitsReader(Protocol):
    """A provider's way of reading its account limits. Provider knowledge lives here."""

    def read(self, session: LimitsSession) -> AccountLimits:
        """Read the limits now, or say why they could not be read. Never raises for the CLI."""
        ...

    def spend_reset_credit(
        self, session: LimitsSession, *, idempotency_key: str, credit_id: str | None
    ) -> ResetOutcome:
        """Spend one reset credit: *credit_id*, or the provider's next one when ``None``.

        *idempotency_key* names one logical attempt: spending twice with the same key
        spends one credit. Raises ``ProviderCapabilityError`` when the provider has no
        reset credits, ``CliNotFoundError`` when its CLI is not installed, and
        ``TurnFailedError`` or ``TurnTimeoutError`` when the call did not settle.
        """
        ...

    def window(self, status: RateLimitStatus) -> LimitWindow | None:
        """The window a turn's ``RateLimitStatus`` describes, in the shape a read returns."""
        ...
