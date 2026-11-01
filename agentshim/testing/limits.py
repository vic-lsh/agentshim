"""Fake accounts for limits reads: what ``claude`` and ``codex app-server`` say about a plan.

Both fakes plug into the process fakes a test already has, so ``read_limits`` and
``use_reset_credit`` run their real code against a stand-in for the CLI::

    peers = ClaudeStreamPeers(account=claude_account(), usage=claude_usage(five_hour=(39, RESET)))
    executor = FakeExecutor([], peers=peers.build)
    limits = read_limits("claude", executor=executor, env={"PATH": "/bin"})

    script = CodexScript(account=CodexAccount(credits=("credit-1",)))
    executor = FakeExecutor([], peers=script.peer)
    result = use_reset_credit("codex", idempotency_key="k", executor=executor, env={})

``claude_usage`` and ``claude_account`` build the CLI's replies in the shape the real
CLI sent, including keys agentshim ignores, so a reader that depends on an
undocumented key fails here. ``CodexAccount`` is stateful: it spends the credits it
holds and remembers each idempotency key, as the server does, so a test can check
that a retry spends once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

#: A usage window as the tests spell it: share used (a percentage, 0 to 100) and the ISO reset time.
UsageWindow = tuple[float | None, str | None]


def claude_account(
    *,
    email: str | None = "person@example.com",
    subscription: str = "Claude Team",
    api_provider: str = "firstParty",
) -> dict[str, Any]:
    """The ``account`` object in the reply to Claude Code's ``initialize``."""
    account: dict[str, Any] = {"subscriptionType": subscription, "apiProvider": api_provider}
    if email is not None:
        account["email"] = email
    return account


def claude_usage(  # noqa: PLR0913 - one keyword per window the reply can hold
    *,
    five_hour: UsageWindow | None = None,
    seven_day: UsageWindow | None = None,
    model_scoped: Sequence[tuple[str, float | None, str | None]] = (),
    seven_day_opus: UsageWindow | None = None,
    seven_day_sonnet: UsageWindow | None = None,
    plan: str | None = "team",
    available: bool = True,
) -> dict[str, Any]:
    """The ``response`` of Claude Code's ``get_usage`` control request.

    Windows left ``None`` are ``null`` in the reply, as for an account that has no such
    window. ``available=False`` is the reply for a login that has no plan limits
    (an API key): ``rate_limits_available`` false and ``rate_limits`` null.
    """

    def window(spec: UsageWindow | None) -> dict[str, Any] | None:
        if spec is None:
            return None
        used, resets = spec
        return {
            "utilization": used,
            "resets_at": resets,
            "limit_dollars": None,
            "used_dollars": None,
            "remaining_dollars": None,
            "locked_reason": None,
        }

    limits: dict[str, Any] | None = None
    if available:
        limits = {
            "five_hour": window(five_hour),
            "seven_day": window(seven_day),
            "seven_day_oauth_apps": None,
            "seven_day_opus": window(seven_day_opus),
            "seven_day_sonnet": window(seven_day_sonnet),
            "extra_usage": {"is_enabled": False, "monthly_limit": None, "used_credits": None},
            "limits": [{"kind": "session", "group": "session", "percent": 1, "is_active": True}],
            "spend": {"used": {"amount_minor": 0, "currency": "USD", "exponent": 2}},
            "model_scoped": [
                {"display_name": name, "utilization": used, "resets_at": resets}
                for name, used, resets in model_scoped
            ],
            "pace": None,
        }
    return {
        "session": {"total_cost_usd": 0, "model_usage": {}},
        "subscription_type": plan if available else None,
        "rate_limits_available": available,
        "rate_limits": limits,
        "behaviors": None,
    }


@dataclass(frozen=True)
class CodexWindow:
    """One window of a Codex bucket: the share used and its length and refill time."""

    used_percent: int
    minutes: int | None = None
    resets_at: int | None = None

    def wire(self, *, used_percent: int | None = None) -> dict[str, Any]:
        """The ``RateLimitWindow`` object."""
        used = self.used_percent if used_percent is None else used_percent
        return {"usedPercent": used, "windowDurationMins": self.minutes, "resetsAt": self.resets_at}


_RESET_AT = 1_794_082_513


def _no_outcomes() -> dict[str, str]:
    return {}


def _no_credits() -> list[str]:
    return []


def _weekly_bucket() -> dict[str, tuple[CodexWindow | None, CodexWindow | None]]:
    return {"codex": (CodexWindow(1, 10_080, _RESET_AT), None)}


@dataclass
class CodexAccount:
    """The account behind a fake ``codex app-server``: its limits and its reset credits.

    ``buckets`` maps a limit id to its ``(primary, secondary)`` windows; the default is
    the single weekly bucket a Pro plan reports. ``credits`` are the ids of the reset
    credits it holds. ``read_error`` makes ``account/rateLimits/read`` fail with that
    message. ``outcomes`` holds the answer given to each idempotency key already used;
    ``spent`` lists the credits spent, in order. Spending a credit empties every window.
    """

    plan: str | None = "pro"
    account_id: str | None = "account-1"
    buckets: dict[str, tuple[CodexWindow | None, CodexWindow | None]] = field(
        default_factory=_weekly_bucket
    )
    credits: tuple[str, ...] = ()
    read_error: str | None = None
    outcomes: dict[str, str] = field(default_factory=_no_outcomes)
    spent: list[str] = field(default_factory=_no_credits)

    def rate_limits(self, *, credit_details: bool = True) -> dict[str, Any]:
        """The result of ``account/rateLimits/read``."""
        emptied = 0 if self.spent else None

        def snapshot(
            limit: str, windows: tuple[CodexWindow | None, CodexWindow | None]
        ) -> dict[str, Any]:
            primary, secondary = windows
            return {
                "limitId": limit,
                "limitName": None if limit == "codex" else limit.title(),
                "normalModelSlug": None,
                "primary": None if primary is None else primary.wire(used_percent=emptied),
                "secondary": None if secondary is None else secondary.wire(used_percent=emptied),
                "credits": {"hasCredits": True, "unlimited": False, "balance": "0"},
                "individualLimit": None,
                "spendControlReached": False,
                "planType": self.plan,
                "rateLimitReachedType": None,
            }

        by_id = {limit: snapshot(limit, windows) for limit, windows in self.buckets.items()}
        held = self.held()
        details = [
            {
                "id": credit,
                "resetType": "codexRateLimits",
                "status": "available",
                "grantedAt": _RESET_AT - 86_400,
                "expiresAt": _RESET_AT + 86_400,
                "title": "Reset",
                "description": None,
            }
            for credit in held
        ]
        return {
            "ordinaryUsageAllowed": True,
            "rateLimits": by_id.get("codex") or next(iter(by_id.values())),
            "rateLimitsByLimitId": by_id,
            "rateLimitResetCredits": {
                "availableCount": len(held),
                "credits": details if credit_details else None,
            },
            "accountId": self.account_id,
            "rateLimitUpsell": None,
        }

    def held(self) -> list[str]:
        """The credits not yet spent."""
        return [credit for credit in self.credits if credit not in self.spent]

    def consume(self, idempotency_key: str, credit_id: str | None) -> str:
        """Spend a credit for *idempotency_key*, once: a repeat of a key repeats its answer."""
        if idempotency_key in self.outcomes:
            return self.outcomes[idempotency_key]
        held = self.held()
        chosen = credit_id if credit_id is not None else (held[0] if held else None)
        if chosen is None or chosen not in held:
            outcome = "noCredit"
        else:
            self.spent.append(chosen)
            outcome = "reset"
        self.outcomes[idempotency_key] = outcome
        return outcome
