"""Codex's account limits, read on request over ``codex app-server``.

``account/rateLimits/read`` returns the account's buckets and its reset credits.
It needs the ``initialize`` handshake and nothing else: no thread, no turn, no
model call. It is a stable method of CLI 0.160.0 (it works without the
``experimentalApi`` capability), and so is ``account/rateLimitResetCredit/consume``,
which spends one reset credit. The two are separate methods of
:class:`CodexLimitsReader` so a caller reads without any path to spending.

A bucket is a ``RateLimitSnapshot`` with a ``primary`` and a ``secondary`` window,
told apart by their length, not their position: a Free or Go plan reports a
monthly window, a Pro plan a weekly one. ``rateLimitsByLimitId`` is the full
multi-bucket answer; ``rateLimits`` is the older single bucket, used when the
other is absent. Window ids are ``<limit id>:<primary|secondary>``, the same
words a turn's ``account/rateLimits/updated`` notification carries, so a reading
from a turn and a read merge by id.
"""

from __future__ import annotations

import datetime as dt
from enum import Enum
from importlib import metadata
from typing import TYPE_CHECKING

from agentshim.core.errors import (
    AgentShimError,
    CliExitError,
    FailureKind,
    TurnFailedError,
    TurnTimeoutError,
)
from agentshim.core.limits import (
    AccountLimits,
    LimitsUnavailableReason,
    LimitWindow,
    ResetCredit,
    ResetCredits,
    ResetOutcome,
    account_digest,
    kind_of_minutes,
    label_of_kind,
)
from agentshim.execution.process import SpawnRequest

from .app_server.channel import Channel, Deadline, Expired, Gone
from .app_server.protocol import (
    INITIALIZED,
    ClientInfo,
    ConsumeAccountRateLimitResetCreditOutcome,
    ConsumeAccountRateLimitResetCreditParams,
    ErrorResponse,
    GetAccountRateLimitsParams,
    InitializeParams,
    Notification,
    RateLimitResetCreditStatus,
    Response,
)

if TYPE_CHECKING:
    from agentshim.core.events import RateLimitStatus
    from agentshim.core.limits import LimitsSession
    from agentshim.execution.process import Process

    from .app_server.protocol import (
        ClientRequestParams,
        GetAccountRateLimitsResponse,
        RateLimitResetCreditsSummary,
        RateLimitSnapshot,
    )

PROVIDER = "codex"
BINARY = "codex"
USAGE_URL = "https://chatgpt.com/codex/settings/usage"
#: The bucket every account has, and the one a snapshot with no id is taken to be.
DEFAULT_LIMIT = "codex"

#: Each step of stopping the server (end of input, terminate, kill) gets this long.
CLOSE_GRACE_S = 2.0

#: Notification prefixes that only a running thread produces. A limits read starts none,
#: so seeing one means something else did, and the read stops.
_THREAD_NOTIFICATIONS = ("turn/", "item/", "thread/")

#: The server's outcome word, as :class:`ResetOutcome`.
_OUTCOMES = {
    ConsumeAccountRateLimitResetCreditOutcome.RESET.value: ResetOutcome.RESET,
    ConsumeAccountRateLimitResetCreditOutcome.NOTHING_TO_RESET.value: ResetOutcome.NOTHING_TO_RESET,
    ConsumeAccountRateLimitResetCreditOutcome.NO_CREDIT.value: ResetOutcome.NO_CREDIT,
    ConsumeAccountRateLimitResetCreditOutcome.ALREADY_REDEEMED.value: ResetOutcome.ALREADY_REDEEMED,
}


class CodexLimitsReader:
    """``LimitsReader`` for Codex."""

    def read(self, session: LimitsSession) -> AccountLimits:
        """Ask ``codex app-server`` for the account's limits, or say why it could not."""
        try:
            with _AppServer.start(session) as server:
                result = server.call(GetAccountRateLimitsParams())
            response = GetAccountRateLimitsParams.parse_result(result)
        except (AgentShimError, OSError) as error:
            return AccountLimits.failed(PROVIDER, LimitsUnavailableReason.PROBE_FAILED, str(error))
        return parse_rate_limits(response)

    def spend_reset_credit(
        self, session: LimitsSession, *, idempotency_key: str, credit_id: str | None
    ) -> ResetOutcome:
        """Spend one reset credit and report what the server did.

        An outcome this library does not know is an error, never a guess: the
        credit may or may not be spent, and the caller must find out by reading.
        """
        params = ConsumeAccountRateLimitResetCreditParams(
            idempotency_key=idempotency_key, credit_id=credit_id
        )
        with _AppServer.start(session) as server:
            result = server.call(params)
        response = ConsumeAccountRateLimitResetCreditParams.parse_result(result)
        word = _word(response.outcome)
        outcome = _OUTCOMES.get(word)
        if outcome is None:
            msg = (
                f"codex answered an unknown reset outcome {word!r}; "
                "the credit may or may not be spent, so read the limits to find out"
            )
            raise TurnFailedError(msg, kind=FailureKind.OTHER, detail=msg)
        return outcome

    def window(self, status: RateLimitStatus) -> LimitWindow | None:
        """The window an ``account/rateLimits/updated`` status describes, named as a read names it.

        A status with no window (it only says a limit was reached) is ``None``.
        """
        used = status.used_fraction
        if status.window not in ("primary", "secondary") or used is None or used < 0:
            return None
        limit = status.limit or DEFAULT_LIMIT
        kind = kind_of_minutes(status.window_minutes)
        scope = None if limit == DEFAULT_LIMIT else limit
        return LimitWindow(
            id=f"{limit}:{status.window}",
            kind=kind,
            label=label_of_kind(kind, scope),
            used_fraction=used,
            resets_at=_from_epoch(status.resets_at),
            window_minutes=status.window_minutes,
        )


def parse_rate_limits(response: GetAccountRateLimitsResponse) -> AccountLimits:
    """Read an ``account/rateLimits/read`` result."""
    buckets = response.rate_limits_by_limit_id or {DEFAULT_LIMIT: response.rate_limits}
    windows: dict[str, LimitWindow] = {}
    for key, snapshot in buckets.items():
        for window in _bucket_windows(snapshot, key):
            windows[window.id] = window
    plan = response.rate_limits.plan_type
    return AccountLimits(
        provider=PROVIDER,
        windows=tuple(windows.values()),
        account=account_digest(PROVIDER, response.account_id) if response.account_id else None,
        plan=None if plan is None else _word(plan),
        reset_credits=_reset_credits(response.rate_limit_reset_credits),
        usage_url=USAGE_URL,
    )


def _bucket_windows(snapshot: RateLimitSnapshot, key: str) -> list[LimitWindow]:
    limit = snapshot.limit_id or key
    scope = None if limit == DEFAULT_LIMIT else (snapshot.limit_name or limit)
    out: list[LimitWindow] = []
    for name, window in (("primary", snapshot.primary), ("secondary", snapshot.secondary)):
        if window is None:
            continue
        kind = kind_of_minutes(window.window_duration_mins)
        out.append(
            LimitWindow(
                id=f"{limit}:{name}",
                kind=kind,
                label=label_of_kind(kind, scope),
                used_fraction=max(window.used_percent, 0) / 100,
                resets_at=_from_epoch(window.resets_at),
                window_minutes=window.window_duration_mins,
            )
        )
    return out


def _reset_credits(summary: RateLimitResetCreditsSummary | None) -> ResetCredits | None:
    if summary is None:
        return None
    described = [
        ResetCredit(
            id=credit.id,
            granted_at=_aware(credit.granted_at),
            expires_at=None if credit.expires_at is None else _aware(credit.expires_at),
            title=credit.title,
            description=credit.description,
        )
        for credit in summary.credits or ()
        if _word(credit.status) == RateLimitResetCreditStatus.AVAILABLE.value
    ]
    return ResetCredits(available=max(summary.available_count, 0), credits=tuple(described))


def _aware(seconds: int) -> dt.datetime:
    return dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc)


def _from_epoch(seconds: float | None) -> dt.datetime | None:
    if seconds is None:
        return None
    try:
        return dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _word(value: object) -> str:
    return value.value if isinstance(value, Enum) else str(value)


# -- the conversation with the server


class _AppServer:
    """One ``codex app-server`` process, past its handshake, for the length of one call."""

    def __init__(self, process: Process, session: LimitsSession) -> None:
        self._channel = Channel(process, lambda _event: None)
        self._deadline = Deadline(session.clock, session.timeout)
        self._timeout = session.timeout

    @classmethod
    def start(cls, session: LimitsSession) -> _AppServer:
        """Spawn the server and finish its handshake, or stop it and say why not."""
        binary = session.executor.find_binary(BINARY, session.env)
        request = SpawnRequest(argv=[binary, "app-server"], cwd=None, env=dict(session.env))
        server = cls(session.executor.spawn(request), session)
        try:
            server.call(
                InitializeParams(client_info=ClientInfo(name="agentshim", version=_version()))
            )
            server.notify_initialized()
        except BaseException:
            server.close()
            raise
        return server

    def __enter__(self) -> _AppServer:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def notify_initialized(self) -> None:
        """Finish the handshake."""
        try:
            self._channel.notify(INITIALIZED)
        except Gone as gone:
            raise self._exited(gone) from None

    def call(self, params: ClientRequestParams) -> object:
        """Send one request and return its result, or raise why it did not come."""
        try:
            request_id = self._channel.request(params)
            while True:
                message = self._channel.read(self._deadline)
                if isinstance(message, (Response, ErrorResponse)) and message.id == request_id:
                    self._channel.settle(request_id)
                    if isinstance(message, ErrorResponse):
                        text = message.error.message
                        msg = f"codex {params.METHOD} failed: {text}"
                        raise TurnFailedError(msg, kind=FailureKind.OTHER, detail=text)
                    return message.result
                if isinstance(message, Notification) and message.method.startswith(
                    _THREAD_NOTIFICATIONS
                ):
                    msg = f"codex started a thread during a limits read ({message.method})"
                    raise TurnFailedError(msg, kind=FailureKind.OTHER, detail=msg)
        except Expired:
            msg = f"codex did not answer {params.METHOD} within {self._timeout}s"
            raise TurnTimeoutError(self._timeout, msg) from None
        except Gone as gone:
            raise self._exited(gone) from None

    def _exited(self, gone: Gone) -> AgentShimError:
        self._channel.drain()
        stderr = self._channel.stderr_tail()
        detail = stderr or f"codex app-server exited with code {gone.returncode}"
        code = gone.returncode if gone.returncode is not None else -1
        return CliExitError(
            (BINARY, "app-server"), code, "", stderr, kind=FailureKind.OTHER, detail=detail
        )

    def close(self) -> None:
        """End the server: end of input, then terminate, then kill, each bounded."""
        process = self._channel.process
        process.close_stdin()
        if process.wait(CLOSE_GRACE_S) is not None:
            return
        process.terminate()
        if process.wait(CLOSE_GRACE_S) is not None:
            return
        process.kill()
        process.wait(CLOSE_GRACE_S)


def _version() -> str:
    try:
        return metadata.version("agentshim")
    except metadata.PackageNotFoundError:  # pragma: no cover - always installed in a checkout
        return "0"
