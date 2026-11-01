"""Claude Code's plan limits, read on request and from a turn's ``rate_limit_event``.

On request, the CLI answers a ``get_usage`` control request over ``stream-json``
without a prompt and without a model call: it asks the claude.ai usage endpoint
for the plan's windows. The Agent SDK exposes it as
``usage_EXPERIMENTAL_MAY_CHANGE_DO_NOT_RELY_ON_THIS_API_YET``, and the CLI says
the shape may change, so the reader is tolerant: it reads only the keys the
SDK documents (``five_hour``, ``seven_day``, ``model_scoped`` and the older
``seven_day_opus`` and ``seven_day_sonnet``), ignores every other key, and
skips a window that states no share used instead of guessing one.

The wire was verified against CLI 2.1.304 and 2.1.316 (see
``tests/fixtures/claude_limits``). ``utilization`` is a percentage, 0 to 100; a
turn's ``rate_limit_event`` carries a fraction, 0 to 1, which
:mod:`agentshim.providers.claude.rate_limits` already reads. The account comes
from the reply to ``initialize`` (``account.email``), which is why the reader
sends it first.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re
from typing import TYPE_CHECKING, Any, cast

from agentshim.core.errors import (
    AgentShimError,
    CliExitError,
    FailureKind,
    ProviderCapabilityError,
    TurnFailedError,
    TurnTimeoutError,
)
from agentshim.core.limits import (
    SESSION_MINUTES,
    WEEKLY_MINUTES,
    AccountLimits,
    LimitsUnavailable,
    LimitsUnavailableReason,
    LimitWindow,
    WindowKind,
    account_digest,
    kind_of_minutes,
    label_of_kind,
)
from agentshim.core.stream import parse_json_object
from agentshim.core.transcripts import parse_timestamp
from agentshim.execution.process import ProcessExited, SpawnRequest, StderrLine

from .provider import STRICT_MCP_FLAG

if TYPE_CHECKING:
    from collections.abc import Mapping

    from agentshim.core.events import RateLimitStatus
    from agentshim.core.limits import LimitsSession, ResetOutcome
    from agentshim.execution.process import Process

PROVIDER = "claude"
BINARY = "claude"
USAGE_URL = "https://claude.ai/settings/usage"

#: How often the read looks at the clock while the CLI has nothing to say.
POLL_S = 0.05
#: Each step of stopping the CLI (end of input, terminate, kill) gets this long.
CLOSE_GRACE_S = 2.0
#: Stderr lines kept to explain a CLI that exits before answering.
STDERR_TAIL_LINES = 20

#: Frames that can only come from a model turn. A limits read sends no prompt, so seeing
#: one means something else started a turn, and the read stops rather than spend tokens.
_TURN_FRAMES = frozenset({"assistant", "stream_event", "result"})

#: The windows with a fixed name: the provider's key (also the id it gets), its kind and length.
_FIXED = (
    ("five_hour", WindowKind.SESSION, SESSION_MINUTES),
    ("seven_day", WindowKind.WEEKLY, WEEKLY_MINUTES),
)
#: Older per-model weekly keys, used only when ``model_scoped`` does not list the model.
_LEGACY_MODELS = (("seven_day_opus", "Opus"), ("seven_day_sonnet", "Sonnet"))
_SCOPED_PREFIX = "seven_day_"
_NOT_SLUG = re.compile(r"[^a-z0-9]+")


class ClaudeLimitsReader:
    """``LimitsReader`` for Claude Code. It has no reset credits."""

    def read(self, session: LimitsSession) -> AccountLimits:
        """Ask the CLI for the plan's windows, or say why it could not."""
        try:
            account, usage = _ask(session)
        except (AgentShimError, OSError) as error:
            return AccountLimits.failed(PROVIDER, LimitsUnavailableReason.PROBE_FAILED, str(error))
        return parse_usage(account, usage)

    def spend_reset_credit(
        self, session: LimitsSession, *, idempotency_key: str, credit_id: str | None
    ) -> ResetOutcome:
        """Claude has no reset credits to spend."""
        del session, idempotency_key, credit_id
        msg = "claude has no reset credits"
        raise ProviderCapabilityError(msg)

    def window(self, status: RateLimitStatus) -> LimitWindow | None:
        """The window a ``rate_limit_event`` status describes, named as a read names it.

        A status that states no share used is ``None``, unless it says the limit is
        reached, which is a share of 1.
        """
        name = status.window
        if not name:
            return None
        used = status.used_fraction
        if used is None:
            if not status.exhausted:
                return None
            used = 1.0
        if not math.isfinite(used) or used < 0:
            return None
        kind, scope, minutes = _describe(name, status.window_minutes)
        return LimitWindow(
            id=name,
            kind=kind,
            label=label_of_kind(kind, scope),
            used_fraction=used,
            resets_at=_from_epoch(status.resets_at),
            window_minutes=minutes,
        )


def _describe(name: str, reported_minutes: int | None) -> tuple[WindowKind, str | None, int | None]:
    """The kind, label scope and length of the window a turn's event calls *name*."""
    for key, kind, minutes in _FIXED:
        if name == key:
            return kind, None, reported_minutes or minutes
    for key, model in _LEGACY_MODELS:
        if name == key:
            return WindowKind.WEEKLY, model, reported_minutes or WEEKLY_MINUTES
    kind = kind_of_minutes(reported_minutes)
    scope = name.replace("_", " ").capitalize() if kind is WindowKind.OTHER else None
    return kind, scope, reported_minutes


def parse_usage(account: Mapping[str, Any] | None, usage: Mapping[str, Any]) -> AccountLimits:
    """Read the ``response`` of ``get_usage``, with the ``account`` of ``initialize``.

    A reply that says plan limits do not apply is ``API_KEY`` (an API key, a cloud provider
    and a login without profile scope all read that way); the detail says which the CLI
    reported. Any key the reply lacks leaves that window out.
    """
    digest = _digest(account)
    plan = _text(usage.get("subscription_type"))
    limits = _mapping(usage.get("rate_limits"))
    if usage.get("rate_limits_available") is False or limits is None:
        provider = _text((account or {}).get("apiProvider")) or "unknown"
        detail = f"plan limits do not apply to this login (api provider: {provider})"
        return AccountLimits(
            provider=PROVIDER,
            account=digest,
            plan=plan,
            unavailable=LimitsUnavailable(LimitsUnavailableReason.API_KEY, detail),
        )
    windows: dict[str, LimitWindow] = {}
    for key, kind, minutes in _FIXED:
        _add(windows, key, kind, None, _mapping(limits.get(key)), minutes)
    for entry in _list(limits.get("model_scoped")):
        scoped = _mapping(entry)
        name = _text(scoped.get("display_name")) if scoped else None
        if scoped is not None and name:
            slug = _NOT_SLUG.sub("_", name.lower()).strip("_")
            _add(
                windows, f"{_SCOPED_PREFIX}{slug}", WindowKind.WEEKLY, name, scoped, WEEKLY_MINUTES
            )
    for key, name in _LEGACY_MODELS:
        if key not in windows:
            _add(windows, key, WindowKind.WEEKLY, name, _mapping(limits.get(key)), WEEKLY_MINUTES)
    return AccountLimits(
        provider=PROVIDER,
        windows=tuple(windows.values()),
        account=digest,
        plan=plan,
        usage_url=USAGE_URL,
    )


def _add(  # noqa: PLR0913 - one window's name, kind, label scope, wire object and length
    windows: dict[str, LimitWindow],
    key: str,
    kind: WindowKind,
    scope: str | None,
    wire: Mapping[str, Any] | None,
    minutes: int,
) -> None:
    """Add the window *wire* describes, unless it states no usable share used."""
    if wire is None:
        return
    percent = wire.get("utilization")
    if isinstance(percent, bool) or not isinstance(percent, (int, float)):
        return
    if not math.isfinite(percent) or percent < 0:
        return
    windows[key] = LimitWindow(
        id=key,
        kind=kind,
        label=label_of_kind(kind, scope),
        used_fraction=percent / 100,
        resets_at=parse_timestamp(wire.get("resets_at")),
        window_minutes=minutes,
    )


def _digest(account: Mapping[str, Any] | None) -> str | None:
    email = _text((account or {}).get("email"))
    return None if email is None else account_digest(PROVIDER, email)


def _from_epoch(seconds: float | None) -> dt.datetime | None:
    if seconds is None:
        return None
    try:
        return dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


# -- the conversation with the CLI


def limits_argv(binary: str) -> list[str]:
    """The command line of a process that answers control requests and does nothing else.

    No prompt, no session file, no MCP servers, no hooks, and only the user's settings
    (not the working directory's), so the read has no side effect of its own and
    starts as little as the CLI allows.
    """
    return [
        binary,
        "--output-format",
        "stream-json",
        "--verbose",
        "--input-format",
        "stream-json",
        "--permission-prompts",
        "none",
        "--no-session-persistence",
        "--setting-sources",
        "user",
        STRICT_MCP_FLAG,
        "--mcp-config",
        json.dumps({"mcpServers": {}}),
        "--settings",
        json.dumps({"disableAllHooks": True}),
    ]


def _ask(session: LimitsSession) -> tuple[Mapping[str, Any] | None, Mapping[str, Any]]:
    """Send ``initialize`` then ``get_usage`` to a fresh CLI; return the account and the usage."""
    binary = session.executor.find_binary(BINARY, session.env)
    env = {name: value for name, value in session.env.items() if name != "CLAUDECODE"}
    process = session.executor.spawn(SpawnRequest(argv=limits_argv(binary), cwd=None, env=env))
    conversation = _Conversation(process, session)
    try:
        initialized = conversation.request("init", {"subtype": "initialize"})
        usage = conversation.request("usage", {"subtype": "get_usage", "skip_behaviors": True})
    finally:
        _stop(process)
    return _mapping(initialized.get("account")), usage


class _Conversation:
    """Control requests to one CLI process, each answered by id, within one deadline."""

    def __init__(self, process: Process, session: LimitsSession) -> None:
        self._process = process
        self._clock = session.clock
        self._timeout = session.timeout
        self._deadline = session.clock.monotonic() + session.timeout
        self._stderr: list[str] = []

    def request(self, request_id: str, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """Send *request* and return the ``response`` body of its success reply."""
        line = {"type": "control_request", "request_id": request_id, "request": dict(request)}
        self._process.write(json.dumps(line, separators=(",", ":")) + "\n")
        subtype = request["subtype"]
        while True:
            item = self._process.next_output(0.0)
            if item is None:
                if self._clock.monotonic() >= self._deadline:
                    msg = f"claude did not answer {subtype} within {self._timeout}s"
                    raise TurnTimeoutError(self._timeout, msg)
                self._clock.wait(POLL_S)
            elif isinstance(item, StderrLine):
                text = item.text.rstrip("\n")
                if text:
                    self._stderr = [*self._stderr, text][-STDERR_TAIL_LINES:]
            elif isinstance(item, ProcessExited):
                raise self._exited(subtype, item.returncode)
            else:
                reply = self._reply(subtype, request_id, item.text)
                if reply is not None:
                    return reply

    def _reply(self, subtype: str, request_id: str, text: str) -> Mapping[str, Any] | None:
        data = parse_json_object(text)
        if data is None:
            return None
        kind = data.get("type")
        if kind in _TURN_FRAMES:
            msg = f"claude started a model turn during a limits read (a {kind!r} frame)"
            raise TurnFailedError(msg, kind=FailureKind.OTHER, detail=msg)
        if kind != "control_response":
            return None
        response = _mapping(data.get("response"))
        if response is None or response.get("request_id") != request_id:
            return None
        if response.get("subtype") == "success":
            return _mapping(response.get("response")) or {}
        detail = str(response.get("error") or f"{subtype} failed")
        msg = f"claude refused {subtype}: {detail}"
        raise TurnFailedError(msg, kind=FailureKind.OTHER, detail=detail)

    def _exited(self, subtype: str, returncode: int) -> AgentShimError:
        stderr = "\n".join(self._stderr)
        detail = stderr or f"claude exited with code {returncode} before answering {subtype}"
        return CliExitError(
            (BINARY,), returncode, "", stderr, kind=FailureKind.OTHER, detail=detail
        )


def _stop(process: Process) -> None:
    """End the CLI: end of input, then terminate, then kill, each bounded."""
    process.close_stdin()
    if process.wait(CLOSE_GRACE_S) is not None:
        return
    process.terminate()
    if process.wait(CLOSE_GRACE_S) is not None:
        return
    process.kill()
    process.wait(CLOSE_GRACE_S)


def _mapping(value: object) -> Mapping[str, Any] | None:
    return cast("Mapping[str, Any]", value) if isinstance(value, dict) else None


def _list(value: object) -> list[object]:
    return cast("list[object]", value) if isinstance(value, list) else []


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
