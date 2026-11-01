"""Read an account's plan limits on request: windows, share used, refill times, reset credits.

Composition layer, next to :mod:`agentshim.probe`: it turns a provider name into
the provider's limits reader and runs it where a turn would run. Nothing here
calls a model. A read starts the provider's CLI with no prompt, asks it for the
limits, and stops it.

* :func:`read_limits` reads the limits now. It never raises for a CLI problem:
  an answer with no windows and a reason (``AccountLimits.unavailable``) lets a
  caller keep its last good reading and say why it is old.
* :func:`limit_window` turns the ``RateLimitStatus`` a turn emitted into the
  window a read returns for it, so readings from turns and from reads merge by
  ``LimitWindow.id`` without the caller knowing a provider's window names.
* :func:`use_reset_credit` spends a reset credit. It is a separate function, never
  reached from a read, because it costs the account something that cannot be undone.

A provider that cannot be read (OpenCode, Gemini, Copilot) answers ``UNSUPPORTED``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentshim.core.clock import SystemClock
from agentshim.core.env import interactive_env
from agentshim.core.errors import ProviderCapabilityError
from agentshim.core.limits import (
    AccountLimits,
    LimitsReader,
    LimitsSession,
    LimitsUnavailable,
    LimitsUnavailableReason,
    LimitWindow,
    ResetCredit,
    ResetCreditResult,
    ResetCredits,
    ResetOutcome,
    WindowKind,
)
from agentshim.core.plugin import LimitsSource
from agentshim.execution.confinement import confine
from agentshim.execution.host import HostCommandExecutor
from agentshim.providers import DEFAULT_CATALOG, ProviderCatalog

if TYPE_CHECKING:
    from collections.abc import Mapping

    from agentshim.core.clock import Clock
    from agentshim.core.events import RateLimitStatus
    from agentshim.execution.confinement import Confinement
    from agentshim.execution.executor import CommandExecutor

__all__ = [
    "AccountLimits",
    "LimitWindow",
    "LimitsUnavailable",
    "LimitsUnavailableReason",
    "ResetCredit",
    "ResetCreditResult",
    "ResetCredits",
    "ResetOutcome",
    "WindowKind",
    "limit_providers",
    "limit_window",
    "read_limits",
    "use_reset_credit",
]


def limit_providers(*, catalog: ProviderCatalog = DEFAULT_CATALOG) -> tuple[str, ...]:
    """The providers in *catalog* whose limits can be read, in name order."""
    return tuple(name for name in catalog.names() if _reader(catalog, name) is not None)


def read_limits(  # noqa: PLR0913 - the same inputs ``probe_provider`` takes, plus the clock
    provider: str,
    *,
    executor: CommandExecutor | None = None,
    confinement: Confinement | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = 15.0,
    clock: Clock | None = None,
    catalog: ProviderCatalog = DEFAULT_CATALOG,
) -> AccountLimits:
    """Read *provider*'s account limits now, without running a turn.

    Arguments mean what they mean for :func:`~agentshim.probe.probe_provider`:
    *executor* defaults to the local host, *confinement* wraps it and supplies the
    environment (so *env* may not also be given), and *env* otherwise defaults to the
    login shell's environment. *timeout* bounds the whole read. *clock* measures it.

    A read that cannot be done is a result, not an error: ``unavailable`` says
    ``UNSUPPORTED`` (the provider has no limits read), ``API_KEY`` (no plan applies),
    or ``PROBE_FAILED`` (no CLI, not logged in, timed out, an error answer; the detail
    says which). An unknown *provider* raises ``ValueError``.
    """
    reader = _reader(catalog, provider)
    if reader is None:
        detail = f"provider {provider!r} has no limits read"
        return AccountLimits.failed(provider, LimitsUnavailableReason.UNSUPPORTED, detail)
    return reader.read(_session(executor, confinement, env, timeout, clock))


def use_reset_credit(  # noqa: PLR0913 - the read's inputs plus the credit and its key
    provider: str,
    *,
    idempotency_key: str,
    credit_id: str | None = None,
    executor: CommandExecutor | None = None,
    confinement: Confinement | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = 15.0,
    clock: Clock | None = None,
    catalog: ProviderCatalog = DEFAULT_CATALOG,
) -> ResetCreditResult:
    """Spend one of the account's reset credits, then read the limits again.

    This costs the account something real and cannot be undone. *credit_id* names the
    credit (``ResetCredit.id``); ``None`` lets the provider pick its next one.
    *idempotency_key* names one logical attempt: a caller that retries after an error it
    cannot read reuses the key, and the provider spends at most one credit for it. Make
    a new key (a UUID) for each separate decision to spend.

    *timeout* bounds the spend and the read after it separately. The result carries the
    provider's ``outcome`` and the limits read afterwards, which can differ from what
    the outcome promises: trust the read.

    Raises:
        ValueError: *provider* is not in the catalog, or *idempotency_key* is empty.
        ProviderCapabilityError: *provider* has no reset credits.
        CliNotFoundError, TurnTimeoutError, TurnFailedError: the call did not settle. The
            credit may or may not be spent: read the limits to find out, and retry with
            the same key if it is still wanted.
    """
    if not idempotency_key.strip():
        msg = "use_reset_credit needs an idempotency_key naming this attempt"
        raise ValueError(msg)
    reader = _reader(catalog, provider)
    if reader is None:
        msg = f"provider {provider!r} has no reset credits"
        raise ProviderCapabilityError(msg)
    session = _session(executor, confinement, env, timeout, clock)
    outcome = reader.spend_reset_credit(
        session, idempotency_key=idempotency_key, credit_id=credit_id
    )
    return ResetCreditResult(outcome=outcome, limits=reader.read(session))


def limit_window(
    provider: str, status: RateLimitStatus, *, catalog: ProviderCatalog = DEFAULT_CATALOG
) -> LimitWindow | None:
    """The window a turn's ``RateLimitStatus`` describes, in the shape :func:`read_limits` returns.

    ``None`` when the status names no window, states no share used and does not say the
    limit is reached, or when *provider* has no limits at all. An unknown *provider*
    raises ``ValueError``.
    """
    reader = _reader(catalog, provider)
    return None if reader is None else reader.window(status)


def _reader(catalog: ProviderCatalog, provider: str) -> LimitsReader | None:
    plugin = catalog.plugin(provider)
    return plugin.limits_reader() if isinstance(plugin, LimitsSource) else None


def _session(
    executor: CommandExecutor | None,
    confinement: Confinement | None,
    env: Mapping[str, str] | None,
    timeout: float,
    clock: Clock | None,
) -> LimitsSession:
    if timeout <= 0:
        msg = "timeout must be positive"
        raise ValueError(msg)
    base = executor if executor is not None else HostCommandExecutor()
    if confinement is not None:
        if env is not None:
            msg = "env conflicts with confinement: the confinement supplies the agent environment"
            raise ValueError(msg)
        base = confine(base, confinement)
        env = dict(confinement.env)
    return LimitsSession(
        executor=base,
        env=dict(env) if env is not None else interactive_env(),
        timeout=timeout,
        clock=clock if clock is not None else SystemClock(),
    )
