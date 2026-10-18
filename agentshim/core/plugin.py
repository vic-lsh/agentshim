"""Method-based contract implemented by built-in and third-party plugins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import datetime as dt
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from agentshim.execution.executor import CommandExecutor, CommandRequest

    from .clock import Clock
    from .conversation import Transport, TransportKind
    from .errors import FailureKind
    from .ids import IdAllocator
    from .models import ModelSpec, ModelTable
    from .pricing import ModelPricing, PricingTable
    from .profile import ProviderProfile
    from .provider import Provider
    from .status import AuthState, ProviderStatus
    from .usage import TokenUsage


class ScriptedLines(Protocol):
    """Build one turn of a provider's real stdout for consumer tests."""

    def __call__(  # noqa: PLR0913
        self,
        *,
        text: str = "",
        session_id: str | None = None,
        usage: TokenUsage | None = None,
        tool_calls: Sequence[tuple[str, Mapping[str, Any], str]] = (),
        structured_output: object | None = None,
        skills_offered: Sequence[str] | None = None,
        skills_invoked: Sequence[str] = (),
    ) -> list[str]:
        """Return stdout lines in this provider's stream format."""
        ...


class ResumeFailureLines(Protocol):
    """Build the stream and exit status of a failed resume."""

    def __call__(
        self, *, session_id: str | None = None
    ) -> tuple[Sequence[str], Sequence[str], int]:
        """Return stdout, stderr, and exit status."""
        ...


class FailureLines(Protocol):
    """Build the stream and exit status of a classified failure."""

    def __call__(
        self, kind: FailureKind, *, session_id: str | None = None
    ) -> tuple[Sequence[str], Sequence[str], int]:
        """Return stdout, stderr, and exit status."""
        ...


class StreamTransportFactory(Protocol):
    """Legacy type for provider-internal stream constructors."""

    def __call__(  # noqa: PLR0913
        self,
        *,
        executor: CommandExecutor,
        env: Mapping[str, str] | None,
        clock: Clock,
        ids: IdAllocator,
        check_timeout: float,
        log: Callable[[str], None] | None,
    ) -> Transport:
        """Create a stream transport."""
        ...


class ProbeFixture(Protocol):
    """Build fake CLI output for a readiness command."""

    def __call__(
        self, args: Sequence[str], *, version: str, auth: AuthState
    ) -> tuple[str, str, int]:
        """Return stdout, stderr, and exit status."""
        ...


@dataclass(frozen=True)
class ProviderTestSupport:
    """Optional provider-owned adapters for the generic test helpers."""

    scripted_lines: ScriptedLines
    resume_failure_lines: ResumeFailureLines
    failure_lines: FailureLines | None = None
    installed_mcp_servers: Callable[[CommandRequest, Path], dict[str, dict[str, Any]]] | None = None
    probe_fixture: ProbeFixture | None = None


@dataclass(frozen=True)
class ConnectionOptions:
    """Dependencies for a requested connection mode."""

    kind: TransportKind
    executor: CommandExecutor | None = None
    env: Mapping[str, str] | None = None
    clock: Clock | None = None
    ids: IdAllocator | None = None
    check_timeout: float = 15.0
    log: Callable[[str], None] | None = None


@dataclass(frozen=True)
class PricingContext:
    """Conditions under which a rate is requested; omitted values mean standard tier."""

    as_of: dt.date | None = None
    region: str | None = None
    tier: str | None = None


@runtime_checkable
class ProviderPlugin(Protocol):
    """Required methods for every provider; unsupported facts return ``None``."""

    @property
    def name(self) -> str:
        """Stable catalog key."""
        ...

    def describe(self) -> ProviderProfile:
        """Describe capabilities independently of local installation state."""
        ...

    def connection_modes(self) -> frozenset[TransportKind]:
        """Declare which connection modes ``connect`` accepts."""
        ...

    def connect(self, options: ConnectionOptions) -> Provider | Transport:
        """Create the requested one-shot provider or stream transport."""
        ...

    def probe(
        self, executor: CommandExecutor, env: Mapping[str, str], timeout: float
    ) -> ProviderStatus:
        """Check readiness on the supplied executor, without running a model."""
        ...

    def model_info(self, model: str, table: ModelTable | None = None) -> ModelSpec | None:
        """Resolve context-window facts; *table* overrides built-in facts."""
        ...

    def pricing_info(
        self, model: str, context: PricingContext, table: PricingTable | None = None
    ) -> ModelPricing | None:
        """Resolve a rate for *model* and *context*, or report it unknown."""
        ...

    def known_models(self) -> tuple[ModelSpec, ...]:
        """Enumerate facts available for a reproducible table snapshot."""
        ...

    def known_prices(self) -> tuple[ModelPricing, ...]:
        """Enumerate standard-tier prices for a reproducible table snapshot."""
        ...

    def test_support(self) -> ProviderTestSupport | None:
        """Return consumer-test adapters, if provided."""
        ...
