"""Provider contract: runtime factories, optional test adapters, and facts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from agentshim.execution.executor import CommandExecutor, CommandRequest

    from .clock import Clock
    from .conversation import Transport
    from .errors import FailureKind
    from .ids import IdAllocator
    from .models import ModelSpec
    from .pricing import ModelPricing
    from .profile import ProviderProfile
    from .provider import Provider
    from .status import AuthState, ProbeSpec
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
    """Build a provider's long-lived transport."""

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
        """Return a checked, ready-to-use transport."""
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
class ProviderPlugin:
    """The single declaration of a provider's runtime and optional test features."""

    name: str
    profile: ProviderProfile
    provider_factory: Callable[[], Provider] | None = None
    probe: ProbeSpec | None = None
    stream_factory: StreamTransportFactory | None = None
    testing: ProviderTestSupport | None = None
    #: Vendor for unprefixed model ids this provider runs, if known.
    model_vendor: str | None = None
    #: Vendor that bills this provider's token usage, if known.
    pricing_vendor: str | None = None
    #: Extra model-id prefixes this provider understands, such as google-vertex/google.
    model_aliases: tuple[tuple[str, str], ...] = ()
    #: Facts supplied by this plugin for model lookup.
    model_entries: tuple[ModelSpec, ...] = ()
    #: Facts supplied by this plugin for cost lookup.
    pricing_entries: tuple[ModelPricing, ...] = ()

    def __post_init__(self) -> None:
        """Reject unusable names and declarations that disagree with the provider."""
        if not self.name or self.name != self.name.strip():
            msg = "provider plugin name must be non-empty and unpadded"
            raise ValueError(msg)
        if self.profile.name != self.name:
            msg = f"plugin {self.name!r} has provider profile name {self.profile.name!r}"
            raise ValueError(msg)
        if self.provider_factory is None and self.stream_factory is None:
            msg = f"plugin {self.name!r} needs a one-shot or stream transport factory"
            raise ValueError(msg)
