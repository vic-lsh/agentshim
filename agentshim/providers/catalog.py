"""A composable catalog of independently defined provider integrations."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from agentshim.core.clock import Clock
    from agentshim.core.conversation import Transport
    from agentshim.core.errors import FailureKind
    from agentshim.core.ids import IdAllocator
    from agentshim.core.profile import ProviderProfile
    from agentshim.core.provider import Provider
    from agentshim.core.status import AuthState, ProbeSpec
    from agentshim.core.usage import TokenUsage
    from agentshim.execution.executor import CommandExecutor, CommandRequest


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


class ProviderCatalog:
    """An immutable, explicitly extended set of provider plugins."""

    def __init__(self, plugins: Sequence[ProviderPlugin] = ()) -> None:
        """Build a catalog; duplicate names fail rather than override another plugin."""
        entries: dict[str, ProviderPlugin] = {}
        for plugin in plugins:
            if plugin.name in entries:
                msg = f"duplicate provider plugin {plugin.name!r}"
                raise ValueError(msg)
            entries[plugin.name] = plugin
        self._plugins: Mapping[str, ProviderPlugin] = MappingProxyType(entries)

    def with_plugin(self, plugin: ProviderPlugin) -> ProviderCatalog:
        """Return a new catalog containing *plugin*, preserving this one."""
        return ProviderCatalog((*self._plugins.values(), plugin))

    def names(self) -> list[str]:
        """List providers in stable order."""
        return sorted(self._plugins)

    def stream_names(self) -> list[str]:
        """List providers with a long-lived transport."""
        return sorted(name for name, plugin in self._plugins.items() if plugin.stream_factory)

    def plugin(self, name: str) -> ProviderPlugin:
        """Find one plugin or report the names available in this catalog."""
        plugin = self._plugins.get(name)
        if plugin is None:
            msg = f"unknown provider {name!r}; available: {self.names()}"
            raise ValueError(msg)
        return plugin

    def provider(self, name: str) -> Provider:
        """Construct a one-shot provider by name."""
        factory = self.plugin(name).provider_factory
        if factory is None:
            msg = f"provider {name!r} has no one-shot transport"
            raise ValueError(msg)
        provider = factory()
        if provider.profile != self.plugin(name).profile:
            msg = f"plugin {name!r} has a different provider factory profile"
            raise ValueError(msg)
        return provider

    def stream_transport(  # noqa: PLR0913
        self,
        name: str,
        *,
        executor: CommandExecutor,
        env: Mapping[str, str] | None,
        clock: Clock,
        ids: IdAllocator,
        check_timeout: float,
        log: Callable[[str], None] | None,
    ) -> Transport:
        """Construct a provider's long-lived transport."""
        factory = self.plugin(name).stream_factory
        if factory is None:
            msg = f"provider {name!r} has no stream transport; available: {self.stream_names()}"
            raise ValueError(msg)
        return factory(
            executor=executor, env=env, clock=clock, ids=ids, check_timeout=check_timeout, log=log
        )

    def probe_spec(self, name: str) -> ProbeSpec:
        """Return a provider's optional readiness checks."""
        spec = self.plugin(name).probe
        if spec is None:
            msg = f"no readiness probe for provider {name!r}"
            raise ValueError(msg)
        return spec

    def test_support(self, name: str) -> ProviderTestSupport:
        """Return optional test adapters for a provider."""
        support = self.plugin(name).testing
        if support is None:
            msg = f"no test support for provider {name!r}"
            raise ValueError(msg)
        return support
