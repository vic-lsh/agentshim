"""An immutable catalog of independently defined provider integrations."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from agentshim.core.clock import Clock
    from agentshim.core.conversation import Transport
    from agentshim.core.ids import IdAllocator
    from agentshim.core.plugin import ProviderPlugin, ProviderTestSupport
    from agentshim.core.provider import Provider
    from agentshim.core.status import ProbeSpec
    from agentshim.execution.executor import CommandExecutor


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

    def model_vendor(self, name: str) -> str | None:
        """Resolve a provider name or model prefix to its declared vendor."""
        plugin = self._plugins.get(name)
        if plugin is not None:
            return plugin.model_vendor
        for entry in self._plugins.values():
            for alias, vendor in entry.model_aliases:
                if alias == name:
                    return vendor
        return None

    def pricing_vendor(self, name: str) -> str | None:
        """Resolve a provider name to its declared billing vendor."""
        plugin = self._plugins.get(name)
        return None if plugin is None else plugin.pricing_vendor

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
