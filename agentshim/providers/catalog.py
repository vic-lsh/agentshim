"""An immutable catalog of independently defined provider integrations."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from agentshim.core.conversation import TransportKind
from agentshim.core.plugin import ConnectionOptions, ProviderPlugin

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from agentshim.core.clock import Clock
    from agentshim.core.conversation import Transport
    from agentshim.core.ids import IdAllocator
    from agentshim.core.plugin import ProviderTestSupport
    from agentshim.core.provider import Provider
    from agentshim.core.status import ProbeSpec
    from agentshim.execution.executor import CommandExecutor


class ProviderCatalog:
    """An immutable, explicitly extended set of provider plugins."""

    _REQUIRED_METHODS = (
        "describe",
        "connection_modes",
        "connect",
        "probe",
        "model_info",
        "pricing_info",
        "known_models",
        "known_prices",
        "test_support",
    )

    def __init__(self, plugins: Sequence[ProviderPlugin] = ()) -> None:
        """Build a catalog; duplicate names fail rather than override another plugin."""
        entries: dict[str, ProviderPlugin] = {}
        for plugin in plugins:
            if not isinstance(cast("object", plugin), ProviderPlugin) or any(
                not callable(getattr(plugin, method, None)) for method in self._REQUIRED_METHODS
            ):
                msg = "provider plugin is missing a required method"
                raise TypeError(msg)
            if (
                not isinstance(cast("object", plugin.name), str)
                or not plugin.name
                or plugin.name != plugin.name.strip()
            ):
                msg = "provider plugin name must be non-empty and unpadded"
                raise ValueError(msg)
            if not plugin.connection_modes():
                msg = f"plugin {plugin.name!r} needs at least one connection mode"
                raise ValueError(msg)
            if plugin.name in entries:
                msg = f"duplicate provider plugin {plugin.name!r}"
                raise ValueError(msg)
            if plugin.describe().name != plugin.name:
                msg = f"plugin {plugin.name!r} has provider profile name {plugin.describe().name!r}"
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
        return sorted(
            name
            for name, plugin in self._plugins.items()
            if TransportKind.STREAM in plugin.connection_modes()
        )

    def plugin(self, name: str) -> ProviderPlugin:
        """Find one plugin or report the names available in this catalog."""
        plugin = self._plugins.get(name)
        if plugin is None:
            msg = f"unknown provider {name!r}; available: {self.names()}"
            raise ValueError(msg)
        return plugin

    def provider(self, name: str) -> Provider:
        """Construct a one-shot provider by name."""
        plugin = self.plugin(name)
        if TransportKind.ONE_SHOT not in plugin.connection_modes():
            msg = f"provider {name!r} has no one-shot transport"
            raise ValueError(msg)
        provider = cast("Provider", plugin.connect(ConnectionOptions(TransportKind.ONE_SHOT)))
        if provider.profile != plugin.describe():
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
        plugin = self.plugin(name)
        if TransportKind.STREAM not in plugin.connection_modes():
            msg = f"provider {name!r} has no stream transport; available: {self.stream_names()}"
            raise ValueError(msg)
        return cast(
            "Transport",
            plugin.connect(
                ConnectionOptions(
                    TransportKind.STREAM,
                    executor=executor,
                    env=env,
                    clock=clock,
                    ids=ids,
                    check_timeout=check_timeout,
                    log=log,
                )
            ),
        )

    def probe_spec(self, name: str) -> ProbeSpec:
        """Return the legacy static probe declaration, if this plugin has one."""
        from .static_plugin import StaticProviderPlugin  # noqa: PLC0415

        plugin = self.plugin(name)
        spec = plugin.readiness if isinstance(plugin, StaticProviderPlugin) else None
        if spec is None:
            msg = f"no readiness probe for provider {name!r}"
            raise ValueError(msg)
        return spec

    def test_support(self, name: str) -> ProviderTestSupport:
        """Return optional test adapters for a provider."""
        support = self.plugin(name).test_support()
        if support is None:
            msg = f"no test support for provider {name!r}"
            raise ValueError(msg)
        return support
