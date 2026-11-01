"""Small adapter for providers whose capabilities and facts are static."""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agentshim.core.conversation import TransportKind
from agentshim.core.errors import ProviderCapabilityError
from agentshim.core.models import ModelTable
from agentshim.core.pricing import PricingTable
from agentshim.core.rates import RateTable, candidate_names
from agentshim.execution.readiness import probe_spec_on

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from agentshim.core.conversation import Transport
    from agentshim.core.limits import LimitsReader
    from agentshim.core.models import ModelSpec
    from agentshim.core.plugin import (
        ConnectionOptions,
        PricingContext,
        ProviderTestSupport,
        StreamTransportFactory,
    )
    from agentshim.core.pricing import ModelPricing
    from agentshim.core.profile import ProviderProfile
    from agentshim.core.provider import Provider
    from agentshim.core.status import ProbeSpec, ProviderStatus
    from agentshim.core.transcripts import TranscriptFormat
    from agentshim.execution.executor import CommandExecutor

_DATE_SUFFIX = re.compile(r"-\d{8}$")
_BRACKET_SUFFIX = re.compile(r"\[[^\]]*\]$")


@dataclass(frozen=True)
class StaticProviderPlugin:
    """Implement the method contract from fixed provider-owned declarations."""

    name: str
    profile: ProviderProfile
    provider_factory: Callable[[], Provider] | None = None
    stream_factory: StreamTransportFactory | None = None
    readiness: ProbeSpec | None = None
    testing: ProviderTestSupport | None = None
    model_vendor: str | None = None
    pricing_vendor: str | None = None
    model_aliases: tuple[tuple[str, str], ...] = ()
    model_entries: tuple[ModelSpec, ...] = ()
    pricing_entries: tuple[ModelPricing, ...] = ()
    #: Flat rates with speed tiers, for estimates; see :mod:`agentshim.core.rates`.
    rates: RateTable = field(default_factory=lambda: RateTable(dt.date.min, {}))
    #: Names the CLI reports that are not a priceable model: placeholders and bare family names.
    unpriceable_models: frozenset[str] = frozenset()
    #: How to read this CLI's own session files for usage, when agentshim can.
    transcript_format: TranscriptFormat | None = None
    #: How to read this CLI's account limits on request, when it can say.
    limits: LimitsReader | None = None

    def __post_init__(self) -> None:
        """Reject inconsistent identity and an adapter with no connection mode."""
        if not self.name or self.name != self.name.strip():
            msg = "provider plugin name must be non-empty and unpadded"
            raise ValueError(msg)
        if self.profile.name != self.name:
            msg = f"plugin {self.name!r} has provider profile name {self.profile.name!r}"
            raise ValueError(msg)
        if self.provider_factory is None and self.stream_factory is None:
            msg = f"plugin {self.name!r} needs a one-shot or stream connection"
            raise ValueError(msg)

    def describe(self) -> ProviderProfile:
        """Return this CLI's declared capabilities."""
        return self.profile

    def connection_modes(self) -> frozenset[TransportKind]:
        """Declare the modes backed by this adapter's implementations."""
        modes: set[TransportKind] = set()
        if self.provider_factory is not None:
            modes.add(TransportKind.ONE_SHOT)
        if self.stream_factory is not None:
            modes.add(TransportKind.STREAM)
        return frozenset(modes)

    def connect(self, options: ConnectionOptions) -> Provider | Transport:
        """Create the requested connection; fail explicitly for an unsupported mode."""
        if options.kind is TransportKind.ONE_SHOT:
            if self.provider_factory is None:
                msg = f"provider {self.name!r} has no one-shot transport"
                raise ProviderCapabilityError(msg)
            return self.provider_factory()
        if self.stream_factory is None:
            msg = f"provider {self.name!r} has no stream transport"
            raise ProviderCapabilityError(msg)
        if options.executor is None or options.clock is None or options.ids is None:
            msg = "stream connection needs executor, clock, and ids"
            raise ValueError(msg)
        return self.stream_factory(
            executor=options.executor,
            env=options.env,
            clock=options.clock,
            ids=options.ids,
            check_timeout=options.check_timeout,
            log=options.log,
        )

    def probe(
        self, executor: CommandExecutor, env: Mapping[str, str], timeout: float
    ) -> ProviderStatus:
        """Run the declared checks on the executor supplied by the caller."""
        return probe_spec_on(self.profile, self.readiness, executor, env, timeout)

    def model_info(self, model: str, table: ModelTable | None = None) -> ModelSpec | None:
        """Look up a model by this provider's vendor and prefix rules."""
        if not model or "[" in model:
            return None
        source = table or ModelTable(
            "plugin",
            max((x.checked for x in self.model_entries), default=dt.date.min),
            self.model_entries,
        )
        key = self._key(model, self.model_vendor, source)
        return None if key is None else source.get(*key)

    def pricing_info(
        self, model: str, context: PricingContext, table: PricingTable | None = None
    ) -> ModelPricing | None:
        """Look up a standard-tier price; other tiers remain unknown."""
        if (
            not model
            or context.as_of is not None
            or context.region not in (None, "global")
            or context.tier not in (None, "standard")
        ):
            return None
        source = table or PricingTable(
            "plugin",
            max((x.checked for x in self.pricing_entries), default=dt.date.min),
            self.pricing_entries,
        )
        key = self._key(_BRACKET_SUFFIX.sub("", model.strip()), self.pricing_vendor, source)
        return None if key is None else source.get(*key)

    def _key(
        self, model: str, default_vendor: str | None, table: ModelTable | PricingTable
    ) -> tuple[str, str] | None:
        name = model.strip().lower()
        vendor = default_vendor
        known_vendors = {entry.vendor for entry in table.entries}
        if "/" in name:
            prefix, _, name = name.partition("/")
            vendor = dict(self.model_aliases).get(prefix, prefix)
            if vendor not in known_vendors:
                return None
        if vendor is None or not name:
            return None
        return (vendor, _DATE_SUFFIX.sub("", name))

    def known_models(self) -> tuple[ModelSpec, ...]:
        """Return the enumerated context-window facts."""
        return self.model_entries

    def known_prices(self) -> tuple[ModelPricing, ...]:
        """Return the enumerated standard-tier rates."""
        return self.pricing_entries

    def known_rates(self) -> RateTable:
        """Return the enumerated flat rates, with speed tiers."""
        return self.rates

    def rate_names(self, model: str) -> tuple[str, ...]:
        """Flat names *model* may be filed under; none for a name that is not a model."""
        names = candidate_names(model)
        if any(name.rpartition("/")[2] in self.unpriceable_models for name in names):
            return ()
        return names

    def transcripts(self) -> TranscriptFormat | None:
        """Return the format of this CLI's session files, if agentshim reads them."""
        return self.transcript_format

    def limits_reader(self) -> LimitsReader | None:
        """Return this CLI's account limits reader, if it has one."""
        return self.limits

    def test_support(self) -> ProviderTestSupport | None:
        """Return this provider's test adapters, if any."""
        return self.testing
