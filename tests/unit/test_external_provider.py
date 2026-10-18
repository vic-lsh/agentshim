"""An out-of-package provider works across the public catalog APIs."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from typing import cast

import pytest
from agentshim import (
    DEFAULT_CATALOG,
    Agent,
    ApprovalPolicy,
    CliAgent,
    ModelPricing,
    ModelSpec,
    ModelTable,
    NativePermissions,
    PricingTable,
    ProviderCatalog,
    ProviderPlugin,
    TransportKind,
    TurnRequest,
    context_window,
    default_models,
    default_pricing,
    get_provider,
    price_for,
    probe_provider,
    provider_names,
)
from agentshim.core.plugin import PricingContext
from agentshim.testing import (
    FakeExecutor,
    FakeTransport,
    FakeTurn,
    RecordingEventHandler,
    probe_executor,
    scripted_turn,
)

from tests.external_provider.plugin import PLUGIN, PROFILE, OutsidePlugin, OutsideProvider

ENV = {"PATH": "/usr/bin:/bin", "HOME": "/home/tester"}


def test_external_provider_works_by_name_through_cli_agent_and_agent() -> None:
    catalog = DEFAULT_CATALOG.with_plugin(PLUGIN)
    assert "outside" not in provider_names()
    assert "outside" in provider_names(catalog=catalog)
    assert isinstance(get_provider("outside", catalog=catalog), OutsideProvider)

    run = scripted_turn("outside", text="from outside", catalog=catalog)
    handler = RecordingEventHandler()
    executor = FakeExecutor(run)
    cli = CliAgent("outside", catalog=catalog, executor=executor, env=ENV, event_handler=handler)
    result = cli.run("hello", cwd="/work")
    assert result.text == "from outside"
    assert result.session_id == "outside-1"
    assert executor.requests[0].stdin == "hello"
    assert handler.events

    executor = FakeExecutor(run)
    agent = Agent(
        "outside",
        catalog=catalog,
        executor=executor,
        env=ENV,
        permissions=NativePermissions.bypass(),
        approvals=ApprovalPolicy.DENY,
    )
    session = agent.session("/work")
    ticket = session.prepare_turn(TurnRequest(prompt="hello"))
    assert session.run(ticket).result.text == "from outside"


def test_external_provider_probe_and_test_helpers() -> None:
    catalog = DEFAULT_CATALOG.with_plugin(PLUGIN)
    executor = probe_executor("outside", catalog=catalog)
    status = probe_provider("outside", catalog=catalog, executor=executor, env=ENV)
    assert status.binary_found
    assert status.version == "1.2.3"
    agent = Agent(
        "outside",
        catalog=catalog,
        executor=executor,
        env=ENV,
        permissions=NativePermissions.bypass(),
        approvals=ApprovalPolicy.DENY,
    )
    assert agent.probe().version == "1.2.3"


def test_external_stream_only_plugin_works_by_name() -> None:
    profile = replace(PLUGIN.profile, name="outside-stream")

    def stream_factory(options: object) -> FakeTransport:
        del options
        return FakeTransport((FakeTurn(text="stream turn"),), profile=profile)

    plugin = replace(
        PLUGIN,
        name="outside-stream",
        profile=profile,
        provider_factory=None,
        stream_factory=stream_factory,
    )
    catalog = DEFAULT_CATALOG.with_plugin(plugin)
    assert "outside-stream" in catalog.stream_names()
    agent = Agent(
        "outside-stream",
        catalog=catalog,
        transport=TransportKind.STREAM,
        executor=FakeExecutor([]),
        env=ENV,
        permissions=NativePermissions.bypass(),
        approvals=ApprovalPolicy.DENY,
    )
    session = agent.session("/work")
    ticket = session.prepare_turn(TurnRequest(prompt="hello"))
    assert session.run(ticket).result.text == "stream turn"
    with pytest.raises(ValueError, match="no one-shot transport"):
        get_provider("outside-stream", catalog=catalog)


def test_external_provider_supplies_vendor_mapping_for_model_facts() -> None:
    catalog = DEFAULT_CATALOG.with_plugin(PLUGIN)
    assert context_window("outside", "sample", catalog=catalog) == 32000
    assert price_for("outside", "sample", catalog=catalog) == PLUGIN.pricing_entries[0]
    standalone = ProviderCatalog((PLUGIN,))
    assert context_window("outside", "sample", catalog=standalone) == 32000
    assert price_for("outside", "sample", catalog=standalone) == PLUGIN.pricing_entries[0]
    assert context_window("outside", "sample-dynamic", catalog=standalone) == 64000
    premium = price_for(
        "outside", "sample", catalog=standalone, context=PricingContext(tier="premium")
    )
    assert premium is not None
    assert premium.input_usd == 3.0
    assert price_for("codex", "gpt-6-luna", context=PricingContext(tier="standard")) is not None
    sibling = replace(
        PLUGIN, name="outside-sibling", profile=replace(PROFILE, name="outside-sibling")
    )
    shared = ProviderCatalog((PLUGIN, sibling))
    assert len(default_models(catalog=shared).entries) == 1
    assert len(default_pricing(catalog=shared).entries) == 1
    changed = replace(
        sibling, model_entries=(replace(sibling.model_entries[0], context_window=64000),)
    )
    with pytest.raises(ValueError, match="conflicting model facts"):
        default_models(catalog=ProviderCatalog((PLUGIN, changed)))
    changed_price = replace(
        sibling,
        pricing_entries=(replace(sibling.pricing_entries[0], input_usd=3.0),),
    )
    with pytest.raises(ValueError, match="conflicting prices"):
        default_pricing(catalog=ProviderCatalog((PLUGIN, changed_price)))
    checked = date(2026, 10, 18)
    models = ModelTable(
        version="external-models",
        last_updated=checked,
        entries=(ModelSpec("outcorp", "sample", 32000, "https://example.com/model", checked),),
    )
    prices = PricingTable(
        version="external-prices",
        last_updated=checked,
        entries=(
            ModelPricing(
                vendor="outcorp",
                model="sample",
                input_usd=1.0,
                cache_read_usd=0.1,
                cache_write_usd=1.0,
                output_usd=2.0,
                source="https://example.com/pricing",
                checked=checked,
            ),
        ),
    )
    assert context_window("outside", "sample", models) is None
    assert context_window("outside", "sample", models, catalog=catalog) == 32000
    assert price_for("outside", "sample", prices) is None
    assert price_for("outside", "sample", prices, catalog=catalog) == prices.entries[0]


def test_catalog_is_immutable_and_rejects_collisions() -> None:
    catalog = DEFAULT_CATALOG.with_plugin(PLUGIN)
    assert "outside" not in DEFAULT_CATALOG.names()
    with pytest.raises(ValueError, match="duplicate provider plugin"):
        catalog.with_plugin(PLUGIN)
    with pytest.raises(ValueError, match="duplicate provider plugin"):
        ProviderCatalog((PLUGIN, PLUGIN))
    with pytest.raises(ValueError, match="profile name"):
        ProviderCatalog((OutsidePlugin(name="wrong", profile=PLUGIN.profile),))
    with pytest.raises(TypeError, match="missing a required method"):
        ProviderCatalog((cast("ProviderPlugin", object()),))
