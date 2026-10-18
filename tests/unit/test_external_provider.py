"""An out-of-package provider works across the public catalog APIs."""

from __future__ import annotations

from dataclasses import replace

import pytest
from agentshim import (
    DEFAULT_CATALOG,
    Agent,
    ApprovalPolicy,
    CliAgent,
    NativePermissions,
    ProviderCatalog,
    ProviderPlugin,
    TransportKind,
    TurnRequest,
    get_provider,
    probe_provider,
    provider_names,
)
from agentshim.testing import (
    FakeExecutor,
    FakeTransport,
    FakeTurn,
    RecordingEventHandler,
    probe_executor,
    scripted_turn,
)

from tests.external_provider.plugin import PLUGIN, OutsideProvider

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

    def stream_factory(**kwargs: object) -> FakeTransport:
        del kwargs
        return FakeTransport((FakeTurn(text="stream turn"),), profile=profile)

    plugin = ProviderPlugin(name="outside-stream", profile=profile, stream_factory=stream_factory)
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


def test_catalog_is_immutable_and_rejects_collisions() -> None:
    catalog = DEFAULT_CATALOG.with_plugin(PLUGIN)
    assert "outside" not in DEFAULT_CATALOG.names()
    with pytest.raises(ValueError, match="duplicate provider plugin"):
        catalog.with_plugin(PLUGIN)
    with pytest.raises(ValueError, match="duplicate provider plugin"):
        ProviderCatalog((PLUGIN, PLUGIN))
    with pytest.raises(ValueError, match="profile name"):
        ProviderPlugin(name="wrong", profile=PLUGIN.profile, provider_factory=OutsideProvider)
