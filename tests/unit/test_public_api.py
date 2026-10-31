"""The public surface is what ``__all__`` says it is."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import agentshim
import agentshim.testing


def test_every_exported_name_is_importable() -> None:
    for name in agentshim.__all__:
        assert hasattr(agentshim, name), name


def test_exported_names_are_sorted_and_unique() -> None:
    assert agentshim.__all__ == sorted(agentshim.__all__)
    assert len(agentshim.__all__) == len(set(agentshim.__all__))


def test_testing_module_exports() -> None:
    for name in agentshim.testing.__all__:
        assert hasattr(agentshim.testing, name), name


def test_version_is_reported() -> None:
    assert agentshim.__version__ == "0.24.0"


def test_core_and_execution_do_not_import_providers_at_module_level() -> None:
    """The layering rule, checked instead of documented.

    ``import-linter`` enforces the whole ordering as a build gate; this
    catches the specific edge that used to exist in ``core.agent`` without
    needing the tool installed.
    """
    root = Path(agentshim.__file__ or "").parent
    offenders: list[str] = []
    for path in sorted((*(root / "core").rglob("*.py"), *(root / "execution").rglob("*.py"))):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            for statement in (
                ast.walk(node) if isinstance(node, (ast.Import, ast.ImportFrom)) else ()
            ):
                if isinstance(statement, ast.ImportFrom) and "providers" in (
                    statement.module or ""
                ):
                    offenders.append(f"{path.name}: {ast.unparse(statement)}")
                if isinstance(statement, ast.Import):
                    offenders.extend(
                        f"{path.name}: import {alias.name}"
                        for alias in statement.names
                        if "providers" in alias.name
                    )
    assert offenders == []


def test_stream_provider_names_are_exactly_the_providers_that_build_a_stream_agent() -> None:
    def builds(name: str) -> bool:
        try:
            agentshim.Agent(
                name,
                permissions=agentshim.NativePermissions.bypass(),
                approvals=agentshim.ApprovalPolicy.DENY,
                executor=agentshim.testing.FakeExecutor([]),
                transport=agentshim.TransportKind.STREAM,
            )
        except ValueError:
            return False
        return True

    assert agentshim.stream_provider_names() == [
        name for name in agentshim.provider_names() if builds(name)
    ]
    assert agentshim.stream_provider_names() == ["claude", "codex"]


def test_provider_names_lists_the_ported_providers() -> None:
    assert agentshim.provider_names() == ["claude", "codex", "copilot", "gemini", "opencode"]


def test_the_provider_classes_the_docs_name_are_exported() -> None:
    """The docs tell callers to construct these directly, so they must be here."""
    for name in (
        "ClaudeProvider",
        "CodexProvider",
        "CopilotProvider",
        "GeminiProvider",
        "OpencodeProvider",
        "SandboxConfig",
        "ClaudeToolPolicy",
    ):
        assert name in agentshim.__all__, name
        assert hasattr(agentshim, name), name


def test_a_provider_package_does_not_export_fold_usage() -> None:
    """``fold_usage`` has a different signature in each package.

    Five names that look alike and take different arguments are worse than no
    shared name at all, so it stays private to its own ``parser.py``.
    """
    for name in agentshim.provider_names():
        package = importlib.import_module(f"agentshim.providers.{name}")
        assert "fold_usage" not in package.__all__, name


def test_the_process_confinement_clock_and_permission_names_are_exported() -> None:
    for name in (
        "ApprovalPolicy",
        "Clock",
        "Confinement",
        "DockerExecConfinement",
        "IdAllocator",
        "NativeMode",
        "NativePermissions",
        "Process",
        "ProcessClosedError",
        "ReapError",
        "ProcessExited",
        "ProcessOutput",
        "RandomIds",
        "SpawnRequest",
        "StderrLine",
        "StdoutLine",
        "StopSignal",
        "SystemClock",
        "confine",
    ):
        assert name in agentshim.__all__, name


def test_the_new_test_doubles_are_exported() -> None:
    for name in (
        "EchoPeer",
        "FakeClock",
        "FakeConfinement",
        "FakePeer",
        "FakeProcess",
        "GateMarker",
        "ReplayGates",
        "SequentialIds",
        "SilentPeer",
    ):
        assert name in agentshim.testing.__all__, name


def test_the_session_tier_names_are_exported() -> None:
    for name in (
        "Agent",
        "ApprovalDenied",
        "RateLimitStatus",
        "Checkpoint",
        "CheckpointStore",
        "Continuity",
        "ContinuityError",
        "Conversation",
        "ConversationSpec",
        "InMemoryCheckpointStore",
        "OneShotTransport",
        "PolicyConfig",
        "RenewalBudget",
        "RetryPolicy",
        "Session",
        "SessionPolicy",
        "SessionState",
        "SessionStateError",
        "Transport",
        "Turn",
        "TurnCancelledError",
        "TurnFailedError",
        "TurnInterrupted",
        "TurnTicket",
        "TurnTimeoutError",
    ):
        assert name in agentshim.__all__, name


def test_the_session_test_doubles_are_exported() -> None:
    for name in (
        "FakeCheckpointStore",
        "FakeConversation",
        "FakeOutcome",
        "FakeTransport",
        "FakeTurn",
        "fake_profile",
        "resume_refused",
        "turn_failed",
        "turn_timeout",
    ):
        assert name in agentshim.testing.__all__, name


def test_the_claude_stream_transport_surface_is_exported() -> None:
    for name in ("ClaudeStreamTransport", "TransportKind"):
        assert name in agentshim.__all__, name
    for name in (
        "ClaudeApiError",
        "ClaudeCrash",
        "ClaudePeerTurn",
        "ClaudeRecordedPeer",
        "ClaudeStreamPeer",
        "ClaudeStreamPeers",
    ):
        assert name in agentshim.testing.__all__, name


def test_the_codex_app_server_transport_and_its_test_double_are_public() -> None:
    for name in ("CodexAppServerTransport", "TransportKind"):
        assert name in agentshim.__all__, name
    for name in (
        "CodexScript",
        "CodexAppServerPeer",
        "Say",
        "Think",
        "RunCommand",
        "ChangeFile",
        "CallMcp",
        "Ask",
        "AskKind",
        "Spend",
        "Retrying",
        "Fail",
        "Hang",
        "Crash",
        "Complain",
    ):
        assert name in agentshim.testing.__all__, name


def test_the_probe_surface_is_exported() -> None:
    for name in ("AuthState", "ProviderStatus", "probe_provider"):
        assert name in agentshim.__all__, name
    assert "probe_executor" in agentshim.testing.__all__


def test_the_rate_limit_event_is_exported() -> None:
    assert "RateLimitStatus" in agentshim.__all__
    assert "ReportRateLimits" in agentshim.testing.__all__


def test_steering_names_are_public() -> None:
    for name in (
        "NoRunningTurnError",
        "SteerableConversation",
        "SteerConsumed",
        "SteerDelivered",
        "SteerRejected",
    ):
        assert name in agentshim.__all__, name
    assert issubclass(agentshim.NoRunningTurnError, agentshim.SessionStateError)
    assert agentshim.ProviderProfile.__dataclass_fields__["supports_steer"].default is False


def test_one_shot_providers_do_not_declare_steering() -> None:
    for name in agentshim.provider_names():
        assert agentshim.get_provider(name).profile.supports_steer is False, name


def test_the_rates_by_model_and_transcript_names_are_exported() -> None:
    for name in (
        "CostBreakdown",
        "FilePosition",
        "FileScan",
        "LITELLM_ENDPOINTS",
        "ModelUsage",
        "RATES_TTL",
        "REFRESH_FLOOR",
        "RateCatalog",
        "RateDocumentError",
        "RateEndpoint",
        "RateEntry",
        "RateFetchError",
        "RateNaming",
        "RateRead",
        "RateSnapshot",
        "RateSource",
        "RateStatus",
        "RateStore",
        "RateTable",
        "Rates",
        "TranscriptFormat",
        "TranscriptParser",
        "TranscriptRecord",
        "TranscriptRoot",
        "TranscriptScan",
        "TranscriptSource",
        "UsageSpeed",
        "bundled_rates",
        "cache_savings_usd",
        "merge_model_usage",
        "parse_litellm",
        "parse_timestamp",
        "price_tokens",
        "rates_for",
        "read_rates",
        "read_transcript_usage",
        "refresh_rates",
        "speed_premium_usd",
        "transcript_providers",
        "transcript_roots",
    ):
        assert name in agentshim.__all__, name


def test_the_rate_test_doubles_are_exported() -> None:
    for name in ("FakeRateSource", "FakeRateStore"):
        assert name in agentshim.testing.__all__, name


def test_recorded_cost_functions_take_a_model_pricing_and_estimates_take_rates() -> None:
    """The type seam: no signature accepts both kinds of rate.

    ``cost_usd`` and ``price_for`` speak ``ModelPricing``/``PricingTable``;
    ``price_tokens`` speaks ``Rates``. Pyright enforces it for callers; this holds the
    declared annotations so a refactor cannot widen one without a failing test.
    """
    import inspect  # noqa: PLC0415

    hints = {
        "cost_usd": inspect.signature(agentshim.cost_usd).parameters["pricing"].annotation,
        "price_for": inspect.signature(agentshim.price_for).parameters["table"].annotation,
        "price_tokens": inspect.signature(agentshim.price_tokens).parameters["rates"].annotation,
    }
    assert hints["cost_usd"] in ("ModelPricing", agentshim.ModelPricing)
    assert "PricingTable" in str(hints["price_for"])
    assert "RateTable" not in str(hints["price_for"])
    assert hints["price_tokens"] in ("Rates", agentshim.Rates)
