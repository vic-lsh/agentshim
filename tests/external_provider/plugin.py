"""A minimal standalone provider with no imports from agentshim.providers.*."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

from agentshim import ModelPricing, ModelSpec, ProviderTestSupport
from agentshim.core.conversation import TransportKind
from agentshim.core.errors import ProviderCapabilityError
from agentshim.core.events import AssistantText
from agentshim.core.mcp import NoopInstallation
from agentshim.core.plugin import ConnectionOptions, PricingContext
from agentshim.core.profile import McpMechanism, OutputSchemaStyle, ProviderProfile
from agentshim.core.provider import ParsedTurn
from agentshim.core.status import AuthState, ProbeSpec
from agentshim.execution.readiness import probe_spec_on

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from agentshim.core.conversation import Transport
    from agentshim.core.errors import AgentShimError, CliExitError
    from agentshim.core.events import AgentEvent
    from agentshim.core.mcp import McpServer
    from agentshim.core.models import ModelTable
    from agentshim.core.pricing import PricingTable
    from agentshim.core.provider import ArgvContext, McpInstallation, Provider
    from agentshim.core.status import ProviderStatus
    from agentshim.core.usage import TokenUsage
    from agentshim.execution.executor import CommandExecutor


PROFILE = ProviderProfile(
    name="outside",
    display_name="Outside Agent",
    binary="outside-agent",
    supports_resume=True,
    supports_reasoning_effort=False,
    mcp=McpMechanism.NONE,
    output_schema=OutputSchemaStyle.NONE,
    schema_dialect=None,
    state_dirs=(),
    darwin_state_dirs=(),
    auth_env_vars=(),
    skill_dirs=(),
    container_install=(),
)


class OutsideParser:
    """Parse this fake CLI's one JSON line."""

    def __init__(self, emit: Callable[[AgentEvent], None]) -> None:
        """Remember the event sink for this turn."""
        self.emit = emit
        self.text = ""
        self.session_id: str | None = None

    def feed_stdout(self, line: str) -> None:
        """Decode one result frame."""
        frame = json.loads(line)
        self.text = frame["text"]
        self.session_id = frame["session_id"]
        self.emit(AssistantText(self.text))

    def feed_stderr(self, line: str) -> None:
        """Ignore irrelevant stderr in this sample CLI."""
        del line

    def finish(self) -> ParsedTurn:
        """Return the parsed result."""
        return ParsedTurn(text=self.text, session_id=self.session_id)


class OutsideProvider:
    """A provider implemented entirely outside the agentshim package."""

    profile = PROFILE

    def build_argv(self, ctx: ArgvContext) -> list[str]:
        """Build one command, including a resume id when present."""
        argv = [ctx.binary_path, "run", "--json"]
        if ctx.resume_session_id:
            argv += ["--resume", ctx.resume_session_id]
        return argv

    def new_parser(
        self, emit: Callable[[AgentEvent], None], *, expect_structured: bool
    ) -> OutsideParser:
        """Create a fresh parser."""
        if expect_structured:
            msg = "outside-agent does not support output schemas"
            raise ValueError(msg)
        return OutsideParser(emit)

    def install_mcp(self, workspace: Path | None, servers: Sequence[McpServer]) -> McpInstallation:
        """Reject MCP, which this sample CLI does not support."""
        del workspace
        if servers:
            msg = "outside-agent does not support MCP"
            raise ValueError(msg)
        return NoopInstallation()

    def classify_exit(self, error: CliExitError, *, resumed: bool) -> AgentShimError:
        """Preserve the CLI exit error."""
        del resumed
        return error


def scripted_lines(  # noqa: PLR0913
    *,
    text: str = "",
    session_id: str | None = None,
    usage: TokenUsage | None = None,
    tool_calls: Sequence[tuple[str, Mapping[str, object], str]] = (),
    structured_output: object | None = None,
    skills_offered: Sequence[str] | None = None,
    skills_invoked: Sequence[str] = (),
) -> list[str]:
    """Create a fixture in the sample CLI's own format."""
    del usage, tool_calls, skills_offered, skills_invoked
    if structured_output is not None:
        msg = "outside-agent does not support output schemas"
        raise ValueError(msg)
    return [json.dumps({"text": text, "session_id": session_id or "outside-1"}) + "\n"]


def resume_failure_lines(
    *, session_id: str | None = None
) -> tuple[Sequence[str], Sequence[str], int]:
    """Create a failed resume fixture."""
    del session_id
    return (), ("cannot resume\n",), 1


def probe_fixture(args: Sequence[str], *, version: str, auth: AuthState) -> tuple[str, str, int]:
    """Create a fake version response."""
    del auth
    if list(args) == ["--version"]:
        return f"outside-agent {version}\n", "", 0
    return "", "unknown command\n", 2


_MODEL = ModelSpec("outcorp", "sample", 32000, "https://example.com/model", date(2026, 10, 18))
_PRICE = ModelPricing(
    "outcorp", "sample", 1.0, 0.1, 1.0, 2.0, "https://example.com/pricing", date(2026, 10, 18)
)


@dataclass(frozen=True)
class OutsidePlugin:
    """A third-party implementation of the required method contract."""

    name: str = "outside"
    profile: ProviderProfile = PROFILE
    provider_factory: Callable[[], Provider] | None = OutsideProvider
    stream_factory: Callable[[ConnectionOptions], Transport] | None = None
    model_entries: tuple[ModelSpec, ...] = (_MODEL,)
    pricing_entries: tuple[ModelPricing, ...] = (_PRICE,)

    def describe(self) -> ProviderProfile:
        """Describe this CLI, independent of installation."""
        return self.profile

    def connection_modes(self) -> frozenset[TransportKind]:
        """Advertise only supported connection modes."""
        modes: set[TransportKind] = set()
        if self.provider_factory is not None:
            modes.add(TransportKind.ONE_SHOT)
        if self.stream_factory is not None:
            modes.add(TransportKind.STREAM)
        return frozenset(modes)

    def connect(self, options: ConnectionOptions) -> Provider | Transport:
        """Create the requested connection."""
        if options.kind is TransportKind.ONE_SHOT and self.provider_factory is not None:
            return self.provider_factory()
        if options.kind is TransportKind.STREAM and self.stream_factory is not None:
            return self.stream_factory(options)
        msg = f"{self.name} does not support {options.kind.value}"
        raise ProviderCapabilityError(msg)

    def probe(
        self, executor: CommandExecutor, env: Mapping[str, str], timeout: float
    ) -> ProviderStatus:
        """Check the external CLI on the caller's executor."""
        return probe_spec_on(self.profile, ProbeSpec(auth=None), executor, env, timeout)

    def model_info(self, model: str, table: ModelTable | None = None) -> ModelSpec | None:
        """Resolve a context window, including a computed model for this example."""
        if model == "sample-dynamic" and table is None:
            return ModelSpec(
                "outcorp", model, 64000, "https://example.com/model", date(2026, 10, 18)
            )
        source = self.model_entries if table is None else table.entries
        return next((entry for entry in source if entry.key == ("outcorp", model)), None)

    def pricing_info(
        self, model: str, context: PricingContext, table: PricingTable | None = None
    ) -> ModelPricing | None:
        """Compute a premium price on demand; otherwise use enumerated facts."""
        if context.tier == "premium" and model == "sample" and table is None:
            return ModelPricing(
                "outcorp",
                model,
                3.0,
                0.3,
                3.0,
                6.0,
                "https://example.com/pricing",
                date(2026, 10, 18),
            )
        if context.tier not in (None, "standard"):
            return None
        source = self.pricing_entries if table is None else table.entries
        return next((entry for entry in source if entry.key == ("outcorp", model)), None)

    def known_models(self) -> tuple[ModelSpec, ...]:
        """Enumerate facts available to a static snapshot."""
        return self.model_entries

    def known_prices(self) -> tuple[ModelPricing, ...]:
        """Enumerate standard-tier facts available to a static snapshot."""
        return self.pricing_entries

    def test_support(self) -> ProviderTestSupport | None:
        """Supply external fixtures to shared test helpers."""
        return ProviderTestSupport(
            scripted_lines=scripted_lines,
            resume_failure_lines=resume_failure_lines,
            probe_fixture=probe_fixture,
        )


PLUGIN = OutsidePlugin()
