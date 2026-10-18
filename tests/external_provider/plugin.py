"""A minimal standalone provider with no imports from agentshim.providers.*."""

from __future__ import annotations

import json
from datetime import date
from typing import TYPE_CHECKING

from agentshim import ModelPricing, ModelSpec, ProviderPlugin, ProviderTestSupport
from agentshim.core.events import AssistantText
from agentshim.core.mcp import NoopInstallation
from agentshim.core.profile import McpMechanism, OutputSchemaStyle, ProviderProfile
from agentshim.core.provider import ParsedTurn
from agentshim.core.status import AuthState, ProbeSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from agentshim.core.errors import AgentShimError, CliExitError
    from agentshim.core.events import AgentEvent
    from agentshim.core.mcp import McpServer
    from agentshim.core.provider import ArgvContext, McpInstallation
    from agentshim.core.usage import TokenUsage


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


PLUGIN = ProviderPlugin(
    name="outside",
    profile=PROFILE,
    provider_factory=OutsideProvider,
    probe=ProbeSpec(auth=None),
    model_vendor="outcorp",
    pricing_vendor="outcorp",
    model_entries=(
        ModelSpec("outcorp", "sample", 32000, "https://example.com/model", date(2026, 10, 18)),
    ),
    pricing_entries=(
        ModelPricing(
            "outcorp",
            "sample",
            1.0,
            0.1,
            1.0,
            2.0,
            "https://example.com/pricing",
            date(2026, 10, 18),
        ),
    ),
    testing=ProviderTestSupport(
        scripted_lines=scripted_lines,
        resume_failure_lines=resume_failure_lines,
        probe_fixture=probe_fixture,
    ),
)
