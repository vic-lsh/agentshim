"""``Session.report_tools``: whether a session's tool servers connected, before any turn."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentshim import (
    Agent,
    ApprovalPolicy,
    McpServerState,
    McpServerStatus,
    NativePermissions,
    StdioMcpServer,
    TransportKind,
)
from agentshim.testing import (
    ClaudeStreamPeers,
    CodexScript,
    FakeClock,
    FakeExecutor,
    SequentialIds,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from agentshim import Session

READY, FAILED, PENDING = McpServerState.READY, McpServerState.FAILED, McpServerState.PENDING


def _claude_agent(peers: ClaudeStreamPeers) -> Agent:
    return Agent(
        "claude",
        transport=TransportKind.STREAM,
        executor=FakeExecutor([], peers=peers.build),
        env={},
        permissions=NativePermissions.bypass(),
        approvals=ApprovalPolicy.DENY,
        clock=FakeClock(),
        ids=SequentialIds(),
    )


def _claude(mcp_status: Sequence[Mapping[str, tuple[str, str]]] | None, *names: str) -> Session:
    agent = _claude_agent(ClaudeStreamPeers(mcp_status=mcp_status))
    servers = tuple(StdioMcpServer(name, "/bin/tool") for name in names)
    session = agent.session("/work", mcp_servers=servers)
    session.warm()
    return session


def _states(session: Session, timeout_s: float = 5.0) -> dict[str, McpServerState] | None:
    report = session.report_tools(timeout_s)
    return None if report is None else {s.name: s.state for s in report}


class TestClaude:
    def test_a_connected_server_is_ready(self) -> None:
        assert _states(_claude([{"vibesys": ("connected", "")}], "vibesys")) == {"vibesys": READY}

    def test_a_failed_server_is_failed_with_the_cli_s_words(self) -> None:
        session = _claude([{"vibesys": ("failed", "spawn ENOENT")}], "vibesys")
        assert session.report_tools(5.0) == (McpServerStatus("vibesys", FAILED, "spawn ENOENT"),)

    def test_a_server_that_needs_sign_in_is_failed(self) -> None:
        assert _states(_claude([{"vibesys": ("needs-auth", "")}], "vibesys")) == {"vibesys": FAILED}

    def test_a_server_still_connecting_is_asked_about_again(self) -> None:
        pending = {"vibesys": ("pending", "")}
        session = _claude([pending, pending, {"vibesys": ("connected", "")}], "vibesys")
        assert _states(session) == {"vibesys": READY}

    def test_a_server_that_stays_pending_is_pending_not_failed(self) -> None:
        assert _states(_claude([{"vibesys": ("pending", "")}], "vibesys")) == {"vibesys": PENDING}

    def test_a_cli_that_refuses_the_question_leaves_every_server_pending(self) -> None:
        assert _states(_claude(None, "vibesys", "other")) == {"vibesys": PENDING, "other": PENDING}

    def test_a_server_not_listed_or_a_state_nobody_knows_is_pending(self) -> None:
        session = _claude([{"vibesys": ("hibernating", "")}], "vibesys", "other")
        assert _states(session) == {"vibesys": PENDING, "other": PENDING}

    def test_a_session_with_no_tool_servers_has_nothing_to_report(self) -> None:
        assert _claude(None).report_tools(5.0) == ()

    def test_a_session_that_is_not_open_cannot_say(self) -> None:
        agent = _claude_agent(ClaudeStreamPeers())
        assert agent.session("/work").report_tools(1.0) is None


def _codex(script: CodexScript, *names: str) -> Session:
    clock = FakeClock()
    executor = FakeExecutor([], peers=script.peer)
    agent = Agent(
        "codex",
        transport=TransportKind.STREAM,
        executor=executor,
        env={},
        permissions=NativePermissions.bypass(),
        approvals=ApprovalPolicy.DENY,
        clock=clock,
        ids=SequentialIds(),
    )
    servers = tuple(StdioMcpServer(name, "/bin/tool") for name in names)
    session = agent.session("/work", mcp_servers=servers)
    session.warm()
    return session


class TestCodex:
    def test_a_started_server_is_ready(self) -> None:
        assert _states(_codex(CodexScript(), "vibesys")) == {"vibesys": READY}

    def test_a_server_that_failed_to_start_is_failed_with_its_error(self) -> None:
        script = CodexScript()
        script.fail_mcp_server("vibesys", "No such file or directory")
        report = _codex(script, "vibesys").report_tools(5.0)
        assert report == (McpServerStatus("vibesys", FAILED, "No such file or directory"),)

    def test_a_name_with_a_dash_is_reported_by_the_name_it_was_given(self) -> None:
        assert _states(_codex(CodexScript(), "my-tools")) == {"my-tools": READY}
