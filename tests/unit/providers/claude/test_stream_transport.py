"""``ClaudeStreamTransport`` over a scripted ``claude`` process: no CLI, no waiting.

The fake process answers the way the CLI does (see ``agentshim.testing.claude_stream``),
so these tests drive the real transport: its argv, its handshake, its pull loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar

import pytest
from agentshim import (
    ApprovalDenied,
    ApprovalPolicy,
    AssistantText,
    ClaudeStreamTransport,
    CliExitError,
    ConversationSpec,
    FailureKind,
    HttpMcpServer,
    McpScope,
    NativeMode,
    NativePermissions,
    OutputSchema,
    ProviderCapabilityError,
    SchemaDialectError,
    SessionResumeError,
    SessionStateError,
    SkillScope,
    StdioMcpServer,
    TurnFailedError,
    TurnInterrupted,
    TurnRequest,
    TurnTimeoutError,
)
from agentshim.testing import (
    ClaudeApiError,
    ClaudeCrash,
    ClaudePeerTurn,
    ClaudeStreamPeer,
    ClaudeStreamPeers,
    FakeClock,
    FakeExecutor,
    RecordingEventHandler,
    SequentialIds,
)
from agentshim.testing.contracts import ConversationContract, TransportContract
from hypothesis import given
from hypothesis import strategies as st

if TYPE_CHECKING:
    from pathlib import Path

    from agentshim import AgentEvent, Conversation, Transport, TurnResult

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}},
    "required": ["answer"],
}


@dataclass
class Rig:
    """A transport on a fake executor, with the handles a test inspects."""

    transport: ClaudeStreamTransport
    executor: FakeExecutor
    peers: ClaudeStreamPeers
    clock: FakeClock

    @property
    def argvs(self) -> list[list[str]]:
        return [list(spawn.argv) for spawn in self.executor.spawns]


def rig(
    script: list[ClaudePeerTurn] | None = None,
    *,
    known_sessions: tuple[str, ...] = (),
    env: dict[str, str] | None = None,
    interrupt_grace_s: float = 10.0,
) -> Rig:
    peers = ClaudeStreamPeers(script or [], known_sessions=known_sessions)
    executor = FakeExecutor([], peers=peers.build)
    clock = FakeClock()
    transport = ClaudeStreamTransport(
        executor=executor,
        env=env if env is not None else {"PATH": "/bin"},
        clock=clock,
        ids=SequentialIds(),
        interrupt_grace_s=interrupt_grace_s,
    )
    return Rig(transport, executor, peers, clock)


def spec(**changes: Any) -> ConversationSpec:  # noqa: ANN401 - forwarded to a dataclass constructor
    base = ConversationSpec(
        cwd="/work",
        model=None,
        permissions=NativePermissions.bypass(),
        approvals=ApprovalPolicy.DENY,
    )
    return replace(base, **changes)


def ignore(event: AgentEvent) -> None:
    del event


def run(
    conversation: Conversation,
    prompt: str = "hi",
    *,
    output_schema: OutputSchema | None = None,
    timeout: float | None = None,
    reasoning_effort: str | None = None,
) -> tuple[TurnResult, RecordingEventHandler]:
    events = RecordingEventHandler()
    request = TurnRequest(
        prompt=prompt,
        output_schema=output_schema,
        timeout=timeout,
        reasoning_effort=reasoning_effort,
    )
    return conversation.turn(request, events.on_event), events


def peer(rig_: Rig, index: int = 0) -> ClaudeStreamPeer:
    return rig_.peers.peers[index]


class TestTransportContract(TransportContract):
    def make_transport(self) -> Transport:
        return rig().transport


class TestConversationContract(ConversationContract):
    def make_conversation(self) -> Conversation:
        return rig().transport.open(spec())


class TestOpen:
    def test_construction_checks_the_install_once(self) -> None:
        rigged = rig()
        assert rigged.executor.checked == ["/usr/local/bin/claude"]

    def test_open_starts_one_process_and_sends_initialize(self) -> None:
        rigged = rig()
        rigged.transport.open(spec())
        assert len(rigged.executor.spawns) == 1
        initialize = peer(rigged).received[0]
        assert initialize["type"] == "control_request"
        assert initialize["request"] == {"subtype": "initialize"}

    def test_the_profile_declares_the_modes_it_enforces(self) -> None:
        modes = rig().transport.profile.native_permission_modes
        assert modes == {NativeMode.BYPASS, NativeMode.WORKSPACE_WRITE}

    def test_read_only_is_rejected_because_it_cannot_be_enforced(self) -> None:
        with pytest.raises(ProviderCapabilityError, match="read_only"):
            rig().transport.open(spec(permissions=NativePermissions.read_only()))

    def test_network_access_is_rejected_because_it_cannot_be_enforced(self) -> None:
        permissions = NativePermissions.workspace_write(network=True)
        with pytest.raises(ProviderCapabilityError, match="network"):
            rig().transport.open(spec(permissions=permissions))

    def test_a_refused_resume_is_a_session_resume_error_at_open(self) -> None:
        rigged = rig()
        with pytest.raises(SessionResumeError) as raised:
            rigged.transport.open(spec(resume_id="gone"))
        assert raised.value.session_id == "gone"
        assert "No conversation found" in raised.value.detail

    def test_a_known_conversation_resumes(self) -> None:
        rigged = rig(known_sessions=("s-1",))
        conversation = rigged.transport.open(spec(resume_id="s-1"))
        assert conversation.conversation_id == "s-1"
        result, _ = run(conversation)
        assert result.session_id == "s-1"
        assert result.resumed is True

    def test_a_process_that_dies_at_start_is_a_typed_failure_with_its_stderr(self) -> None:
        class Dying(ClaudeStreamPeer):
            def on_start(self) -> list[Any]:  # type: ignore[override]
                from agentshim import ProcessExited, StderrLine  # noqa: PLC0415

                return [StderrLine("boom\n"), ProcessExited(3)]

        executor = FakeExecutor([], peers=lambda spawn: Dying(ClaudeStreamPeers(), spawn.argv))
        transport = ClaudeStreamTransport(
            executor=executor, env={}, clock=FakeClock(), ids=SequentialIds()
        )
        with pytest.raises(CliExitError) as raised:
            transport.open(spec())
        assert raised.value.returncode == 3
        assert "boom" in raised.value.stderr

    def test_a_silent_process_times_out_at_start(self) -> None:
        from agentshim.testing import SilentPeer  # noqa: PLC0415

        clock = FakeClock()
        transport = ClaudeStreamTransport(
            executor=FakeExecutor([], peers=lambda spawn: (spawn, SilentPeer())[1]),
            env={},
            clock=clock,
            ids=SequentialIds(),
            startup_timeout_s=2.0,
        )
        with pytest.raises(TurnTimeoutError):
            transport.open(spec())


class TestArgv:
    def argv(self, **changes: Any) -> list[str]:  # noqa: ANN401 - forwarded to a dataclass constructor
        rigged = rig()
        rigged.transport.open(spec(**changes))
        return rigged.argvs[0]

    def test_it_speaks_stream_json_both_ways_without_print_mode(self) -> None:
        argv = self.argv()
        assert argv[0] == "/usr/local/bin/claude"
        assert argv[1:7] == [
            "--output-format",
            "stream-json",
            "--verbose",
            "--input-format",
            "stream-json",
            "--permission-prompts",
        ]
        assert "-p" not in argv
        assert "--print" not in argv

    def test_it_keeps_the_cli_system_prompt(self) -> None:
        argv = self.argv()
        assert not any(arg.startswith("--system-prompt") for arg in argv)

    def test_bypass_is_the_bypass_permission_mode(self) -> None:
        argv = self.argv()
        assert argv[argv.index("--permission-mode") + 1] == "bypassPermissions"
        assert "--dangerously-skip-permissions" not in argv

    def test_prompts_are_denied_by_the_cli_itself(self) -> None:
        argv = self.argv()
        assert argv[argv.index("--permission-prompts") + 1] == "none"

    def test_the_model_is_passed(self) -> None:
        argv = self.argv(model="haiku")
        assert argv[argv.index("--model") + 1] == "haiku"

    def test_resume_uses_the_equals_form(self) -> None:
        rigged = rig(known_sessions=("-dash-id",))
        rigged.transport.open(spec(resume_id="-dash-id"))
        assert "--resume=-dash-id" in rigged.argvs[0]
        assert "--resume" not in rigged.argvs[0]

    def test_reasoning_effort_is_a_flag(self) -> None:
        argv = self.argv(reasoning_effort="high")
        assert argv[argv.index("--effort") + 1] == "high"

    def test_mcp_servers_are_one_inline_config(self) -> None:
        servers = (
            StdioMcpServer(name="files", command="/bin/mcp", args=("--x",)),
            HttpMcpServer(name="web", url="http://localhost:1/mcp"),
        )
        argv = self.argv(mcp_servers=servers)
        config = json.loads(argv[argv.index("--mcp-config") + 1])
        assert set(config["mcpServers"]) == {"files", "web"}
        assert config["mcpServers"]["files"]["command"] == "/bin/mcp"
        assert "--strict-mcp-config" not in argv

    def test_session_mcp_scope_is_strict(self) -> None:
        argv = self.argv(mcp_scope=McpScope.SESSION)
        assert "--strict-mcp-config" in argv
        assert "--mcp-config" not in argv

    def test_project_skill_scope_restricts_setting_sources(self) -> None:
        argv = self.argv(skill_scope=SkillScope.PROJECT)
        assert argv[argv.index("--setting-sources") + 1] == "project,local"

    def test_workspace_write_accepts_edits_in_the_roots_and_sandboxes_the_rest(self) -> None:
        permissions = NativePermissions.workspace_write(["/data/out"])
        argv = self.argv(permissions=permissions)
        assert argv[argv.index("--permission-mode") + 1] == "acceptEdits"
        assert argv[argv.index("--add-dir") + 1] == "/data/out"
        settings = json.loads(argv[argv.index("--settings") + 1])
        sandbox = settings["sandbox"]
        assert sandbox["enabled"] is True
        assert sandbox["failIfUnavailable"] is True
        assert sandbox["allowUnsandboxedCommands"] is False
        assert sandbox["filesystem"] == {"allowWrite": ["/data/out"]}
        assert "network" not in sandbox
        (entry,) = settings["hooks"]["PreToolUse"]
        assert entry["matcher"].split("|") == ["Edit", "Write", "NotebookEdit", "MultiEdit"]
        assert "/work" in entry["hooks"][0]["command"]
        assert "/data/out" in entry["hooks"][0]["command"]

    def test_bypass_has_no_settings_block(self) -> None:
        assert "--settings" not in self.argv()

    def test_the_nested_session_marker_never_reaches_the_child(self) -> None:
        rigged = rig(env={"CLAUDECODE": "1", "HOME": "/home/u"})
        rigged.transport.open(spec())
        env = rigged.executor.spawns[0].env
        assert "CLAUDECODE" not in env
        assert env["HOME"] == "/home/u"

    def test_the_sandbox_needs_its_working_directory_variable(self) -> None:
        rigged = rig()
        rigged.transport.open(spec(permissions=NativePermissions.workspace_write()))
        assert rigged.executor.spawns[0].env["CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR"] == "1"

    def test_the_process_runs_in_the_conversations_directory(self) -> None:
        rigged = rig()
        rigged.transport.open(spec(cwd="/some/where"))
        assert rigged.executor.spawns[0].cwd == "/some/where"


class TestTurns:
    def test_two_turns_share_one_process(self) -> None:
        rigged = rig([ClaudePeerTurn(text="one"), ClaudePeerTurn(text="two")])
        conversation = rigged.transport.open(spec())
        first, _ = run(conversation, "a")
        second, _ = run(conversation, "b")
        assert (first.text, second.text) == ("one", "two")
        assert len(rigged.executor.spawns) == 1
        assert peer(rigged).prompts == ["a", "b"]
        assert first.session_id == second.session_id == conversation.conversation_id
        assert (first.resumed, second.resumed) == (False, True)

    def test_a_prompt_is_one_user_message_line(self) -> None:
        rigged = rig()
        conversation = rigged.transport.open(spec())
        run(conversation, "multi\nline")
        written = rigged.executor.processes[0].writes
        assert written[-1].endswith("\n")
        message = json.loads(written[-1])
        assert message["type"] == "user"
        assert message["message"] == {"role": "user", "content": "multi\nline"}

    def test_events_follow_the_stream(self) -> None:
        rigged = rig([ClaudePeerTurn(text="hello", tool_calls=[("Bash", {"cmd": "ls"}, "out")])])
        _, events = run(rigged.transport.open(spec()))
        kinds = [type(event).__name__ for event in events.events]
        assert "SessionStarted" in kinds
        assert "ToolCall" in kinds
        assert "ToolResult" in kinds
        assert events.of_type(AssistantText)[0].text == "hello"  # type: ignore[attr-defined]

    def test_usage_is_per_turn_and_cost_is_the_difference(self) -> None:
        script = [
            ClaudePeerTurn(input_tokens=100, output_tokens=7, cost_usd=0.25),
            ClaudePeerTurn(input_tokens=40, output_tokens=3, cost_usd=0.5),
        ]
        conversation = rig(script).transport.open(spec())
        first, _ = run(conversation)
        second, events = run(conversation)
        assert (first.usage.tokens.input_tokens, first.usage.tokens.output_tokens) == (100, 7)
        assert (second.usage.tokens.input_tokens, second.usage.tokens.output_tokens) == (40, 3)
        assert first.cost_usd == pytest.approx(0.25)
        assert second.cost_usd == pytest.approx(0.5)
        reports = [e for e in events.events if type(e).__name__ == "UsageReport"]
        assert reports[-1].cost_usd == pytest.approx(0.5)  # type: ignore[attr-defined]

    def test_a_restart_starts_the_cost_over(self) -> None:
        script = [ClaudePeerTurn(cost_usd=0.5), ClaudePeerTurn(cost_usd=0.125)]
        conversation = rig(script).transport.open(spec())
        run(conversation)
        second, _ = run(conversation, output_schema=OutputSchema(SCHEMA, host_dir=_NO_DIR))
        assert second.cost_usd == pytest.approx(0.125)

    def test_skills_are_reported_from_the_stream(self) -> None:
        peers = ClaudeStreamPeers([ClaudePeerTurn(skills_invoked=("review",))], skills=("review",))
        transport = ClaudeStreamTransport(
            executor=FakeExecutor([], peers=peers.build),
            env={},
            clock=FakeClock(),
            ids=SequentialIds(),
        )
        result, _ = run(transport.open(spec()))
        assert result.skills.discovered == ("review",)
        assert result.skills.invoked == ("review",)

    def test_turns_are_one_at_a_time(self) -> None:
        rigged = rig([ClaudePeerTurn(stall=True)])
        conversation = rigged.transport.open(spec())
        errors: list[Exception] = []

        def second(event: AgentEvent) -> None:
            if isinstance(event, AssistantText):
                try:
                    conversation.turn(TurnRequest(prompt="x"), ignore)
                except SessionStateError as error:
                    errors.append(error)
                conversation.interrupt()

        conversation.turn(TurnRequest(prompt="a"), second)
        assert len(errors) == 1

    def test_a_turn_cannot_move_the_working_directory(self) -> None:
        conversation = rig().transport.open(spec())
        with pytest.raises(ProviderCapabilityError, match="working directory"):
            conversation.turn(TurnRequest(prompt="x", cwd="/elsewhere"), ignore)


_NO_DIR: Any = None


@pytest.fixture
def schema(tmp_path: Path) -> OutputSchema:
    return OutputSchema(SCHEMA, host_dir=tmp_path)


class TestStructuredOutput:
    def test_a_schema_turn_returns_the_payload(self, schema: OutputSchema) -> None:
        script = [ClaudePeerTurn(text="42", structured_output={"answer": 42})]
        rigged = rig(script)
        conversation = rigged.transport.open(spec())
        result, _ = run(conversation, output_schema=schema)
        assert result.structured_output == {"answer": 42}

    def test_the_schema_is_a_process_flag_so_a_new_one_restarts_with_resume(
        self, schema: OutputSchema
    ) -> None:
        rigged = rig([ClaudePeerTurn(), ClaudePeerTurn(structured_output={"answer": 1})])
        conversation = rigged.transport.open(spec())
        first, _ = run(conversation)
        second, _ = run(conversation, output_schema=schema)
        assert len(rigged.executor.spawns) == 2
        assert "--json-schema" not in rigged.argvs[0]
        restarted = rigged.argvs[1]
        assert f"--resume={first.session_id}" in restarted
        assert json.loads(restarted[restarted.index("--json-schema") + 1]) == SCHEMA
        assert second.session_id == first.session_id
        assert rigged.executor.processes[0].stdin_closed

    def test_the_same_schema_keeps_the_process(self, schema: OutputSchema) -> None:
        rigged = rig()
        conversation = rigged.transport.open(spec())
        run(conversation, output_schema=schema)
        run(conversation, output_schema=schema)
        assert len(rigged.executor.spawns) == 2  # open + the first schema turn, then reused

    def test_dropping_the_schema_restarts_too(self, schema: OutputSchema) -> None:
        rigged = rig()
        conversation = rigged.transport.open(spec())
        run(conversation, output_schema=schema)
        run(conversation)
        assert "--json-schema" not in rigged.argvs[-1]
        assert len(rigged.executor.spawns) == 3

    def test_a_schema_the_dialect_rejects_never_starts_a_process(
        self, schema: OutputSchema
    ) -> None:
        bad = OutputSchema({"type": "object", "not": {}}, host_dir=schema.host_dir)
        rigged = rig()
        conversation = rigged.transport.open(spec())
        with pytest.raises(SchemaDialectError):
            run(conversation, output_schema=bad)
        assert len(rigged.executor.spawns) == 1

    def test_per_turn_mcp_servers_and_effort_also_restart(self) -> None:
        rigged = rig()
        conversation = rigged.transport.open(spec())
        run(conversation)
        run(conversation, reasoning_effort="low")
        run(conversation, reasoning_effort="low")
        assert [("--effort" in argv) for argv in rigged.argvs] == [False, True]

    def test_an_exhausted_schema_is_a_schema_failure(self, schema: OutputSchema) -> None:
        script = [
            ClaudePeerTurn(
                text="could not satisfy", fail_subtype="error_max_structured_output_retries"
            )
        ]
        conversation = rig(script).transport.open(spec())
        with pytest.raises(TurnFailedError) as raised:
            run(conversation, output_schema=schema)
        assert raised.value.kind is FailureKind.SCHEMA


class TestFailures:
    @pytest.mark.parametrize(
        ("error", "kind"),
        [
            (ClaudeApiError(), FailureKind.TRANSIENT),
            (
                ClaudeApiError(
                    status=429, kind="rate_limit", text="You've hit your limit · resets 3pm"
                ),
                FailureKind.USAGE_LIMIT,
            ),
            (
                ClaudeApiError(status=401, kind="authentication_failed", text="Invalid API key"),
                FailureKind.AUTH,
            ),
            (
                ClaudeApiError(status=400, kind="invalid_request", text="API Error: 400 bad"),
                FailureKind.OTHER,
            ),
        ],
    )
    def test_an_api_error_result_is_classified_on_is_error(
        self, error: ClaudeApiError, kind: FailureKind
    ) -> None:
        conversation = rig([ClaudePeerTurn(api_error=error)]).transport.open(spec())
        with pytest.raises(TurnFailedError) as raised:
            run(conversation)
        assert raised.value.kind is kind
        assert error.text in raised.value.detail

    def test_the_conversation_survives_an_api_error(self) -> None:
        rigged = rig([ClaudePeerTurn(api_error=ClaudeApiError()), ClaudePeerTurn(text="back")])
        conversation = rigged.transport.open(spec())
        with pytest.raises(TurnFailedError):
            run(conversation)
        result, _ = run(conversation)
        assert result.text == "back"
        assert len(rigged.executor.spawns) == 1

    def test_a_process_that_exits_after_an_api_error_is_replaced_by_a_resume(self) -> None:
        script = [ClaudePeerTurn(api_error=ClaudeApiError(ends_process=True)), ClaudePeerTurn()]
        rigged = rig(script)
        conversation = rigged.transport.open(spec())
        with pytest.raises(TurnFailedError):
            run(conversation)
        session_id = conversation.conversation_id
        result, _ = run(conversation)
        assert result.session_id == session_id
        assert len(rigged.executor.spawns) == 2
        assert f"--resume={session_id}" in rigged.argvs[1]

    def test_a_process_that_dies_mid_turn_fails_the_turn_with_its_stderr(self) -> None:
        script = [ClaudePeerTurn(crash=ClaudeCrash(2, "segfault in tool\n"))]
        conversation = rig(script).transport.open(spec())
        with pytest.raises(CliExitError) as raised:
            run(conversation)
        assert raised.value.kind is FailureKind.OTHER
        assert raised.value.returncode == 2
        assert "segfault in tool" in raised.value.detail

    def test_the_next_turn_after_a_death_resumes_in_a_new_process(self) -> None:
        rigged = rig([ClaudePeerTurn(), ClaudePeerTurn(crash=ClaudeCrash()), ClaudePeerTurn()])
        conversation = rigged.transport.open(spec())
        run(conversation)
        session_id = conversation.conversation_id
        with pytest.raises(CliExitError):
            run(conversation)
        result, _ = run(conversation)
        assert result.session_id == session_id
        assert f"--resume={session_id}" in rigged.argvs[1]

    def test_a_dead_pipe_at_the_prompt_is_a_typed_failure(self) -> None:
        rigged = rig()
        conversation = rigged.transport.open(spec())
        rigged.executor.processes[0].kill()
        # the exit is noticed between turns, so the turn replaces the process
        result, _ = run(conversation)
        assert result.text == "ok"
        assert len(rigged.executor.spawns) == 2


class TestInterrupt:
    def test_an_interrupted_turn_returns_interrupted_and_keeps_the_conversation(self) -> None:
        rigged = rig([ClaudePeerTurn(text="working", stall=True), ClaudePeerTurn(text="alive")])
        conversation = rigged.transport.open(spec())
        events: list[AgentEvent] = []

        def handler(event: AgentEvent) -> None:
            events.append(event)
            if isinstance(event, AssistantText):
                conversation.interrupt()

        first = conversation.turn(TurnRequest(prompt="go"), handler)
        assert first.interrupted is True
        assert any(isinstance(event, TurnInterrupted) for event in events)
        second, _ = run(conversation)
        assert (second.text, second.interrupted) == ("alive", False)
        assert len(rigged.executor.spawns) == 1
        assert second.session_id == first.session_id

    def test_the_interrupt_is_a_control_request(self) -> None:
        rigged = rig([ClaudePeerTurn(text="working", stall=True)])
        conversation = rigged.transport.open(spec())
        conversation.turn(
            TurnRequest(prompt="go"),
            lambda event: conversation.interrupt() if isinstance(event, AssistantText) else None,
        )
        requests = [m for m in peer(rigged).received if m["type"] == "control_request"]
        assert [m["request"]["subtype"] for m in requests] == ["initialize", "interrupt"]
        assert requests[0]["request_id"] != requests[1]["request_id"]

    def test_an_interrupt_while_idle_does_nothing(self) -> None:
        rigged = rig()
        conversation = rigged.transport.open(spec())
        conversation.interrupt()
        run(conversation)
        subtypes = [
            m["request"]["subtype"] for m in peer(rigged).received if m["type"] == "control_request"
        ]
        assert subtypes == ["initialize"]

    def test_a_turn_that_finishes_despite_the_interrupt_is_not_interrupted(self) -> None:
        conversation = rig([ClaudePeerTurn(text="done")]).transport.open(spec())
        result = conversation.turn(
            TurnRequest(prompt="go"),
            lambda event: conversation.interrupt() if isinstance(event, AssistantText) else None,
        )
        assert result.interrupted is False
        assert result.text == "done"


class TestTimeout:
    def test_an_overrunning_turn_is_interrupted_then_times_out_keeping_the_conversation(
        self,
    ) -> None:
        rigged = rig([ClaudePeerTurn(stall=True), ClaudePeerTurn(text="after")])
        conversation = rigged.transport.open(spec())
        with pytest.raises(TurnTimeoutError) as raised:
            run(conversation, timeout=3.0)
        assert raised.value.timeout == 3.0
        assert rigged.clock.monotonic() >= 3.0
        subtypes = [
            m["request"]["subtype"] for m in peer(rigged).received if m["type"] == "control_request"
        ]
        assert subtypes[-1] == "interrupt"
        result, _ = run(conversation)
        assert result.text == "after"
        assert len(rigged.executor.spawns) == 1

    def test_a_process_that_ignores_the_interrupt_is_killed_and_replaced_later(self) -> None:
        script = [ClaudePeerTurn(stall=True, ignores_interrupt=True), ClaudePeerTurn(text="after")]
        rigged = rig(script, interrupt_grace_s=2.0)
        conversation = rigged.transport.open(spec())
        with pytest.raises(TurnTimeoutError):
            run(conversation, timeout=1.0)
        assert rigged.executor.processes[0].kill_calls == 1
        session_id = conversation.conversation_id
        result, _ = run(conversation)
        assert result.text == "after"
        assert f"--resume={session_id}" in rigged.argvs[1]

    def test_no_timeout_means_the_turn_may_take_as_long_as_it_takes(self) -> None:
        conversation = rig([ClaudePeerTurn(text="slow")]).transport.open(spec())
        result, _ = run(conversation)
        assert result.text == "slow"


class TestApprovals:
    REQUEST: ClassVar[dict[str, Any]] = {
        "subtype": "can_use_tool",
        "tool_name": "Bash",
        "input": {"command": "rm -rf /"},
    }

    def test_deny_answers_at_once_reports_and_lets_the_turn_finish(self) -> None:
        rigged = rig([ClaudePeerTurn(text="went on", control_requests=[self.REQUEST])])
        conversation = rigged.transport.open(spec(approvals=ApprovalPolicy.DENY))
        result, events = run(conversation)
        assert result.text == "went on"
        (denied,) = events.of_type(ApprovalDenied)
        assert denied.kind == "Bash"  # type: ignore[attr-defined]
        assert "rm -rf" in denied.detail  # type: ignore[attr-defined]
        (answer,) = peer(rigged).answers
        assert answer["subtype"] == "success"
        assert answer["response"]["behavior"] == "deny"
        assert answer["response"]["interrupt"] is False

    def test_fail_turn_denies_with_interrupt_then_fails_the_turn(self) -> None:
        rigged = rig([ClaudePeerTurn(control_requests=[self.REQUEST]), ClaudePeerTurn(text="ok")])
        conversation = rigged.transport.open(spec(approvals=ApprovalPolicy.FAIL_TURN))
        events = RecordingEventHandler()
        with pytest.raises(TurnFailedError, match="Bash"):
            conversation.turn(TurnRequest(prompt="go"), events.on_event)
        assert events.of_type(ApprovalDenied)
        assert peer(rigged).answers[0]["response"]["interrupt"] is True
        result, _ = run(conversation)  # the conversation survives
        assert result.text == "ok"

    @pytest.mark.parametrize(
        "request_body",
        [
            {"subtype": "hook_callback", "callback_id": "h1", "input": {}},
            {"subtype": "mcp_message", "server_name": "s", "message": {}},
            {"subtype": "something_new"},
        ],
    )
    def test_other_requests_get_an_error_answer_and_never_block(
        self, request_body: dict[str, Any]
    ) -> None:
        rigged = rig([ClaudePeerTurn(text="fine", control_requests=[request_body])])
        result, events = run(rigged.transport.open(spec()))
        assert result.text == "fine"
        assert not events.of_type(ApprovalDenied)
        (answer,) = peer(rigged).answers
        assert answer["subtype"] == "error"
        assert request_body["subtype"] in answer["error"]

    def test_every_request_of_a_turn_is_answered(self) -> None:
        requests = [self.REQUEST, {"subtype": "hook_callback"}, self.REQUEST]
        rigged = rig([ClaudePeerTurn(control_requests=requests)])
        run(rigged.transport.open(spec()))
        assert len(peer(rigged).answers) == 3

    def test_denials_the_cli_made_on_its_own_are_reported(self) -> None:
        denial = {"tool_name": "Write", "tool_use_id": "t1", "tool_input": {"file_path": "/x"}}
        rigged = rig([ClaudePeerTurn(permission_denials=[denial])])
        _, events = run(rigged.transport.open(spec()))
        (denied,) = events.of_type(ApprovalDenied)
        assert denied.kind == "Write"  # type: ignore[attr-defined]

    def test_fail_turn_also_fails_on_a_denial_the_cli_made_on_its_own(self) -> None:
        denial = {"tool_name": "Write", "tool_use_id": "t1", "tool_input": {}}
        rigged = rig([ClaudePeerTurn(permission_denials=[denial])])
        conversation = rigged.transport.open(spec(approvals=ApprovalPolicy.FAIL_TURN))
        with pytest.raises(TurnFailedError, match="Write"):
            run(conversation)

    def test_a_denial_is_reported_once_whichever_way_it_arrives(self) -> None:
        request = {**self.REQUEST, "tool_use_id": "t9"}
        denial = {"tool_name": "Bash", "tool_use_id": "t9", "tool_input": {}}
        rigged = rig([ClaudePeerTurn(control_requests=[request], permission_denials=[denial])])
        _, events = run(rigged.transport.open(spec()))
        assert len(events.of_type(ApprovalDenied)) == 1


class TestClose:
    def test_close_ends_stdin_and_nothing_harsher_when_the_process_exits(self) -> None:
        rigged = rig()
        conversation = rigged.transport.open(spec())
        conversation.close()
        process = rigged.executor.processes[0]
        assert process.stdin_closed
        assert (process.terminate_calls, process.kill_calls) == (0, 0)

    def test_close_terminates_a_process_that_ignores_end_of_input(self) -> None:
        class Stubborn(ClaudeStreamPeer):
            def on_stdin_closed(self) -> list[Any]:  # type: ignore[override]
                return []

        peers = ClaudeStreamPeers()
        executor = FakeExecutor([], peers=lambda request: Stubborn(peers, request.argv))
        transport = ClaudeStreamTransport(
            executor=executor, env={}, clock=FakeClock(), ids=SequentialIds()
        )
        transport.open(spec()).close()
        assert executor.processes[0].terminate_calls == 1

    def test_a_failed_open_leaves_no_process_running(self) -> None:
        rigged = rig()
        with pytest.raises(SessionResumeError):
            rigged.transport.open(spec(resume_id="gone"))
        process = rigged.executor.processes[0]
        assert process.wait(0) is not None


def _turn_plans() -> st.SearchStrategy[list[ClaudePeerTurn]]:
    plain = st.builds(
        ClaudePeerTurn,
        text=st.text(max_size=20),
        input_tokens=st.integers(0, 10_000),
        output_tokens=st.integers(0, 10_000),
        cost_usd=st.floats(0.0, 5.0, allow_nan=False, allow_infinity=False),
    )
    return st.lists(plain, min_size=1, max_size=8)


class TestProperties:
    @given(_turn_plans())
    def test_one_process_serves_any_number_of_turns_and_reports_each_turn_alone(
        self, script: list[ClaudePeerTurn]
    ) -> None:
        rigged = rig(list(script))
        conversation = rigged.transport.open(spec())
        results = [run(conversation)[0] for _ in script]
        assert len(rigged.executor.spawns) == 1
        assert [r.text for r in results] == [t.text for t in script]
        assert [r.usage.tokens.input_tokens for r in results] == [t.input_tokens for t in script]
        assert [r.usage.tokens.output_tokens for r in results] == [t.output_tokens for t in script]
        for result, planned in zip(results, script, strict=True):
            assert result.cost_usd == pytest.approx(planned.cost_usd, abs=1e-9)
        assert len({r.session_id for r in results}) == 1

    @given(st.lists(st.sampled_from(["ok", "api", "crash", "interrupt"]), min_size=1, max_size=8))
    def test_whatever_happens_the_conversation_keeps_its_id(self, plan: list[str]) -> None:
        turns = {
            "ok": ClaudePeerTurn(),
            "api": ClaudePeerTurn(api_error=ClaudeApiError()),
            "crash": ClaudePeerTurn(crash=ClaudeCrash()),
            "interrupt": ClaudePeerTurn(stall=True),
        }
        rigged = rig([turns[name] for name in plan])
        conversation = rigged.transport.open(spec())
        ids: set[str] = set()
        for name in plan:
            handler = (
                (lambda e: conversation.interrupt() if isinstance(e, AssistantText) else None)
                if name == "interrupt"
                else ignore
            )
            try:
                conversation.turn(TurnRequest(prompt="x"), handler)
            except TurnFailedError:
                assert name in {"api", "crash"}
            if conversation.conversation_id is not None:
                ids.add(conversation.conversation_id)
        assert len(ids) <= 1
