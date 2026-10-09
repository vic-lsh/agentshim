"""``CodexAppServerTransport`` against the fake Codex server.

What is checked here is what the transport sends and what it makes of the
replies; the recorded transcripts (``test_transport_replay.py``) check the
same against the real CLI's own output.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from agentshim import (
    ApprovalDenied,
    ApprovalPolicy,
    AssistantText,
    FailureKind,
    HttpMcpServer,
    Lifecycle,
    NativePermissions,
    OutputSchema,
    ProviderCapabilityError,
    ProviderError,
    ProviderUsage,
    Reasoning,
    SchemaDialectError,
    SessionResumeError,
    SessionStarted,
    SessionStateError,
    SkillInvoked,
    Stderr,
    StdioMcpServer,
    TokenUsage,
    ToolCall,
    ToolResult,
    TurnFailedError,
    TurnInterrupted,
    TurnRequest,
    TurnTimeoutError,
    UsageReport,
)
from agentshim.testing import (
    Ask,
    AskKind,
    CallMcp,
    ChangeFile,
    CodexScript,
    Complain,
    Crash,
    Fail,
    Hang,
    Retrying,
    RunCommand,
    Say,
    Spend,
    Think,
)
from hypothesis import given
from hypothesis import strategies as st

from tests.unit.providers.codex.app_server.harness import SAFETY_TIMEOUT_S, rig, spec

if TYPE_CHECKING:
    from pathlib import Path

    from agentshim import AgentEvent


def _params(script: CodexScript, method: str, index: int = -1) -> dict[str, object]:
    return script.requests(method)[index].params.to_wire()  # type: ignore[return-value]


# -- starting a conversation


def test_open_spawns_app_server_through_the_executor_and_runs_the_handshake() -> None:
    r = rig()
    conversation = r.open()
    request = r.executor.spawns[0]
    assert list(request.argv)[:2] == ["/usr/local/bin/codex", "app-server"]
    assert request.cwd == "/work"
    assert request.env["PATH"] == "/usr/bin:/bin"
    methods = [m.to_wire().get("method") for m in r.script.received]
    assert methods == ["initialize", "initialized", "thread/start"]
    assert r.script.violations == []
    assert conversation.conversation_id == "thread-1"
    conversation.close()


def test_the_launchers_path_is_kept_for_the_commands_the_agent_runs() -> None:
    r = rig()
    r.open().close()
    argv = list(r.executor.spawns[0].argv)
    assert 'shell_environment_policy.set.PATH="/usr/bin:/bin"' in argv


def test_thread_start_names_everything_the_users_config_could_otherwise_decide() -> None:
    r = rig(CodexScript(inherited_sandbox="workspace-write", inherited_approval="untrusted"))
    r.open(model="gpt-6-luna").close()
    assert _params(r.script, "thread/start") == {
        "cwd": "/work",
        "model": "gpt-6-luna",
        "sandbox": "danger-full-access",
        "approvalPolicy": "never",
        "ephemeral": False,
    }


def test_resume_names_the_same_settings_and_skips_replaying_history() -> None:
    script = CodexScript()
    script.add_thread("old-thread")
    r = rig(script)
    conversation = r.open(resume_id="old-thread")
    assert _params(script, "thread/resume") == {
        "threadId": "old-thread",
        "cwd": "/work",
        "model": "fake-model",
        "sandbox": "danger-full-access",
        "approvalPolicy": "never",
        "excludeTurns": True,
    }
    assert conversation.conversation_id == "old-thread"
    conversation.close()


def test_a_thread_codex_does_not_have_is_a_refused_resume_at_open() -> None:
    r = rig()
    with pytest.raises(SessionResumeError) as caught:
        r.open(resume_id="gone")
    assert caught.value.session_id == "gone"
    assert "no rollout found" in str(caught.value)
    assert r.executor.processes[0].stdin_closed  # the process that refused was shut down


def test_a_permission_the_server_did_not_apply_is_a_capability_error() -> None:
    r = rig(CodexScript(ignore_requested_permissions=True, inherited_approval="untrusted"))
    with pytest.raises(ProviderCapabilityError, match="approval policy"):
        r.open()
    assert r.executor.processes[0].stdin_closed


def test_a_sandbox_the_server_did_not_apply_is_a_capability_error() -> None:
    r = rig(CodexScript(ignore_requested_permissions=True, inherited_sandbox="read-only"))
    with pytest.raises(ProviderCapabilityError, match="sandbox"):
        r.open()


def test_a_handshake_that_never_finishes_times_out_on_the_clock() -> None:
    from agentshim.testing import SilentPeer  # noqa: PLC0415 - only this test needs it

    r = rig(peers=lambda _request: SilentPeer(), startup_timeout=5.0)
    with pytest.raises(TurnTimeoutError):
        r.open()


def test_a_server_that_dies_during_the_handshake_is_a_failed_turn_with_its_stderr() -> None:
    class Dies:
        def on_start(self) -> list[object]:
            from agentshim import ProcessExited, StderrLine  # noqa: PLC0415

            return [StderrLine("codex: cannot read config\n"), ProcessExited(2)]

        def on_stdin(self, data: str) -> list[object]:
            del data
            return []

        def on_stdin_closed(self) -> list[object]:
            return []

    r = rig(peers=lambda _request: Dies())
    with pytest.raises(TurnFailedError, match="cannot read config"):
        r.open()


# -- permissions on the wire


def test_every_turn_restates_the_sandbox_and_approval_policy() -> None:
    r = rig()
    conversation = r.open(model="gpt-6-luna")
    r.turn(conversation, "one")
    r.turn(conversation, "two")
    for request in r.script.requests("turn/start"):
        wire = request.params.to_wire()
        assert wire["approvalPolicy"] == "never"
        assert wire["sandboxPolicy"] == {"type": "dangerFullAccess"}
        assert wire["model"] == "gpt-6-luna"
        assert wire["cwd"] == "/work"
    conversation.close()


@pytest.mark.parametrize(
    ("permissions", "sandbox", "approval", "policy"),
    [
        (NativePermissions.bypass(), "danger-full-access", "never", {"type": "dangerFullAccess"}),
        (
            NativePermissions.read_only(),
            "read-only",
            "on-request",
            {"type": "readOnly", "networkAccess": False},
        ),
        (
            NativePermissions.workspace_write(["/data", "/cache"], network=True),
            "workspace-write",
            "on-request",
            {
                "type": "workspaceWrite",
                "writableRoots": ["/data", "/cache"],
                "networkAccess": True,
                "excludeSlashTmp": False,
                "excludeTmpdirEnvVar": False,
            },
        ),
    ],
)
def test_each_native_mode_maps_to_codexs_sandbox_approval_and_policy(
    permissions: NativePermissions, sandbox: str, approval: str, policy: dict[str, object]
) -> None:
    r = rig()
    conversation = r.open(permissions=permissions)
    r.turn(conversation)
    start = _params(r.script, "thread/start")
    assert (start["sandbox"], start["approvalPolicy"]) == (sandbox, approval)
    turn = _params(r.script, "turn/start")
    assert (turn["approvalPolicy"], turn["sandboxPolicy"]) == (approval, policy)
    conversation.close()


@given(
    roots=st.lists(st.from_regex(r"/[a-z]{1,8}(/[a-z]{1,8}){0,2}", fullmatch=True), max_size=4),
    network=st.booleans(),
)
def test_workspace_write_roots_and_network_reach_every_turn_unchanged(
    roots: list[str], *, network: bool
) -> None:
    r = rig()
    conversation = r.open(permissions=NativePermissions.workspace_write(roots, network=network))
    r.turn(conversation, "one")
    r.turn(conversation, "two")
    for request in r.script.requests("turn/start"):
        policy = request.params.to_wire()["sandboxPolicy"]
        assert policy["writableRoots"] == roots  # type: ignore[index]
        assert policy["networkAccess"] is network  # type: ignore[index]
    conversation.close()


# -- turns


def test_a_turns_events_follow_the_items_codex_reports() -> None:
    script = CodexScript()
    script.turn(
        Think("planning"),
        Say("looking", phase="commentary"),
        RunCommand("ls -la", output="a.txt\n"),
        ChangeFile("a.txt"),
        CallMcp("docs", "search", {"q": "x"}, output="found"),
        Say("all done"),
    )
    r = rig(script)
    conversation = r.open()
    result = r.turn(conversation)
    kinds = [type(event).__name__ for event in r.events]
    assert kinds[0] == "SessionStarted"
    assert r.events[0] == SessionStarted("thread-1")
    assert Reasoning("planning") in r.events
    assert AssistantText("looking") in r.events
    assert AssistantText("all done") in r.events
    calls = r.of_type(ToolCall)
    assert [call.tool for call in calls] == ["execute", "file_change", "mcp_tool_call"]
    assert calls[0].args == {"command": "ls -la"}
    results = r.of_type(ToolResult)
    assert [res.tool for res in results] == ["execute", "file_change", "mcp_tool_call"]
    assert results[0].stdout == "a.txt\n"
    assert results[0].exit_code == 0
    assert results[2].stdout == '[{"type": "text", "text": "found"}]'
    assert len(r.of_type(UsageReport)) == 1
    assert result.text == "all done"
    conversation.close()


def test_a_failed_tool_reports_on_stderr_with_a_nonzero_exit_code() -> None:
    script = CodexScript()
    script.turn(RunCommand("false", output="boom", exit_code=1), CallMcp("d", "t", error="denied"))
    r = rig(script)
    conversation = r.open()
    r.turn(conversation)
    command, mcp = r.of_type(ToolResult)
    assert (command.stdout, command.stderr, command.exit_code) == ("", "boom", 1)
    assert (mcp.stdout, mcp.stderr, mcp.exit_code) == ("", "denied", 1)
    conversation.close()


def test_a_shell_read_of_a_skill_file_is_reported_as_a_skill_load() -> None:
    script = CodexScript()
    script.turn(RunCommand("sed -n '1,200p' .agents/skills/deploy/SKILL.md", output="..."))
    r = rig(script)
    conversation = r.open()
    result = r.turn(conversation)
    loads = r.of_type(SkillInvoked)
    assert [load.name for load in loads] == ["deploy"]
    assert result.skills.invoked == ("deploy",)
    assert result.skills.invocation_count == 1
    conversation.close()


def test_a_turn_with_no_skill_reads_reports_zero_loads_not_unknown() -> None:
    r = rig()
    conversation = r.open()
    assert r.turn(conversation).skills.invocation_count == 0
    conversation.close()


def test_stderr_lines_and_warnings_are_events() -> None:
    script = CodexScript()
    script.turn(Complain("PATH aliases under /tmp"), Say("ok"))
    r = rig(script)
    conversation = r.open()
    r.turn(conversation)
    assert Stderr("PATH aliases under /tmp") in r.events
    conversation.close()


def test_the_answer_is_the_final_message_not_the_commentary() -> None:
    script = CodexScript()
    script.turn(Say("narration", phase="commentary"), Say("the answer"), Say("late", "commentary"))
    r = rig(script)
    conversation = r.open()
    assert r.turn(conversation).text == "the answer"
    conversation.close()


def test_conversation_flags_report_resumption() -> None:
    script = CodexScript()
    script.add_thread("old")
    r = rig(script)
    fresh = r.open()
    assert r.turn(fresh).resumed is False
    assert r.turn(fresh).resumed is True
    resumed = r.open(resume_id="old")
    assert r.turn(resumed).resumed is True
    fresh.close()
    resumed.close()


def test_the_turn_asks_codex_for_the_reasoning_effort_of_the_request_or_the_spec() -> None:
    r = rig()
    conversation = r.open(reasoning_effort="low")
    r.turn(conversation)
    r.turn(conversation, reasoning_effort="high")
    efforts = [req.params.to_wire().get("effort") for req in r.script.requests("turn/start")]
    assert efforts == ["low", "high"]
    conversation.close()


def test_a_request_cwd_overrides_the_conversations_for_that_turn() -> None:
    r = rig()
    conversation = r.open()
    r.turn(conversation, cwd="/elsewhere")
    assert _params(r.script, "turn/start")["cwd"] == "/elsewhere"
    conversation.close()


# -- structured output


_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def test_an_output_schema_is_sent_inline_and_the_answer_is_parsed(tmp_path: Path) -> None:
    script = CodexScript()
    script.turn(Say(json.dumps({"answer": 42})))
    r = rig(script)
    conversation = r.open()
    result = r.turn(conversation, output_schema=OutputSchema(_SCHEMA, tmp_path))
    assert _params(script, "turn/start")["outputSchema"] == _SCHEMA
    assert result.structured_output == {"answer": 42}
    assert list(tmp_path.iterdir()) == []  # nothing is written to disk
    conversation.close()


def test_an_answer_that_is_not_json_has_no_structured_output(tmp_path: Path) -> None:
    script = CodexScript()
    script.turn(Say("not json"))
    r = rig(script)
    conversation = r.open()
    result = r.turn(conversation, output_schema=OutputSchema(_SCHEMA, tmp_path))
    assert result.structured_output is None
    assert result.text == "not json"
    conversation.close()


def test_a_schema_outside_codexs_strict_dialect_is_rejected_before_anything_is_sent(
    tmp_path: Path,
) -> None:
    r = rig()
    conversation = r.open()
    loose = {"type": "object", "properties": {"a": {"type": "string"}}}
    with pytest.raises(SchemaDialectError):
        r.turn(conversation, output_schema=OutputSchema(loose, tmp_path))
    assert r.script.requests("turn/start") == []
    r.turn(conversation)  # the conversation is unharmed
    conversation.close()


def test_no_schema_means_no_structured_output() -> None:
    script = CodexScript()
    script.turn(Say('{"a": 1}'))
    r = rig(script)
    conversation = r.open()
    assert r.turn(conversation).structured_output is None
    conversation.close()


# -- token usage


def test_a_turns_usage_is_the_growth_of_the_threads_total() -> None:
    script = CodexScript()
    script.turn(
        Spend(
            input_tokens=1000, output_tokens=50, cached_input_tokens=600, reasoning_output_tokens=20
        )
    )
    script.turn(Spend(input_tokens=400, output_tokens=10, cached_input_tokens=100))
    r = rig(script)
    conversation = r.open()
    first = r.turn(conversation)
    second = r.turn(conversation)
    assert first.usage.tokens == TokenUsage(
        input_tokens=1000,
        output_tokens=50,
        cache_read_input_tokens=600,
        reasoning_output_tokens=20,
        turns=1,
    )
    assert second.usage.tokens == TokenUsage(
        input_tokens=400, output_tokens=10, cache_read_input_tokens=100, turns=1
    )
    assert second.usage.raw is not None
    assert second.usage.raw["input_tokens"] == 1400
    assert second.usage.provider == "codex"
    assert first.usage.increment_known
    conversation.close()


def test_a_turn_with_several_model_requests_adds_them_up() -> None:
    script = CodexScript()
    script.turn(
        Spend(input_tokens=100, output_tokens=5), Say("x"), Spend(input_tokens=150, output_tokens=7)
    )
    r = rig(script)
    conversation = r.open()
    assert r.turn(conversation).usage.tokens.input_tokens == 250
    conversation.close()


def test_a_resumed_conversation_continues_from_the_previous_reports_totals() -> None:
    script = CodexScript()
    script.add_thread("old")
    script.turn(Spend(input_tokens=300, output_tokens=30))
    r = rig(script)
    previous = ProviderUsage(
        tokens=TokenUsage(input_tokens=100, turns=1),
        provider="codex",
        raw={
            "input_tokens": 100,
            "cached_input_tokens": 0,
            "output_tokens": 10,
            "reasoning_output_tokens": 0,
            "cache_write_input_tokens": 0,
            "total_tokens": 110,
        },
    )
    conversation = r.open(resume_id="old", previous_usage=previous)
    usage = r.turn(conversation).usage
    # The fake thread started at zero, so its total after the turn is 300; the baseline was 100.
    assert usage.tokens.input_tokens == 200
    assert usage.increment_known
    conversation.close()


def test_a_resume_with_no_baseline_cannot_say_what_the_turn_cost_and_says_so() -> None:
    script = CodexScript()
    script.add_thread("old")
    script.turn(Spend(input_tokens=300, output_tokens=30))
    script.turn(Spend(input_tokens=50, output_tokens=5))
    r = rig(script)
    conversation = r.open(resume_id="old")
    first = r.turn(conversation)
    assert first.usage.increment_known is False
    assert first.usage.tokens == TokenUsage(turns=1)
    assert first.cost_usd is None
    second = r.turn(conversation)  # the first turn's total is the baseline from here on
    assert second.usage.increment_known
    assert second.usage.tokens.input_tokens == 50
    conversation.close()


def test_the_usage_replayed_when_a_thread_resumes_is_not_this_turns() -> None:
    script = CodexScript()
    script.add_thread("old")
    script.turn(Spend(input_tokens=10, output_tokens=1))
    r = rig(script)
    previous = ProviderUsage(provider="codex", raw={"input_tokens": 0, "output_tokens": 0})
    conversation = r.open(resume_id="old", previous_usage=previous)
    assert r.turn(conversation).usage.tokens.input_tokens == 10
    conversation.close()


def test_cost_comes_from_the_pricing_table_for_a_known_model() -> None:
    script = CodexScript()
    script.turn(Spend(input_tokens=1_000_000, output_tokens=0))
    r = rig(script)
    conversation = r.open(model="gpt-5")
    result = r.turn(conversation)
    assert result.cost_usd is not None
    assert result.cost_usd > 0
    assert r.of_type(UsageReport)[0].cost_usd == result.cost_usd  # type: ignore[attr-defined]
    conversation.close()


def test_an_unpriced_model_has_no_cost() -> None:
    r = rig()
    conversation = r.open(model="no-such-model")
    assert r.turn(conversation).cost_usd is None
    conversation.close()


@given(
    spends=st.lists(st.tuples(st.integers(0, 5000), st.integers(0, 500)), min_size=1, max_size=6),
    baseline=st.integers(0, 10_000),
)
def test_per_turn_usage_sums_to_the_growth_of_the_total(
    spends: list[tuple[int, int]], baseline: int
) -> None:
    script = CodexScript()
    script.add_thread("old")
    for input_tokens, output_tokens in spends:
        script.turn(Spend(input_tokens=input_tokens, output_tokens=output_tokens))
    r = rig(script)
    previous = ProviderUsage(provider="codex", raw={"input_tokens": 0, "output_tokens": 0})
    del baseline
    conversation = r.open(resume_id="old", previous_usage=previous)
    summed = TokenUsage()
    for _ in spends:
        summed += r.turn(conversation).usage.tokens
    assert summed.input_tokens == sum(i for i, _ in spends)
    assert summed.output_tokens == sum(o for _, o in spends)
    assert summed.turns == len(spends)
    conversation.close()


# -- failures


@pytest.mark.parametrize(
    ("info", "kind"),
    [
        ("usageLimitExceeded", FailureKind.USAGE_LIMIT),
        ("sessionBudgetExceeded", FailureKind.USAGE_LIMIT),
        ("serverOverloaded", FailureKind.TRANSIENT),
        ("rateLimitExceeded", FailureKind.TRANSIENT),
        ("internalServerError", FailureKind.TRANSIENT),
        ({"responseStreamDisconnected": {"httpStatusCode": 502}}, FailureKind.TRANSIENT),
        ({"httpConnectionFailed": {"httpStatusCode": None}}, FailureKind.TRANSIENT),
        ({"responseTooManyFailedAttempts": {"httpStatusCode": 429}}, FailureKind.TRANSIENT),
        ({"httpConnectionFailed": {"httpStatusCode": 401}}, FailureKind.AUTH),
        ("unauthorized", FailureKind.AUTH),
        ("badRequest", FailureKind.OTHER),
        ("contextWindowExceeded", FailureKind.OTHER),
        ("somethingNew", FailureKind.OTHER),
        ({"responseStreamConnectionFailed": {"httpStatusCode": 400}}, FailureKind.OTHER),
    ],
)
def test_a_failed_turn_is_classified_from_codex_error_info(
    info: str | dict[str, object], kind: FailureKind
) -> None:
    script = CodexScript()
    script.turn(Fail("it broke", info))
    r = rig(script)
    conversation = r.open()
    with pytest.raises(TurnFailedError) as caught:
        r.turn(conversation)
    assert caught.value.kind is kind
    assert "it broke" in caught.value.detail
    assert ProviderError("it broke") in r.events
    conversation.close()


def test_an_other_error_is_classified_by_what_its_message_says() -> None:
    script = CodexScript()
    script.turn(Fail("You've hit your usage limit. Try again later.", "other"))
    script.turn(Fail('{"status":400,"error":{"message":"bad model"}}', "other"))
    r = rig(script)
    conversation = r.open()
    with pytest.raises(TurnFailedError) as limit:
        r.turn(conversation)
    assert limit.value.kind is FailureKind.USAGE_LIMIT
    with pytest.raises(TurnFailedError) as other:
        r.turn(conversation)
    assert other.value.kind is FailureKind.OTHER
    conversation.close()


def test_a_retrying_error_is_not_the_end_of_the_turn() -> None:
    script = CodexScript()
    script.turn(Retrying("reconnecting 1/5"), Retrying("reconnecting 2/5"), Say("recovered"))
    r = rig(script)
    conversation = r.open()
    result = r.turn(conversation)
    assert result.text == "recovered"
    assert ProviderError("retrying: reconnecting 1/5") in r.events
    conversation.close()


def test_a_failed_turn_leaves_the_conversation_usable() -> None:
    script = CodexScript()
    script.turn(Fail("overloaded", "serverOverloaded"))
    script.turn(Say("fine"))
    r = rig(script)
    conversation = r.open()
    with pytest.raises(TurnFailedError):
        r.turn(conversation)
    assert r.turn(conversation).text == "fine"
    conversation.close()


def test_a_process_that_dies_mid_turn_fails_the_turn_with_its_last_stderr() -> None:
    script = CodexScript()
    script.turn(Say("starting", "commentary"), Crash(returncode=139, stderr="segfault in sandbox"))
    r = rig(script)
    conversation = r.open()
    with pytest.raises(TurnFailedError) as caught:
        r.turn(conversation)
    assert caught.value.kind is FailureKind.OTHER
    assert "139" in str(caught.value)
    assert "segfault in sandbox" in caught.value.detail
    assert r.turn(conversation).text == "ok"  # the next turn runs on a resumed process
    conversation.close()


def test_a_json_rpc_error_answering_turn_start_fails_the_turn() -> None:
    script = CodexScript()
    r = rig(script)
    conversation = r.open()
    script.forget("thread-1")
    with pytest.raises(TurnFailedError, match="thread not found"):
        r.turn(conversation)
    conversation.close()


# -- approvals and other server requests


@pytest.mark.parametrize(
    "permissions", [NativePermissions.read_only(), NativePermissions.workspace_write()]
)
def test_a_command_that_needs_approval_is_declined_under_deny_and_the_turn_goes_on(
    permissions: NativePermissions,
) -> None:
    script = CodexScript()
    script.turn(RunCommand("rm -rf build", output="gone", approval=True), Say("could not"))
    r = rig(script)
    conversation = r.open(permissions=permissions, approvals=ApprovalPolicy.DENY)
    result = r.turn(conversation)
    assert script.answers == [("item/commandExecution/requestApproval", {"decision": "decline"})]
    assert r.of_type(ApprovalDenied) == [ApprovalDenied("command", "rm -rf build")]
    (tool_result,) = r.of_type(ToolResult)
    assert tool_result.exit_code == 1  # the declined command is a failed tool
    assert result.text == "could not"
    conversation.close()


def test_bypass_never_asks_so_the_command_just_runs() -> None:
    script = CodexScript()
    script.turn(RunCommand("make", output="built", approval=True))
    r = rig(script)
    conversation = r.open(permissions=NativePermissions.bypass())
    r.turn(conversation)
    assert script.answers == []
    assert r.of_type(ApprovalDenied) == []
    assert r.of_type(ToolResult)[0].stdout == "built"
    conversation.close()


def test_a_file_change_that_needs_approval_is_declined_under_deny() -> None:
    script = CodexScript()
    script.turn(ChangeFile("src/a.py", approval=True), Say("skipped"))
    r = rig(script)
    conversation = r.open(permissions=NativePermissions.workspace_write())
    r.turn(conversation)
    assert script.answers == [("item/fileChange/requestApproval", {"decision": "decline"})]
    assert [d.kind for d in r.of_type(ApprovalDenied)] == ["file_change"]
    conversation.close()


@pytest.mark.parametrize(
    ("ask", "answer"),
    [
        (AskKind.PERMISSIONS, {"permissions": {}, "scope": "turn"}),
        (AskKind.ELICITATION, {"action": "decline"}),
        (AskKind.USER_INPUT, {"answers": {}}),
        (
            AskKind.LEGACY_EXEC,
            {"decision": {"denied": {"rejection": "refused by the agentshim approval policy"}}},
        ),
    ],
)
def test_every_kind_of_request_gets_a_refusal_under_deny(
    ask: AskKind, answer: dict[str, object]
) -> None:
    script = CodexScript()
    script.turn(Ask(ask), Say("moved on"))
    r = rig(script)
    conversation = r.open()
    result = r.turn(conversation)
    assert script.answers == [(ask.value, answer)]
    assert len(r.of_type(ApprovalDenied)) == 1
    assert result.text == "moved on"
    conversation.close()


@pytest.mark.parametrize("ask", [AskKind.UNKNOWN, AskKind.TOKEN_REFRESH])
def test_a_request_the_client_does_not_support_gets_a_json_rpc_error_not_silence(
    ask: AskKind,
) -> None:
    script = CodexScript()
    script.turn(Ask(ask), Say("moved on"))
    r = rig(script)
    conversation = r.open()
    result = r.turn(conversation)
    ((method, answer),) = script.answers
    assert method == ask.value
    assert answer["error"]["code"] == -32601  # type: ignore[index]
    assert r.of_type(ApprovalDenied) == []
    assert any(isinstance(e, Lifecycle) and e.kind == "unsupported_request" for e in r.events)
    assert result.text == "moved on"
    conversation.close()


@pytest.mark.parametrize(
    ("step", "method", "decision"),
    [
        (RunCommand("rm x", approval=True), "item/commandExecution/requestApproval", "cancel"),
        (ChangeFile("x", approval=True), "item/fileChange/requestApproval", "cancel"),
    ],
)
def test_fail_turn_cancels_the_request_and_fails_the_turn(
    step: RunCommand | ChangeFile, method: str, decision: str
) -> None:
    script = CodexScript()
    script.turn(step, Say("never said"))
    r = rig(script)
    conversation = r.open(
        permissions=NativePermissions.read_only(), approvals=ApprovalPolicy.FAIL_TURN
    )
    with pytest.raises(TurnFailedError) as caught:
        r.turn(conversation)
    assert script.answers == [(method, {"decision": decision})]
    assert caught.value.kind is FailureKind.OTHER
    assert "approval" in str(caught.value)
    assert r.of_type(ApprovalDenied) == []
    r.turn(conversation)  # the conversation survives: the next turn runs
    conversation.close()


def test_fail_turn_interrupts_a_turn_it_cannot_cancel_by_answering() -> None:
    script = CodexScript()
    script.turn(Ask(AskKind.USER_INPUT), Hang())
    r = rig(script)
    conversation = r.open(approvals=ApprovalPolicy.FAIL_TURN)
    with pytest.raises(TurnFailedError, match="user_input"):
        r.turn(conversation)
    assert len(script.requests("turn/interrupt")) == 1
    conversation.close()


@given(
    steps=st.lists(
        st.one_of(
            st.builds(Say, st.text(max_size=8), st.sampled_from(["final_answer", "commentary"])),
            st.builds(Think, st.text(min_size=1, max_size=8)),
            st.builds(
                RunCommand,
                st.text(min_size=1, max_size=8),
                st.just("out"),
                st.integers(0, 2),
                st.booleans(),
            ),
            st.builds(ChangeFile, st.text(min_size=1, max_size=8), st.booleans()),
            st.builds(Spend, st.integers(0, 100), st.integers(0, 100)),
            st.sampled_from([Ask(kind) for kind in AskKind]),
            st.just(Retrying()),
        ),
        max_size=8,
    ),
    permissions=st.sampled_from(
        [
            NativePermissions.bypass(),
            NativePermissions.read_only(),
            NativePermissions.workspace_write(),
        ]
    ),
    approvals=st.sampled_from(list(ApprovalPolicy)),
)
def test_whatever_a_turn_asks_every_server_request_is_answered_and_the_protocol_is_kept(
    steps: list[object], permissions: NativePermissions, approvals: ApprovalPolicy
) -> None:
    script = CodexScript()
    script.turn(*steps)  # type: ignore[arg-type]
    r = rig(script)
    conversation = r.open(permissions=permissions, approvals=approvals)
    try:
        r.turn(conversation)
    except TurnFailedError:
        assert approvals is ApprovalPolicy.FAIL_TURN  # nothing else fails a scripted turn
    assert script.violations == []
    asked = [
        s
        for s in steps
        if isinstance(s, Ask)
        or (
            isinstance(s, (RunCommand, ChangeFile))
            and s.approval
            and permissions.mode.value != "bypass"
        )
    ]
    if approvals is ApprovalPolicy.DENY:
        assert len(script.answers) == len(asked)  # one answer per request, none left hanging
    else:
        assert len(script.answers) <= len(asked)  # the first refusal ended the turn
    conversation.close()


# -- interrupt


def test_interrupting_a_running_turn_returns_an_interrupted_result() -> None:
    script = CodexScript()
    script.turn(Say("working", "commentary"), Hang())
    script.turn(Say("next"))
    r = rig(script)
    conversation = r.open()
    seen: list[AgentEvent] = []

    def emit(event: AgentEvent) -> None:
        seen.append(event)
        if isinstance(event, AssistantText):
            conversation.interrupt()

    result = conversation.turn(TurnRequest(prompt="go", timeout=SAFETY_TIMEOUT_S), emit)
    assert result.interrupted is True
    assert result.text == "working"
    assert TurnInterrupted() in seen
    (interrupt,) = script.requests("turn/interrupt")
    assert interrupt.params.to_wire()["threadId"] == "thread-1"
    assert r.turn(conversation).text == "next"  # kept the conversation
    conversation.close()


def test_an_interrupt_asked_before_the_turn_id_is_known_is_sent_once_it_is() -> None:
    script = CodexScript()
    script.turn(Hang())
    r = rig(script)
    conversation = r.open()

    def emit(event: AgentEvent) -> None:
        if isinstance(event, SessionStarted):  # first event, before turn/start is even sent
            conversation.interrupt()

    result = conversation.turn(TurnRequest(prompt="go", timeout=SAFETY_TIMEOUT_S), emit)
    assert result.interrupted is True
    assert len(script.requests("turn/interrupt")) == 1
    conversation.close()


def test_interrupting_an_idle_conversation_sends_nothing() -> None:
    r = rig()
    conversation = r.open()
    conversation.interrupt()
    assert r.script.requests("turn/interrupt") == []
    assert r.turn(conversation).interrupted is False
    conversation.close()


def test_interrupting_twice_sends_one_request() -> None:
    script = CodexScript()
    script.turn(Say("x", "commentary"), Hang())
    r = rig(script)
    conversation = r.open()

    def emit(event: AgentEvent) -> None:
        if isinstance(event, AssistantText):
            conversation.interrupt()
            conversation.interrupt()

    conversation.turn(TurnRequest(prompt="go", timeout=SAFETY_TIMEOUT_S), emit)
    assert len(script.requests("turn/interrupt")) == 1
    conversation.close()


# -- timeouts


def test_a_turn_that_overruns_its_timeout_is_interrupted_and_the_conversation_kept() -> None:
    script = CodexScript()
    script.turn(Hang())
    script.turn(Say("after"))
    r = rig(script)
    conversation = r.open()
    with pytest.raises(TurnTimeoutError) as caught:
        r.turn(conversation, timeout=5.0)
    assert caught.value.timeout == 5.0
    assert len(script.requests("turn/interrupt")) == 1
    assert r.turn(conversation).text == "after"
    conversation.close()


def test_a_server_that_ignores_the_interrupt_writes_the_conversation_off() -> None:
    script = CodexScript()
    script.turn(Hang(ignores_interrupt=True))
    r = rig(script, interrupt_grace=3.0)
    conversation = r.open()
    with pytest.raises(TurnTimeoutError):
        r.turn(conversation, timeout=2.0)
    with pytest.raises(TurnFailedError, match="did not stop"):
        r.turn(conversation)
    conversation.close()


# -- MCP servers


def test_mcp_servers_are_given_to_the_thread_and_re_sent_on_resume() -> None:
    script = CodexScript()
    script.add_thread("old")
    servers = (
        StdioMcpServer("issue-tracker", "/bin/mcp", ("--db", "x"), {"TOKEN": "t"}, 30.0, 60.0),
        HttpMcpServer("docs", "https://mcp.example/docs", {"Authorization": "Bearer x"}),
    )
    r = rig(script)
    r.open(mcp_servers=servers).close()
    r.open(mcp_servers=servers, resume_id="old").close()
    expected = {
        "mcp_servers": {
            "issue_tracker": {
                "required": True,
                "command": "/bin/mcp",
                "args": ["--db", "x"],
                "env": {"TOKEN": "t"},
                "startup_timeout_sec": 30.0,
                "tool_timeout_sec": 60.0,
            },
            "docs": {
                "required": True,
                "url": "https://mcp.example/docs",
                "http_headers": {"Authorization": "Bearer x"},
            },
        }
    }
    assert _params(script, "thread/start")["config"] == expected
    assert _params(script, "thread/resume")["config"] == expected


def test_a_conversation_with_no_mcp_servers_sends_no_config() -> None:
    r = rig()
    r.open().close()
    assert "config" not in _params(r.script, "thread/start")


def test_an_sse_server_is_refused_because_codex_cannot_reach_it() -> None:
    r = rig()
    with pytest.raises(ProviderCapabilityError, match="sse"):
        r.open(mcp_servers=(HttpMcpServer("old", "https://x/sse", transport="sse"),))
    assert r.executor.spawns == []  # refused before any process started


def test_two_servers_that_collapse_to_one_key_are_refused() -> None:
    r = rig()
    servers = (StdioMcpServer("a-b", "x"), StdioMcpServer("a_b", "y"))
    with pytest.raises(ProviderCapabilityError, match="a_b"):
        r.open(mcp_servers=servers)


def test_a_configured_server_that_fails_to_start_fails_the_turn_with_its_error() -> None:
    script = CodexScript()
    script.fail_mcp_server("tracker", "No such file or directory (os error 2)")
    r = rig(script)
    conversation = r.open(mcp_servers=(StdioMcpServer("tracker", "/nonexistent/mcp"),))
    with pytest.raises(TurnFailedError) as caught:
        r.turn(conversation)
    assert "tracker" in str(caught.value)
    assert "No such file" in caught.value.detail
    with pytest.raises(TurnFailedError, match="tracker"):  # and the next turn cannot pretend
        r.turn(conversation)
    conversation.close()


def test_a_failing_server_nobody_configured_does_not_matter() -> None:
    script = CodexScript()
    script.fail_mcp_server("codex_apps", "no account")
    r = rig(script)
    conversation = r.open(mcp_servers=(StdioMcpServer("tracker", "/bin/mcp"),))
    assert r.turn(conversation).text == "ok"
    conversation.close()


def test_mcp_servers_cannot_change_mid_conversation() -> None:
    r = rig()
    conversation = r.open(mcp_servers=(StdioMcpServer("tracker", "/bin/mcp"),))
    with pytest.raises(ProviderCapabilityError, match="fixed"):
        r.turn(conversation, mcp_servers=(StdioMcpServer("other", "/bin/x"),))
    r.turn(conversation, mcp_servers=(StdioMcpServer("tracker", "/bin/mcp"),))
    conversation.close()


def test_extra_args_and_env_cannot_reach_a_running_server() -> None:
    r = rig()
    conversation = r.open()
    with pytest.raises(ProviderCapabilityError, match="extra_args"):
        r.turn(conversation, extra_args=("--x",))
    with pytest.raises(ProviderCapabilityError, match="environment"):
        r.turn(conversation, env={"A": "b"})
    conversation.close()


# -- scopes


def test_session_mcp_scope_hides_the_users_servers_but_not_the_ones_given(tmp_path: Path) -> None:
    from agentshim import McpScope  # noqa: PLC0415

    home = tmp_path / "home"
    (home / ".codex").mkdir(parents=True)
    (home / ".codex" / "config.toml").write_text(
        '[mcp_servers.mine]\ncommand = "/bin/mine"\n[mcp_servers.tracker]\ncommand = "/bin/user"\n'
    )
    r = rig()
    r.transport._env["HOME"] = str(home)  # noqa: SLF001 - point the scan at the fixture home
    servers = (StdioMcpServer("tracker", "/bin/tracker"),)
    r.open(mcp_servers=servers, mcp_scope=McpScope.SESSION).close()
    argv = list(r.executor.spawns[0].argv)
    assert "mcp_servers.mine.enabled=false" in argv
    assert "mcp_servers.tracker.enabled=false" not in argv
    assert "features.plugins=false" in argv


def test_project_config_scope_is_not_offered() -> None:
    from agentshim import ConfigScope  # noqa: PLC0415

    r = rig()
    with pytest.raises(ProviderCapabilityError, match="scope"):
        r.open(config_scope=ConfigScope.PROJECT)


# -- closing


def test_close_ends_stdin_and_the_server_exits_without_being_killed() -> None:
    r = rig()
    conversation = r.open()
    conversation.close()
    process = r.executor.processes[0]
    assert process.stdin_closed
    assert (process.terminate_calls, process.kill_calls) == (0, 0)
    assert r.script.closed == 1


def test_a_server_that_ignores_end_of_input_is_terminated() -> None:
    r = rig(CodexScript(exits_at_end_of_input=False))
    conversation = r.open()
    conversation.close()
    process = r.executor.processes[0]
    assert process.terminate_calls == 1
    assert process.kill_calls == 0


def test_a_turn_after_close_is_a_state_error() -> None:
    r = rig()
    conversation = r.open()
    conversation.close()
    with pytest.raises(SessionStateError):
        r.turn(conversation)


def test_a_second_turn_while_one_runs_is_a_state_error() -> None:
    script = CodexScript()
    script.turn(Say("a", "commentary"), Hang())
    r = rig(script)
    conversation = r.open()
    outcome: list[str] = []

    def emit(event: AgentEvent) -> None:
        if isinstance(event, AssistantText):
            try:
                conversation.turn(TurnRequest(prompt="second"), lambda _e: None)
            except SessionStateError:
                outcome.append("refused")
            conversation.interrupt()

    conversation.turn(TurnRequest(prompt="go", timeout=SAFETY_TIMEOUT_S), emit)
    assert outcome == ["refused"]
    conversation.close()


def test_each_open_is_its_own_process_and_thread() -> None:
    r = rig()
    first, second = r.open(), r.open()
    assert len(r.executor.spawns) == 2
    assert first.conversation_id != second.conversation_id
    first.close()
    assert r.turn(second).text == "ok"
    second.close()


def test_the_binary_is_checked_once_at_construction() -> None:
    r = rig()
    assert r.executor.checked == ["/usr/local/bin/codex"]
    r.open().close()
    r.open().close()
    assert r.executor.checked == ["/usr/local/bin/codex"]


def test_spec_helper_builds_what_the_tests_think_it_does() -> None:
    assert spec().model == "fake-model"


# -- a process that died is replaced by resuming its thread


def test_a_process_killed_between_turns_is_replaced_by_resuming_its_thread() -> None:
    script = CodexScript()
    script.turn(Say("one")).turn(Say("two"))
    r = rig(script)
    conversation = r.open()
    first = r.turn(conversation)
    thread_id = conversation.conversation_id
    r.executor.processes[0].kill()  # died while idle
    second = r.turn(conversation)
    assert second.text == "two"
    assert script.spawned == 2
    assert conversation.conversation_id == thread_id == first.session_id
    assert _params(script, "thread/resume")["threadId"] == thread_id
    assert second.resumed is True
    conversation.close()
    assert r.executor.processes[1].stdin_closed


def test_a_process_that_crashed_mid_turn_fails_that_turn_and_the_next_one_resumes() -> None:
    script = CodexScript()
    script.turn(Say("start"), Crash(returncode=9, stderr="boom")).turn(Say("again"))
    r = rig(script)
    conversation = r.open()
    with pytest.raises(TurnFailedError, match="boom"):
        r.turn(conversation)
    assert r.turn(conversation).text == "again"
    assert script.spawned == 2
    assert len(script.requests("thread/resume")) == 1
    conversation.close()


def test_a_replacement_keeps_the_usage_baseline_of_the_thread() -> None:
    script = CodexScript()
    script.turn(Spend(input_tokens=100, output_tokens=5)).turn(
        Spend(input_tokens=40, output_tokens=2)
    )
    r = rig(script)
    conversation = r.open()
    r.turn(conversation)
    r.executor.processes[0].kill()
    usage = r.turn(conversation).usage
    assert usage.tokens.input_tokens == 40
    assert usage.increment_known
    conversation.close()


def test_a_replacement_that_codex_refuses_to_resume_is_a_session_resume_error() -> None:
    script = CodexScript()
    r = rig(script)
    conversation = r.open()
    r.turn(conversation)
    thread_id = conversation.conversation_id
    assert thread_id is not None
    script.forget(thread_id)
    r.executor.processes[0].kill()
    with pytest.raises(SessionResumeError) as caught:
        r.turn(conversation)
    assert caught.value.session_id == thread_id
    assert r.executor.processes[1].stdin_closed  # the process that refused was shut down
    conversation.close()


def test_a_conversation_closed_cannot_be_revived_by_a_dead_process() -> None:
    r = rig()
    conversation = r.open()
    r.turn(conversation)
    r.executor.processes[0].kill()
    conversation.close()
    with pytest.raises(SessionStateError):
        r.turn(conversation)
    assert r.script.spawned == 1
