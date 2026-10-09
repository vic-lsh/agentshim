"""Every line of the recorded app-server sessions parses (server) or round-trips (client)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from agentshim.providers.codex.app_server import protocol as p

from .conftest import FIXTURES

SESSIONS = ["session_1", "session_2", "session_3"]
SERVER_REPLY_TYPES: dict[type, type] = {
    p.CommandExecutionRequestApprovalParams: p.CommandExecutionRequestApprovalResponse,
}
EXTRA_KEY = "zzNewServerField"


@dataclass
class Replay:
    """What one transcript parsed into."""

    client: list[p.ClientMessage] = field(default_factory=list)
    server: list[p.ServerMessage] = field(default_factory=list)
    results: dict[int | str, Any] = field(default_factory=dict)
    replies: list[Any] = field(default_factory=list)

    def notifications(self, params_type: type) -> list[Any]:
        return [
            m.params
            for m in self.server
            if isinstance(m, p.Notification) and isinstance(m.params, params_type)
        ]


def lines(name: str) -> list[tuple[str, Any]]:
    path = FIXTURES / f"{name}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return [(row["dir"], row["msg"]) for row in rows]


def contains(actual: Any, expected: Any) -> bool:
    """True when ``actual`` has everything ``expected`` has (it may add defaults)."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            k in actual and contains(actual[k], v) for k, v in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(contains(a, e) for a, e in zip(actual, expected, strict=True))
        )
    return bool(actual == expected)


def with_extra_keys(value: Any) -> Any:
    """Add an unknown key to every object, the way a newer server would."""
    if isinstance(value, dict):
        return {**{k: with_extra_keys(v) for k, v in value.items()}, EXTRA_KEY: 1}
    if isinstance(value, list):
        return [with_extra_keys(v) for v in value]
    return value


def replay(name: str) -> Replay:
    out = Replay()
    pending: dict[int | str, Any] = {}
    server_requests: dict[int | str, p.ServerRequest] = {}
    for direction, raw in lines(name):
        if direction == "out":
            client = p.parse_client_message(raw)
            assert contains(client.to_wire(), raw), f"client line changed: {raw}"
            assert p.parse_client_message(client.to_wire()) == client
            out.client.append(client)
            if isinstance(client, p.ClientRequest):
                pending[client.id] = client.params
            elif isinstance(client, p.Response):
                reply_type = SERVER_REPLY_TYPES[type(server_requests[client.id].params)]
                out.replies.append(reply_type.from_wire(client.result))
        else:
            server = p.parse_server_message(raw)
            assert p.parse_server_message(server.to_wire()) == server
            out.server.append(server)
            if isinstance(server, p.ServerRequest):
                server_requests[server.id] = server
            elif isinstance(server, p.Response):
                out.results[server.id] = pending[server.id].parse_result(server.result)
    return out


@pytest.mark.parametrize("name", SESSIONS)
class TestEveryLine:
    def test_replays_without_error(self, name: str) -> None:
        result = replay(name)
        assert len(result.client) + len(result.server) == len(lines(name))

    def test_server_lines_with_unknown_keys_still_parse_to_the_same_messages(
        self, name: str
    ) -> None:
        for direction, raw in lines(name):
            if direction != "in":
                continue
            plain = p.parse_server_message(raw)
            noisy = p.parse_server_message(with_extra_keys(raw))
            if isinstance(plain, p.Notification) and isinstance(noisy, p.Notification):
                # free-form JSON members (config, arguments) keep the extra key; the typed
                # shape must still be decoded
                assert type(noisy.params) is type(plain.params)
            else:
                assert type(noisy) is type(plain)

    def test_only_unlisted_methods_fall_back_to_unknown_params(self, name: str) -> None:
        unlisted = {
            "remoteControl/status/changed",
            "account/updated",
            "thread/settings/updated",
            "thread/goal/cleared",
        }
        for message in replay(name).server:
            if isinstance(message, p.Notification):
                assert isinstance(message.params, p.UnknownParams) == (message.method in unlisted)

    def test_every_notification_carries_its_emission_time(self, name: str) -> None:
        notifications = [m for m in replay(name).server if isinstance(m, p.Notification)]
        assert notifications
        assert all(isinstance(n.emitted_at_ms, int) for n in notifications)


class TestSession1:
    def test_handshake_and_thread_ids(self) -> None:
        result = replay("session_1")
        init = result.results[1]
        assert isinstance(init, p.InitializeResponse)
        assert init.user_agent.startswith("agentshim-research/0.160.0")
        first, second = result.results[5], result.results[7]
        assert isinstance(first, p.ThreadStartResponse)
        assert first.thread.id == "01a11e5a-f97d-71e3-b244-0f4148254a2b"
        assert second.thread.id == "01a11e5b-08aa-73d0-beb6-6d5a150c4784"
        assert isinstance(first.sandbox, p.SandboxPolicyDangerFullAccess)
        assert second.sandbox == p.SandboxPolicyReadOnly(network_access=False)
        assert first.approval_policy == p.AskForApprovalKind.NEVER

    def test_resume_failures_are_error_responses_with_the_exact_message(self) -> None:
        errors = [m for m in replay("session_1").server if isinstance(m, p.ErrorResponse)]
        assert [e.id for e in errors] == [3, 4]
        assert errors[0].error.code == -32600
        assert errors[0].error.message == (
            "no rollout found for thread id 00000000-0000-0000-0000-000000000000"
        )
        assert errors[1].error.message.startswith("invalid session id")

    def test_turns_complete_and_the_final_answers_are_read_from_agent_messages(self) -> None:
        result = replay("session_1")
        completed = result.notifications(p.TurnCompletedNotification)
        assert [c.turn.status for c in completed] == [p.TurnStatus.COMPLETED] * 2
        finals = [
            n.item.text
            for n in result.notifications(p.ItemCompletedNotification)
            if isinstance(n.item, p.ThreadItemAgentMessage)
            and n.item.phase == p.MessagePhase.FINAL_ANSWER
        ]
        assert finals == ["ok", "two"]

    def test_token_usage_is_decoded(self) -> None:
        usage = replay("session_1").notifications(p.ThreadTokenUsageUpdatedNotification)[0]
        total = usage.token_usage.total
        assert (total.total_tokens, total.input_tokens, total.cached_input_tokens) == (
            13639,
            13634,
            11008,
        )
        assert (total.output_tokens, total.reasoning_output_tokens) == (5, 0)
        assert usage.token_usage.model_context_window == 258400

    def test_requests_sent_are_the_typed_forms(self) -> None:
        sent = [m for m in replay("session_1").client if isinstance(m, p.ClientRequest)]
        start = next(m.params for m in sent if isinstance(m.params, p.ThreadStartParams))
        assert start == p.ThreadStartParams(
            cwd="$CWD",
            model="gpt-6-luna",
            sandbox=p.SandboxMode.DANGER_FULL_ACCESS,
            approval_policy=p.AskForApprovalKind.NEVER,
            ephemeral=False,
        )


class TestSession2:
    def test_declined_command_approval_is_a_server_request_and_a_typed_reply(self) -> None:
        result = replay("session_2")
        request = next(m for m in result.server if isinstance(m, p.ServerRequest))
        assert request.id == 0
        assert request.method == "item/commandExecution/requestApproval"
        assert isinstance(request.params, p.CommandExecutionRequestApprovalParams)
        assert request.params.command == "/bin/bash -lc 'echo hi'"
        assert request.params.proposed_execpolicy_amendment == ("echo", "hi")
        assert result.replies == [
            p.CommandExecutionRequestApprovalResponse(
                decision=p.CommandExecutionApprovalDecisionKind.DECLINE
            )
        ]
        resolved = result.notifications(p.ServerRequestResolvedNotification)
        assert [r.request_id for r in resolved] == [0]

    def test_turn_statuses_and_final_texts(self) -> None:
        result = replay("session_2")
        statuses = [c.turn.status for c in result.notifications(p.TurnCompletedNotification)]
        assert statuses == [
            p.TurnStatus.COMPLETED,
            p.TurnStatus.COMPLETED,
            p.TurnStatus.INTERRUPTED,
        ]
        finals = [
            n.item.text
            for n in result.notifications(p.ItemCompletedNotification)
            if isinstance(n.item, p.ThreadItemAgentMessage)
            and n.item.phase == p.MessagePhase.FINAL_ANSWER
        ]
        assert finals == ["The command was rejected, so I couldn\u2019t run it.", "done"]

    def test_command_execution_items_carry_status_and_output(self) -> None:
        completed = replay("session_2").notifications(p.ItemCompletedNotification)
        commands = [n.item for n in completed if isinstance(n.item, p.ThreadItemCommandExecution)]
        assert [c.status for c in commands] == [
            p.CommandExecutionStatus.DECLINED,
            p.CommandExecutionStatus.COMPLETED,
        ]
        assert commands[1].aggregated_output == "hi2\n"
        assert commands[1].exit_code == 0

    def test_token_usage_is_cumulative_for_the_thread(self) -> None:
        updates = replay("session_2").notifications(p.ThreadTokenUsageUpdatedNotification)
        totals = [u.token_usage.total.total_tokens for u in updates]
        assert totals == sorted(totals)
        assert totals[0] == 14977
        assert totals[-1] == 76058

    def test_interrupt_names_the_turn_that_ended_interrupted(self) -> None:
        result = replay("session_2")
        interrupt = next(
            m.params
            for m in result.client
            if isinstance(m, p.ClientRequest) and isinstance(m.params, p.TurnInterruptParams)
        )
        assert isinstance(result.results[7], p.TurnInterruptResponse)
        last = result.notifications(p.TurnCompletedNotification)[-1]
        assert last.turn.id == interrupt.turn_id
        assert last.turn.status == p.TurnStatus.INTERRUPTED
        assert last.turn.error is None


class TestSession3:
    def test_resume_returns_the_same_thread_with_overridden_sandbox(self) -> None:
        result = replay("session_3")
        resumed = result.results[3]
        assert isinstance(resumed, p.ThreadResumeResponse)
        assert resumed.thread.id == "01a11e5b-5231-7d01-b2a0-6c2169de27e7"
        assert isinstance(resumed.sandbox, p.SandboxPolicyDangerFullAccess)

    def test_structured_output_turn_returns_json_text(self) -> None:
        result = replay("session_3")
        texts = [
            n.item.text
            for n in result.notifications(p.ItemCompletedNotification)
            if isinstance(n.item, p.ThreadItemAgentMessage)
        ]
        assert json.loads(texts[0]) == {"command": "echo hi"}
        sent = next(
            m.params
            for m in result.client
            if isinstance(m, p.ClientRequest)
            and isinstance(m.params, p.TurnStartParams)
            and m.params.output_schema is not None
        )
        assert isinstance(sent.output_schema, dict)

    def test_failed_turn_carries_the_error_and_no_items(self) -> None:
        result = replay("session_3")
        failed = result.notifications(p.TurnCompletedNotification)[-1].turn
        assert failed.status == p.TurnStatus.FAILED
        assert failed.items == ()
        assert failed.error is not None
        assert failed.error.codex_error_info == p.CodexErrorInfoKind.OTHER
        assert "no-such-model-xyz" in failed.error.message
        errors = result.notifications(p.ErrorNotification)
        assert [e.will_retry for e in errors] == [False]
        assert errors[0].error == failed.error

    def test_failed_mcp_server_startup_is_visible(self) -> None:
        statuses = replay("session_3").notifications(p.McpServerStatusUpdatedNotification)
        bogus = [s for s in statuses if s.name == "bogus"]
        assert [s.status for s in bogus] == [
            p.McpServerStartupState.STARTING,
            p.McpServerStartupState.FAILED,
        ]
        assert bogus[-1].error is not None
        assert "failed to start" in bogus[-1].error

    def test_per_thread_mcp_config_is_sent_as_free_form_json(self) -> None:
        start = next(
            m.params
            for m in replay("session_3").client
            if isinstance(m, p.ClientRequest)
            and isinstance(m.params, p.ThreadStartParams)
            and m.params.config is not None
        )
        assert start.config == {
            "mcp_servers": {"bogus": {"command": "/nonexistent/mcp", "args": ["--x"]}}
        }
