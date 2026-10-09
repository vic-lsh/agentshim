"""Decoding rules and exact wire output, by example."""

from __future__ import annotations

import pytest
from agentshim.core.errors import AgentShimError
from agentshim.providers.codex.app_server import INITIALIZED, CodexProtocolError
from agentshim.providers.codex.app_server import protocol as p

TURN = {"id": "t1", "items": [], "status": "inProgress"}


class TestOutboundWireForm:
    def test_initialize_uses_camel_case_and_omits_unset_members(self) -> None:
        request = p.ClientRequest(
            id=1,
            params=p.InitializeParams(
                client_info=p.ClientInfo(name="agentshim", version="0.15"),
                capabilities=p.InitializeCapabilities(
                    experimental_api=True, request_attestation=False
                ),
            ),
        )
        wire = request.to_wire()
        assert wire["method"] == "initialize"
        assert wire["id"] == 1
        assert wire["params"] == {
            "clientInfo": {"name": "agentshim", "version": "0.15"},
            "capabilities": {"experimentalApi": True, "requestAttestation": False},
        }

    def test_initialized_has_no_id_and_no_params(self) -> None:
        assert INITIALIZED.to_wire() == {"method": "initialized"}

    def test_thread_start_matches_the_recorded_request_exactly(self) -> None:
        request = p.ClientRequest(
            id=5,
            params=p.ThreadStartParams(
                cwd="/repo",
                model="gpt-6-luna",
                sandbox=p.SandboxMode.DANGER_FULL_ACCESS,
                approval_policy=p.AskForApprovalKind.NEVER,
                ephemeral=False,
            ),
        )
        assert request.to_wire() == {
            "id": 5,
            "method": "thread/start",
            "params": {
                "approvalPolicy": "never",
                "cwd": "/repo",
                "ephemeral": False,
                "model": "gpt-6-luna",
                "sandbox": "danger-full-access",
            },
        }

    def test_turn_start_carries_tagged_input_and_a_per_turn_sandbox_policy(self) -> None:
        params = p.TurnStartParams(
            thread_id="th",
            input=(p.UserInputText(text="hi", text_elements=()),),
            sandbox_policy=p.SandboxPolicyWorkspaceWrite(
                writable_roots=("/work",), network_access=True
            ),
            output_schema={"type": "object"},
        )
        assert params.to_wire() == {
            "threadId": "th",
            "input": [{"type": "text", "text": "hi", "text_elements": []}],
            "sandboxPolicy": {
                "type": "workspaceWrite",
                "writableRoots": ["/work"],
                "networkAccess": True,
                "excludeSlashTmp": False,
                "excludeTmpdirEnvVar": False,
            },
            "outputSchema": {"type": "object"},
        }

    def test_approval_decisions_encode_as_bare_strings_and_single_key_objects(self) -> None:
        decline = p.CommandExecutionRequestApprovalResponse(
            decision=p.CommandExecutionApprovalDecisionKind.DECLINE
        )
        assert p.reply(0, decline).to_wire() == {"id": 0, "result": {"decision": "decline"}}
        amend = p.CommandExecutionApprovalDecisionAcceptWithExecpolicyAmendment(
            execpolicy_amendment=("echo", "hi")
        )
        assert amend.to_wire() == {
            "acceptWithExecpolicyAmendment": {"execpolicy_amendment": ["echo", "hi"]}
        }

    def test_every_client_request_type_knows_its_method(self) -> None:
        assert p.InitializeParams.METHOD == "initialize"
        assert p.ThreadResumeParams.METHOD == "thread/resume"
        assert p.TurnInterruptParams.METHOD == "turn/interrupt"

    def test_construction_is_keyword_only_and_frozen(self) -> None:
        params = p.TurnInterruptParams(thread_id="a", turn_id="b")
        with pytest.raises(AttributeError):
            params.thread_id = "c"  # type: ignore[misc]
        with pytest.raises(TypeError):
            p.TurnInterruptParams("a", "b")  # type: ignore[misc]


class TestInboundTolerance:
    def test_unknown_keys_are_ignored_at_every_level(self) -> None:
        wire = {
            "method": "turn/started",
            "params": {"threadId": "th", "turn": {**TURN, "futureField": {"x": 1}}, "extra": 1},
            "emittedAtMs": 5,
            "alsoNew": True,
        }
        message = p.parse_server_message(wire)
        assert isinstance(message, p.Notification)
        assert isinstance(message.params, p.TurnStartedNotification)
        assert message.params.turn.id == "t1"

    def test_unknown_enum_value_survives_as_the_raw_string(self) -> None:
        turn = p.Turn.from_wire({**TURN, "status": "teleported"})
        assert turn.status == "teleported"
        assert not isinstance(turn.status, p.TurnStatus)

    def test_known_enum_value_becomes_the_member(self) -> None:
        assert p.Turn.from_wire(TURN).status is p.TurnStatus.IN_PROGRESS

    def test_unknown_tagged_variant_keeps_the_whole_object(self) -> None:
        raw = {"type": "holographicOutput", "id": "x", "payload": [1, 2]}
        turn = p.Turn.from_wire({**TURN, "items": [raw]})
        assert turn.items == (p.UnknownThreadItem(raw=raw),)
        assert turn.items[0].to_wire() == raw

    def test_known_tagged_variant_is_selected_by_its_tag(self) -> None:
        item = p.thread_item_from_wire(
            {"type": "agentMessage", "id": "m", "text": "ok", "phase": "final_answer"}
        )
        assert item == p.ThreadItemAgentMessage(
            id="m", text="ok", phase=p.MessagePhase.FINAL_ANSWER
        )

    def test_unknown_error_info_keeps_the_raw_value(self) -> None:
        assert p.codex_error_info_from_wire("brandNewKind") == p.UnknownCodexErrorInfo(
            raw="brandNewKind"
        )
        assert p.codex_error_info_from_wire({"brandNew": {"x": 1}}) == p.UnknownCodexErrorInfo(
            raw={"brandNew": {"x": 1}}
        )

    def test_error_info_variants_decode(self) -> None:
        assert p.codex_error_info_from_wire("serverOverloaded") is (
            p.CodexErrorInfoKind.SERVER_OVERLOADED
        )
        decoded = p.codex_error_info_from_wire(
            {"responseStreamDisconnected": {"httpStatusCode": 503}}
        )
        assert decoded == p.CodexErrorInfoResponseStreamDisconnected(http_status_code=503)

    def test_unknown_notification_method_keeps_its_params(self) -> None:
        message = p.parse_server_message({"method": "future/thing", "params": {"a": 1}})
        assert message == p.Notification(method="future/thing", params=p.UnknownParams({"a": 1}))

    def test_unknown_server_request_is_still_a_request_the_caller_must_answer(self) -> None:
        message = p.parse_server_message({"id": 7, "method": "future/ask", "params": {"q": 1}})
        assert message == p.ServerRequest(
            id=7, method="future/ask", params=p.UnknownParams({"q": 1})
        )

    def test_request_ids_may_be_strings(self) -> None:
        message = p.parse_server_message({"id": "abc", "result": {}})
        assert message == p.Response(id="abc", result={})


class TestMalformedInput:
    def test_missing_required_field_names_the_path(self) -> None:
        wire = {
            "method": "turn/started",
            "params": {"threadId": "th", "turn": {"items": [], "status": "x"}},
        }
        with pytest.raises(CodexProtocolError) as caught:
            p.parse_server_message(wire)
        assert caught.value.path == "Notification.params.turn"
        assert "'id'" in caught.value.problem

    def test_wrongly_typed_field_names_the_path_and_types(self) -> None:
        with pytest.raises(CodexProtocolError, match=r"Turn\.id: expected string, got number"):
            p.Turn.from_wire({**TURN, "id": 5})

    def test_booleans_are_not_integers(self) -> None:
        with pytest.raises(CodexProtocolError, match="expected integer, got boolean"):
            p.ErrorResponse.from_wire({"id": 1, "error": {"code": True, "message": "m"}})

    def test_nested_list_item_errors_carry_the_index(self) -> None:
        with pytest.raises(CodexProtocolError, match=r"\.items\[1\]"):
            p.Turn.from_wire({**TURN, "items": [{"type": "plan", "id": "a", "text": "t"}, 3]})

    def test_the_error_is_an_agentshim_error(self) -> None:
        with pytest.raises(AgentShimError):
            p.parse_server_message([])

    @pytest.mark.parametrize("wire", [{}, {"id": 1}, {"jsonrpc": "2.0"}, 3, "x", None])
    def test_envelope_without_method_result_or_error_is_rejected(self, wire: object) -> None:
        with pytest.raises(CodexProtocolError):
            p.parse_server_message(wire)

    def test_a_message_that_is_only_a_result_without_an_id_is_rejected(self) -> None:
        with pytest.raises(CodexProtocolError):
            p.parse_server_message({"result": {}})

    def test_client_message_for_an_unlisted_method_is_rejected(self) -> None:
        with pytest.raises(CodexProtocolError, match="thread/fork"):
            p.parse_client_message({"id": 1, "method": "thread/fork", "params": {}})


class TestOutboundStrictness:
    def test_a_sandbox_mode_the_schema_does_not_list_is_rejected(self) -> None:
        with pytest.raises(CodexProtocolError, match="danger-everything"):
            p.ThreadStartParams.from_wire({"sandbox": "danger-everything"})

    def test_a_decision_the_schema_does_not_list_is_rejected(self) -> None:
        with pytest.raises(CodexProtocolError):
            p.CommandExecutionRequestApprovalResponse.from_wire({"decision": "maybe"})

    def test_union_used_only_by_the_client_has_no_unknown_variant(self) -> None:
        assert not hasattr(p, "UnknownCommandExecutionApprovalDecision")
        assert hasattr(p, "UnknownThreadItem")
