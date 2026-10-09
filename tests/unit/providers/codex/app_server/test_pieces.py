"""The small pure parts of the Codex app-server transport."""

from __future__ import annotations

import pytest
from agentshim import ApprovalPolicy, FailureKind, NativePermissions, ProviderCapabilityError
from agentshim.providers.codex.app_server import protocol as p
from agentshim.providers.codex.app_server.approvals import answer_request
from agentshim.providers.codex.app_server.channel import POLL_S, Deadline
from agentshim.providers.codex.app_server.errors import classify_turn_error
from agentshim.providers.codex.app_server.permissions import check_applied, codex_permissions
from agentshim.providers.codex.app_server.transport import final_text
from agentshim.providers.codex.app_server.usage import (
    baseline_from,
    turn_usage,
    zero_totals,
)
from agentshim.testing import FakeClock
from hypothesis import given
from hypothesis import strategies as st


def _error(info: object, message: str = "boom") -> p.TurnError:
    return p.TurnError(message=message, codex_error_info=info)  # type: ignore[arg-type]


# -- failure kinds


@given(
    info=st.one_of(
        st.none(),
        st.sampled_from(list(p.CodexErrorInfoKind)),
        st.builds(p.UnknownCodexErrorInfo, st.text(max_size=8)),
        st.builds(
            p.CodexErrorInfoHttpConnectionFailed, http_status_code=st.none() | st.integers(0, 700)
        ),
        st.builds(
            p.CodexErrorInfoResponseStreamDisconnected,
            http_status_code=st.none() | st.integers(0, 700),
        ),
    ),
    message=st.text(max_size=40),
)
def test_every_turn_error_has_a_failure_kind(info: object, message: str) -> None:
    assert isinstance(classify_turn_error(_error(info, message)), FailureKind)


@given(status=st.integers(500, 599) | st.sampled_from([408, 429]) | st.none())
def test_a_lost_connection_is_transient_unless_the_server_said_no(status: int | None) -> None:
    info = p.CodexErrorInfoResponseStreamDisconnected(http_status_code=status)
    assert classify_turn_error(_error(info)) is FailureKind.TRANSIENT


@given(status=st.sampled_from([401, 403]))
def test_a_lost_connection_that_ended_in_401_or_403_is_an_auth_failure(status: int) -> None:
    info = p.CodexErrorInfoHttpConnectionFailed(http_status_code=status)
    assert classify_turn_error(_error(info)) is FailureKind.AUTH


def test_structured_info_wins_over_what_the_message_happens_to_say() -> None:
    info = p.CodexErrorInfoKind.USAGE_LIMIT_EXCEEDED
    assert classify_turn_error(_error(info, "rate limit, overloaded")) is FailureKind.USAGE_LIMIT


# -- permissions


@given(
    roots=st.lists(st.from_regex(r"/[a-z]{1,6}", fullmatch=True), max_size=3),
    network=st.booleans(),
)
def test_what_the_mapping_asks_for_is_what_the_echo_check_accepts(
    roots: list[str], *, network: bool
) -> None:
    for permissions in (
        NativePermissions.bypass(),
        NativePermissions.read_only(),
        NativePermissions.workspace_write(roots, network=network),
    ):
        mapped = codex_permissions(permissions)
        check_applied(mapped, mapped.policy, mapped.approval)


def test_a_read_only_sandbox_with_the_network_on_is_not_what_was_asked() -> None:
    mapped = codex_permissions(NativePermissions.read_only())
    with pytest.raises(ProviderCapabilityError, match="network"):
        check_applied(mapped, p.SandboxPolicyReadOnly(network_access=True), mapped.approval)


def test_a_different_sandbox_is_named_in_the_error() -> None:
    mapped = codex_permissions(NativePermissions.read_only())
    with pytest.raises(ProviderCapabilityError, match="DangerFullAccess"):
        check_applied(mapped, p.SandboxPolicyDangerFullAccess(), mapped.approval)


# -- approvals


def _request(params: p.ServerRequestParams, method: str) -> p.ServerRequest:
    return p.ServerRequest(id=7, method=method, params=params)


@pytest.mark.parametrize("policy", list(ApprovalPolicy))
def test_every_request_type_is_answered_with_the_requests_own_id(policy: ApprovalPolicy) -> None:
    requests = [
        _request(
            p.CommandExecutionRequestApprovalParams(
                item_id="i", started_at_ms=1, thread_id="t", turn_id="u", command="ls"
            ),
            "item/commandExecution/requestApproval",
        ),
        _request(
            p.FileChangeRequestApprovalParams(
                item_id="i", started_at_ms=1, thread_id="t", turn_id="u"
            ),
            "item/fileChange/requestApproval",
        ),
        _request(
            p.McpServerElicitationRequestParams(server_name="s", thread_id="t"),
            "mcpServer/elicitation/request",
        ),
        _request(p.UnknownParams(raw={}), "something/new"),
    ]
    for request in requests:
        answer = answer_request(request, policy)
        assert answer.response.to_wire()["id"] == 7


def test_deny_declines_and_fail_turn_cancels_a_command() -> None:
    request = _request(
        p.CommandExecutionRequestApprovalParams(
            item_id="i", started_at_ms=1, thread_id="t", turn_id="u", command="rm x"
        ),
        "item/commandExecution/requestApproval",
    )
    deny = answer_request(request, ApprovalPolicy.DENY)
    cancel = answer_request(request, ApprovalPolicy.FAIL_TURN)
    assert deny.response.to_wire()["result"] == {"decision": "decline"}  # type: ignore[union-attr]
    assert cancel.response.to_wire()["result"] == {"decision": "cancel"}  # type: ignore[union-attr]
    assert deny.refusal is not None
    assert (deny.refusal.kind, deny.refusal.detail) == ("command", "rm x")


# -- the deadline


def test_a_deadline_counts_silent_reads_even_when_the_clock_stands_still() -> None:
    clock = FakeClock()
    deadline = Deadline(clock, 3.0)
    assert not deadline.expired()
    for _ in range(3):
        wait = deadline.next_wait()
        assert wait == pytest.approx(min(POLL_S, 3.0))
        deadline.silence(wait)
    assert deadline.expired()


def test_a_deadline_follows_the_clock_when_it_moves() -> None:
    clock = FakeClock()
    deadline = Deadline(clock, 10.0)
    clock.advance(9.5)
    assert not deadline.expired()
    assert deadline.next_wait() == pytest.approx(0.5)
    clock.advance(0.5)
    assert deadline.expired()


def test_a_deadline_without_a_limit_never_expires_and_polls_at_the_slice() -> None:
    deadline = Deadline(FakeClock(), None)
    for _ in range(1000):
        deadline.silence(deadline.next_wait())
    assert not deadline.expired()
    assert deadline.next_wait() == POLL_S


# -- the answer


def _message(text: str, phase: str | None) -> p.ThreadItemAgentMessage:
    return p.ThreadItemAgentMessage(id="m", text=text, phase=phase)


@given(
    messages=st.lists(
        st.tuples(st.text(max_size=5), st.sampled_from(["final_answer", "commentary", None])),
        max_size=6,
    )
)
def test_the_answer_is_the_last_final_message_else_the_last_non_commentary_else_the_last(
    messages: list[tuple[str, str | None]],
) -> None:
    items = [_message(text, phase) for text, phase in messages]
    finals = [t for t, ph in messages if ph == "final_answer"]
    plain = [t for t, ph in messages if ph != "commentary"]
    expected = (finals or plain or [t for t, _ in messages] or [""])[-1]
    assert final_text(items) == expected


# -- usage


@given(
    base=st.integers(0, 10**6),
    grow=st.integers(0, 10**6),
    cached=st.integers(0, 10**6),
)
def test_a_turns_increment_is_total_minus_baseline_and_never_invalid(
    base: int, grow: int, cached: int
) -> None:
    baseline = zero_totals()
    baseline["input_tokens"] = base
    total = dict(baseline)
    total["input_tokens"] = base + grow
    total["cached_input_tokens"] = min(cached, base + grow)
    usage = turn_usage(total, baseline)
    assert usage.tokens.input_tokens == grow
    assert usage.tokens.cache_read_input_tokens <= usage.tokens.input_tokens
    assert usage.increment_known


def test_a_usage_report_from_another_provider_is_no_baseline() -> None:
    from agentshim import ProviderUsage  # noqa: PLC0415

    assert baseline_from(ProviderUsage(provider="claude", raw={"input_tokens": 5})) is None
    assert baseline_from(ProviderUsage(provider="codex", raw=None)) is None
    assert baseline_from(None) is None
    baseline = baseline_from(ProviderUsage(provider="codex", raw={"input_tokens": 5}))
    assert baseline is not None
    assert baseline["input_tokens"] == 5
