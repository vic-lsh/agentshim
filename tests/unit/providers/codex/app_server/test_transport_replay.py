"""The transport against the real CLI's own recorded output.

Each test serves a stretch of a recorded session (``replay.py``) to a real
``CodexAppServerTransport`` and checks both directions: the client must send
what the recording's client sent (modulo what the transport chooses for itself),
and what the server said must come out as the right events, result and errors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from agentshim import (
    ApprovalDenied,
    ApprovalPolicy,
    AssistantText,
    FailureKind,
    NativePermissions,
    OutputSchema,
    ProviderUsage,
    SessionResumeError,
    StdioMcpServer,
    TokenUsage,
    ToolCall,
    ToolResult,
    TurnFailedError,
    TurnInterrupted,
    TurnRequest,
)
from agentshim.providers.codex.app_server import CodexAppServerTransport
from agentshim.testing import FakeClock, FakeExecutor

from tests.unit.providers.codex.app_server.harness import ENV, SAFETY_TIMEOUT_S, spec
from tests.unit.providers.codex.app_server.replay import Entry, ReplayPeer, load

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from agentshim import AgentEvent, Conversation

THREAD_OF_SESSION_2 = "01a11e5b-5231-7d01-b2a0-6c2169de27e7"


def _served(
    entries: Sequence[Entry],
    *,
    skip: frozenset[str] = frozenset(),
    tolerated: frozenset[str] = frozenset(),
) -> tuple[CodexAppServerTransport, ReplayPeer]:
    peer = ReplayPeer(entries, skip=skip, tolerated=tolerated)
    executor = FakeExecutor([], peers=lambda _request: peer)
    transport = CodexAppServerTransport(executor=executor, env=dict(ENV), clock=FakeClock())
    return transport, peer


def _open(transport: CodexAppServerTransport, **changes: object) -> Conversation:
    changes.setdefault("model", "gpt-6-luna")
    return transport.open(spec(cwd="$CWD", **changes))


def _turn(
    conversation: Conversation, prompt: str, events: list[AgentEvent], **request: object
) -> object:
    request.setdefault("timeout", SAFETY_TIMEOUT_S)
    return conversation.turn(TurnRequest(prompt=prompt, **request), events.append)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def s1() -> list[Entry]:
    return load("1")


@pytest.fixture(scope="module")
def s2() -> list[Entry]:
    return load("2")


@pytest.fixture(scope="module")
def s3() -> list[Entry]:
    return load("3")


def test_resuming_a_thread_codex_never_had_is_a_refused_resume(s1: list[Entry]) -> None:
    transport, peer = _served([*s1[0:3], *s1[3:6]])
    with pytest.raises(SessionResumeError) as caught:
        _open(transport, resume_id="00000000-0000-0000-0000-000000000000")
    assert "no rollout found for thread id" in caught.value.detail
    assert peer.finished


def test_resuming_a_malformed_thread_id_is_a_refused_resume(s1: list[Entry]) -> None:
    transport, peer = _served([*s1[0:3], *s1[6:8]])
    with pytest.raises(SessionResumeError, match="invalid session id"):
        _open(transport, resume_id="not-a-uuid")
    assert peer.finished


def test_a_plain_turn_answers_and_reports_the_threads_usage(s1: list[Entry]) -> None:
    transport, peer = _served([*s1[0:3], *s1[8:28]])
    conversation = _open(transport, reasoning_effort="low")
    events: list[AgentEvent] = []
    result = _turn(conversation, "reply with the word ok", events)
    assert result.text == "ok"  # type: ignore[attr-defined]
    assert result.session_id == "01a11e5a-f97d-71e3-b244-0f4148254a2b"  # type: ignore[attr-defined]
    assert AssistantText("ok") in events
    usage = result.usage  # type: ignore[attr-defined]
    assert usage.increment_known
    assert usage.tokens == TokenUsage(
        input_tokens=13634, output_tokens=5, cache_read_input_tokens=11008, turns=1
    )
    assert peer.finished
    conversation.close()


def test_a_declined_approval_a_run_command_and_an_interrupt_in_one_conversation(
    s2: list[Entry],
) -> None:
    transport, peer = _served(s2[0:104])
    conversation = _open(
        transport,
        permissions=NativePermissions.workspace_write(),
        approvals=ApprovalPolicy.DENY,
        reasoning_effort="low",
    )
    events: list[AgentEvent] = []

    first = _turn(conversation, "Run the shell command: echo hi . Then say done.", events)
    assert ApprovalDenied("command", "/bin/bash -lc 'echo hi'") in events
    declined = [e for e in events if isinstance(e, ToolResult)]
    assert declined[0].exit_code == 1  # a declined command is a failed tool
    assert "rejected" in first.text  # type: ignore[attr-defined]
    assert first.interrupted is False  # type: ignore[attr-defined]

    events.clear()
    second = _turn(conversation, "Run the shell command: echo hi2 . Then say done.", events)
    ran = [e for e in events if isinstance(e, ToolResult)]
    assert ran
    assert ran[0].stdout == "hi2\n"
    assert ran[0].exit_code == 0
    assert second.usage.increment_known  # type: ignore[attr-defined]
    assert second.usage.tokens.input_tokens > 0  # type: ignore[attr-defined]

    events.clear()

    def interrupt_the_sleep(event: AgentEvent) -> None:
        events.append(event)
        if isinstance(event, ToolCall) and "sleep 60" in str(event.args):
            conversation.interrupt()

    third = conversation.turn(
        TurnRequest(prompt="Run the shell command: sleep 60", timeout=SAFETY_TIMEOUT_S),
        interrupt_the_sleep,
    )
    assert third.interrupted is True
    assert TurnInterrupted() in events
    assert peer.finished
    conversation.close()


def test_a_thread_resumes_in_a_new_process_and_answers_in_the_schema(
    s3: list[Entry], tmp_path: Path
) -> None:
    transport, peer = _served([*s3[0:3], *s3[3:30]])
    schema = s3[8].message["params"]["outputSchema"]  # type: ignore[index]
    # What the previous process last reported: the thread's total at resume.
    resume_total = s3[9].message["params"]["tokenUsage"]["total"]  # type: ignore[index]
    previous = ProviderUsage(
        provider="codex",
        raw={
            "input_tokens": resume_total["inputTokens"],
            "cached_input_tokens": resume_total["cachedInputTokens"],
            "cache_write_input_tokens": resume_total["cacheWriteInputTokens"],
            "output_tokens": resume_total["outputTokens"],
            "reasoning_output_tokens": resume_total["reasoningOutputTokens"],
            "total_tokens": resume_total["totalTokens"],
        },
    )
    conversation = _open(
        transport, resume_id=THREAD_OF_SESSION_2, previous_usage=previous, reasoning_effort="low"
    )
    events: list[AgentEvent] = []
    result = _turn(
        conversation,
        "What shell command did I ask you to run first? Answer in the schema.",
        events,
        output_schema=OutputSchema(schema, tmp_path),
    )
    assert result.structured_output == {"command": "echo hi"}  # type: ignore[attr-defined]
    assert result.resumed is True  # type: ignore[attr-defined]
    after = s3[26].message["params"]["tokenUsage"]["total"]  # type: ignore[index]
    # The usage the server replayed at resume belongs to the previous turn, not this one.
    assert result.usage.tokens.input_tokens == after["inputTokens"] - resume_total["inputTokens"]  # type: ignore[attr-defined]
    assert result.usage.tokens.input_tokens == 16023  # type: ignore[attr-defined]
    assert peer.finished
    conversation.close()


def test_a_server_that_fails_to_start_a_configured_mcp_server_fails_the_turn(
    s3: list[Entry],
) -> None:
    # The research driver let the turn run on; the transport interrupts it, once.
    transport, peer = _served(
        [*s3[0:3], *s3[30:47]], skip=frozenset({"model"}), tolerated=frozenset({"turn/interrupt"})
    )
    conversation = _open(
        transport, mcp_servers=(StdioMcpServer("bogus", "/nonexistent/mcp", ("--x",)),)
    )
    with pytest.raises(TurnFailedError) as caught:
        conversation.turn(TurnRequest(prompt="hi", timeout=SAFETY_TIMEOUT_S), lambda _e: None)
    assert "bogus" in str(caught.value)
    assert "No such file or directory" in caught.value.detail
    assert sum(m.get("method") == "turn/interrupt" for m in peer.sent) == 1
    assert peer.finished


def test_a_turn_the_model_provider_rejects_fails_with_what_it_said(s3: list[Entry]) -> None:
    # The recorded thread was started with a server nobody configured: its failed
    # start-up is none of the turn's business, and the rejected model is.
    transport, peer = _served([*s3[0:3], *s3[30:47]], skip=frozenset({"model", "config"}))
    conversation = _open(transport)
    with pytest.raises(TurnFailedError) as caught:
        conversation.turn(TurnRequest(prompt="hi", timeout=SAFETY_TIMEOUT_S), lambda _e: None)
    assert caught.value.kind is FailureKind.OTHER
    assert "no-such-model-xyz" in caught.value.detail
    assert peer.finished
