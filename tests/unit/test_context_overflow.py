"""A turn the model's context window cannot hold fails as ``CONTEXT_OVERFLOW``.

Every case below is the CLI's own recorded output (``tests/fixtures``), or a
line the CLI wrote in a production run where the stream itself was not kept.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from agentshim import CliAgent, CliExitError, FailureKind, SessionResumeError
from agentshim.providers.claude.failures import classify_failure as classify_claude
from agentshim.providers.codex.failures import classify_failure as classify_codex
from agentshim.testing import FakeExecutor, FakeRun

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_ENV = {"PATH": "/usr/bin:/bin", "HOME": "/home/tester"}

#: What a Claude Code build in a production run (VibeSys finding S185) wrote
#: when a resumed conversation outgrew the window and the CLI's own
#: compaction then failed on a gateway error.
_COMPACTION_FAILED = (
    "Prompt is too long · automatic compaction failed: API Error: 503 No available "
    "accounts: no available accounts. This is a server-side issue, usually temporary — "
    "try again in a moment."
)


def _lines(provider: str, name: str) -> list[str]:
    return (FIXTURES / provider / name).read_text().splitlines(keepends=True)


def _fail(provider: str, run: FakeRun, *, resume: str | None = None) -> CliExitError:
    agent = CliAgent(provider, executor=FakeExecutor(run), env=dict(_ENV))
    session = agent.start_session(session_id=resume)
    with pytest.raises(CliExitError) as excinfo:
        session.turn("hi")
    return excinfo.value


_RECORDED = [
    # Claude Code's own estimate is over its blocking limit: no request is sent.
    ("claude", "haiku45_prompt_too_long_blocking_limit.jsonl"),
    # The API refused the request with a 400.
    ("claude", "haiku45_prompt_too_long_api_400.jsonl"),
    # ``codex exec``: the Responses API's ``context_length_exceeded``.
    ("codex", "exec_context_overflow_gpt6_luna.jsonl"),
]


@pytest.mark.parametrize(("provider", "name"), _RECORDED)
@pytest.mark.parametrize("resume", [None, "s-1"])
def test_a_recorded_overflow_is_a_context_overflow(
    provider: str, name: str, resume: str | None
) -> None:
    """A resumed turn that overflows still has its conversation: it is not a lost one."""
    error = _fail(provider, FakeRun(stdout=_lines(provider, name), returncode=1), resume=resume)

    assert not isinstance(error, SessionResumeError)
    assert error.kind is FailureKind.CONTEXT_OVERFLOW
    assert error.detail


def test_claude_reads_the_terminal_reason_before_any_text() -> None:
    """The CLI's ``terminal_reason`` decides, whatever the message says."""
    for reason in ("prompt_too_long", "blocking_limit", "rapid_refill_breaker"):
        kind = classify_claude(
            subtype="success",
            api_error="invalid_request",
            status=400,
            text="something else entirely",
            terminal_reason=reason,
        )
        assert kind is FailureKind.CONTEXT_OVERFLOW


def test_an_older_claude_build_is_read_from_its_text() -> None:
    """A build without ``terminal_reason`` names the overflow only in its text.

    The production text also carries a 503, which alone would be transient:
    the overflow is what a retry of the same conversation cannot outlast.
    """
    result = {
        "type": "result",
        "subtype": "success",
        "is_error": True,
        "result": _COMPACTION_FAILED,
        "num_turns": 1,
    }
    message = {"role": "assistant", "content": [{"type": "text", "text": _COMPACTION_FAILED}]}
    assistant = {"type": "assistant", "message": message, "error": "invalid_request"}
    run = FakeRun(stdout=[json.dumps(assistant) + "\n", json.dumps(result) + "\n"], returncode=1)

    assert _fail("claude", run, resume="s-1").kind is FailureKind.CONTEXT_OVERFLOW


def test_claude_api_wording_of_the_overflow_is_recognized() -> None:
    kind = classify_claude(
        subtype=None,
        api_error=None,
        status=None,
        text='API Error: 400 {"type":"error","error":{"type":"invalid_request_error",'
        '"message":"prompt is too long: 213411 tokens > 200000 maximum"}}',
        terminal_reason=None,
    )
    assert kind is FailureKind.CONTEXT_OVERFLOW


def test_a_tool_that_prints_the_phrase_does_not_fail_the_turn() -> None:
    """Only the stream's error text is read: tool output never is."""
    call = {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "cat log"}}
    output = {"type": "tool_result", "tool_use_id": "t1", "content": "Prompt is too long"}
    frames = [
        {"type": "assistant", "message": {"role": "assistant", "content": [call]}},
        {"type": "user", "message": {"role": "user", "content": [output]}},
        {"type": "result", "subtype": "success", "is_error": False, "result": "done"},
    ]
    run = FakeRun(stdout=[json.dumps(frame) + "\n" for frame in frames])
    agent = CliAgent("claude", executor=FakeExecutor(run), env=dict(_ENV))
    assert agent.start_session().turn("hi").text == "done"


@pytest.mark.parametrize(
    "text",
    [
        # Codex's own message for its ``ContextWindowExceeded`` error.
        "Codex ran out of room in the model's context window. Start a new thread or clear "
        "earlier history before retrying.",
        # The Responses API body, as ``codex exec`` relays it (recorded above).
        '{"type":"error","error":{"message":"Your input exceeds the context window of this '
        'model. Please adjust your input and try again.","type":"invalid_request_error",'
        '"param":"input","code":"context_length_exceeded"},"status":400,"sequence_number":3}',
    ],
)
def test_codex_text_naming_the_overflow_is_a_context_overflow(text: str) -> None:
    assert classify_codex(text) is FailureKind.CONTEXT_OVERFLOW


def test_codex_reads_the_error_code_not_a_phrase_in_a_message() -> None:
    """A message that only mentions the code is not the code."""
    body = json.dumps(
        {
            "type": "error",
            "error": {"message": "context_length_exceeded is not this", "code": "bad_request"},
            "status": 400,
        }
    )
    assert classify_codex(body) is FailureKind.OTHER
