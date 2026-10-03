"""A failed turn's ``CliExitError`` says why it failed, through ``kind``.

Callers choose a retry policy from ``FailureKind`` alone, so these tests pin
what each provider's real failure report classifies as, that the error text
reaches the exception message, and that a classified failure of a resumed
turn is not mistaken for a lost conversation.
"""

from __future__ import annotations

import json

import pytest
from agentshim import CliAgent, CliExitError, FailureKind, SessionResumeError
from agentshim.testing import (
    FakeExecutor,
    FakeRun,
    scripted_failure,
    scripted_resume_failure,
    scripted_turn,
)
from hypothesis import given
from hypothesis import strategies as st

_ENV = {"PATH": "/usr/bin:/bin", "HOME": "/home/tester"}
_CLASSIFYING_PROVIDERS = ("claude", "codex")


def _fail(provider: str, run: FakeRun, *, resume: str | None = None) -> CliExitError:
    agent = CliAgent(provider, executor=FakeExecutor(run), env=dict(_ENV))
    session = agent.start_session(session_id=resume)
    with pytest.raises(CliExitError) as excinfo:
        session.turn("hi")
    return excinfo.value


def _claude_result(**fields: object) -> FakeRun:
    """A Claude run whose only frame is an error ``result`` with *fields*."""
    frame = {"type": "result", "is_error": True, **fields}
    return FakeRun(stdout=[json.dumps(frame) + "\n"], returncode=1)


_SCRIPTABLE = [
    (provider, kind)
    for provider in _CLASSIFYING_PROVIDERS
    for kind in FailureKind
    if not (provider == "codex" and kind is FailureKind.SCHEMA)
]


@pytest.mark.parametrize(("provider", "kind"), _SCRIPTABLE)
def test_a_failed_turn_carries_its_kind_and_the_provider_text(
    provider: str, kind: FailureKind
) -> None:
    error = _fail(provider, scripted_failure(provider, kind))

    assert error.kind is kind
    assert error.detail
    assert error.detail in str(error)


@pytest.mark.parametrize(
    ("provider", "kind"), [(p, k) for p, k in _SCRIPTABLE if k is not FailureKind.OTHER]
)
def test_a_classified_failure_of_a_resumed_turn_keeps_the_conversation(
    provider: str, kind: FailureKind
) -> None:
    """An overload says nothing about whether the conversation still exists."""
    agent = CliAgent(
        provider, executor=FakeExecutor(scripted_failure(provider, kind)), env=dict(_ENV)
    )
    session = agent.start_session(session_id="s-1")

    with pytest.raises(CliExitError) as excinfo:
        session.turn("hi")

    assert not isinstance(excinfo.value, SessionResumeError)
    assert excinfo.value.kind is kind
    assert session.session_id == "s-1"


@pytest.mark.parametrize("provider", _CLASSIFYING_PROVIDERS)
def test_a_lost_conversation_is_still_a_resume_failure(provider: str) -> None:
    error = _fail(provider, scripted_resume_failure(provider), resume="gone")

    assert isinstance(error, SessionResumeError)
    assert error.kind is FailureKind.OTHER


def test_an_error_subtype_with_no_result_text_still_says_what_failed() -> None:
    """Regression: Claude reports this in the stream, so stderr was empty and so was the message."""
    error = _fail(
        "claude",
        _claude_result(subtype="error_during_execution", errors=["the turn stopped unexpectedly"]),
    )

    assert "the turn stopped unexpectedly" in str(error)
    assert error.kind is FailureKind.OTHER


def test_codex_has_no_schema_failure_to_script() -> None:
    """Codex constrains decoding to the schema, so no real run fails this way."""
    with pytest.raises(ValueError, match="constrained"):
        scripted_failure("codex", FailureKind.SCHEMA)


def _rejection(index: int, errors: str, tool: str = "StructuredOutput") -> list[str]:
    call = {"type": "tool_use", "id": f"t{index}", "name": tool, "input": {}}
    result = {
        "type": "tool_result",
        "tool_use_id": f"t{index}",
        "content": errors,
        "is_error": True,
    }
    return [
        json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [call]}})
        + "\n",
        json.dumps({"type": "user", "message": {"role": "user", "content": [result]}}) + "\n",
    ]


_VALIDATION_ERRORS = st.text(
    alphabet=st.characters(codec="utf-8", exclude_categories=("Cs", "Cc")), min_size=1
).filter(str.strip)


@given(rejections=st.lists(_VALIDATION_ERRORS, min_size=1, max_size=5))
def test_used_up_schema_retries_carry_the_last_validation_errors(rejections: list[str]) -> None:
    """Regression: r10's planner failed its schema five times and the run saw no reason."""
    lines = [line for index, errors in enumerate(rejections) for line in _rejection(index, errors)]
    lines.append(
        json.dumps(
            {"type": "result", "is_error": True, "subtype": "error_max_structured_output_retries"}
        )
        + "\n"
    )

    error = _fail("claude", FakeRun(stdout=lines, returncode=1))

    assert error.kind is FailureKind.SCHEMA
    assert error.detail == rejections[-1]
    assert rejections[-1].strip() in str(error)


def test_used_up_schema_retries_without_a_rejection_still_classify() -> None:
    error = _fail("claude", _claude_result(subtype="error_max_structured_output_retries"))

    assert error.kind is FailureKind.SCHEMA
    assert "error_max_structured_output_retries" in error.detail


def test_only_a_structured_output_rejection_is_reported_as_the_validation_errors() -> None:
    """A failed Bash call that prints schema-like text must not become the detail."""
    lines = [
        *_rejection(0, "the real validation errors"),
        *_rejection(1, "Output does not match required schema: jq", tool="Bash"),
        json.dumps(
            {"type": "result", "is_error": True, "subtype": "error_max_structured_output_retries"}
        )
        + "\n",
    ]

    error = _fail("claude", FakeRun(stdout=lines, returncode=1))

    assert error.detail == "the real validation errors"


_TRANSIENT_STATUSES = st.sampled_from([408, 429]) | st.integers(500, 599)


@given(status=_TRANSIENT_STATUSES)
def test_an_older_claude_api_error_in_result_text_is_transient(status: int) -> None:
    """Builds without ``api_error_status`` only say ``API Error: <status>`` in the result."""
    error = _fail("claude", _claude_result(subtype="success", result=f"API Error: {status} oops"))

    assert error.kind is FailureKind.TRANSIENT


@given(status=_TRANSIENT_STATUSES)
def test_a_claude_api_error_status_is_transient(status: int) -> None:
    error = _fail(
        "claude", _claude_result(subtype="success", result="failed", api_error_status=status)
    )

    assert error.kind is FailureKind.TRANSIENT


@pytest.mark.parametrize("status", [401, 403])
def test_a_claude_auth_status_is_auth(status: int) -> None:
    error = _fail(
        "claude", _claude_result(subtype="success", result="denied", api_error_status=status)
    )

    assert error.kind is FailureKind.AUTH


def test_a_claude_429_that_is_a_quota_is_a_usage_limit() -> None:
    error = _fail(
        "claude",
        _claude_result(
            subtype="success", result="You've hit your limit · resets 3pm", api_error_status=429
        ),
    )

    assert error.kind is FailureKind.USAGE_LIMIT


@pytest.mark.parametrize("provider", _CLASSIFYING_PROVIDERS)
def test_tool_output_quoting_an_api_error_is_not_classified(provider: str) -> None:
    """Only the provider's own error report decides, never what a tool printed."""
    failed = scripted_turn(
        provider,
        text="the build broke",
        tool_calls=[("Bash", {"command": "make"}, "API Error: 529 overloaded_error")],
        returncode=1,
    )

    assert _fail(provider, failed).kind is FailureKind.OTHER


@pytest.mark.parametrize("provider", ["copilot", "gemini", "opencode"])
def test_a_provider_that_reports_no_failure_kind_classifies_as_other(provider: str) -> None:
    error = _fail(provider, FakeRun(returncode=1, stderr=["API Error: 529\n"]))

    assert error.kind is FailureKind.OTHER
