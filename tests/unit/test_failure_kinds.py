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


@pytest.mark.parametrize("provider", _CLASSIFYING_PROVIDERS)
@pytest.mark.parametrize("kind", list(FailureKind))
def test_a_failed_turn_carries_its_kind_and_the_provider_text(
    provider: str, kind: FailureKind
) -> None:
    error = _fail(provider, scripted_failure(provider, kind))

    assert error.kind is kind
    assert error.detail
    assert error.detail in str(error)


@pytest.mark.parametrize("provider", _CLASSIFYING_PROVIDERS)
@pytest.mark.parametrize("kind", [FailureKind.TRANSIENT, FailureKind.USAGE_LIMIT, FailureKind.AUTH])
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
        _claude_result(
            subtype="error_max_structured_output_retries",
            errors=["no StructuredOutput call produced a valid output"],
        ),
    )

    assert "no StructuredOutput call produced a valid output" in str(error)
    assert error.kind is FailureKind.OTHER


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
