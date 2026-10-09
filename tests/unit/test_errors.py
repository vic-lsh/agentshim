"""The exception hierarchy callers catch on."""

from __future__ import annotations

import pytest
from agentshim import (
    AgentShimError,
    CliCheckError,
    CliExitError,
    CliNotFoundError,
    CliTimeoutError,
    ContinuityError,
    FailureKind,
    McpConfigError,
    ProviderCapabilityError,
    SchemaDialectError,
    SessionResumeError,
    SessionStateError,
    TurnCancelledError,
    TurnFailedError,
    TurnTimeoutError,
)


@pytest.mark.parametrize(
    "error",
    [
        CliNotFoundError("claude"),
        CliCheckError("/bin/claude", "broken"),
        CliExitError(["claude"], 1),
        SessionResumeError(["claude"], 1, "s1"),
        CliTimeoutError(["claude"], 30.0),
        ProviderCapabilityError("nope"),
        SchemaDialectError(["bad"]),
        McpConfigError("bad config"),
        TurnFailedError("x"),
        TurnTimeoutError(1.0),
        ContinuityError("a", None),
        SessionStateError("x"),
        TurnCancelledError("x"),
    ],
)
def test_every_error_is_an_agentshim_error(error: Exception) -> None:
    assert isinstance(error, AgentShimError)


def test_resume_failure_is_an_exit_error() -> None:
    error = SessionResumeError(["claude", "--resume", "s1"], 1, "s1", stderr="gone")
    assert isinstance(error, CliExitError)
    assert error.session_id == "s1"
    assert error.returncode == 1
    assert error.stderr == "gone"
    assert "s1" in str(error)


def test_schema_dialect_error_is_a_capability_error() -> None:
    error = SchemaDialectError(["uses allOf", "non-local $ref"])
    assert isinstance(error, ProviderCapabilityError)
    assert error.problems == ["uses allOf", "non-local $ref"]
    assert "allOf" in str(error)


def test_exit_error_keeps_the_argv_and_streams() -> None:
    error = CliExitError(["claude", "-p"], 3, "out", "err")
    assert error.argv == ("claude", "-p")
    assert (error.stdout, error.stderr) == ("out", "err")
    assert "code 3" in str(error)


def test_timeout_error_keeps_the_budget() -> None:
    error = CliTimeoutError(["claude"], 12.5)
    assert error.timeout == 12.5
    assert "12.5" in str(error)


def test_not_found_error_names_the_binary() -> None:
    error = CliNotFoundError("claude")
    assert error.binary == "claude"
    assert "claude" in str(error)


def test_turn_failed_error_carries_a_kind_and_detail() -> None:
    error = TurnFailedError("boom", kind=FailureKind.AUTH, detail="login expired")
    assert (error.kind, error.detail) == (FailureKind.AUTH, "login expired")
    assert str(error) == "boom"
    assert TurnFailedError("x").kind is FailureKind.OTHER


def test_cli_exit_error_is_a_turn_failed_error_with_its_fields_intact() -> None:
    error = CliExitError(["codex"], 2, "o", "e", kind=FailureKind.TRANSIENT, detail="overloaded")
    assert isinstance(error, TurnFailedError)
    assert (error.kind, error.detail) == (FailureKind.TRANSIENT, "overloaded")
    assert "overloaded" in str(error)
    assert (error.argv, error.returncode) == (("codex",), 2)


def test_a_resume_error_is_a_turn_failed_error_of_kind_other() -> None:
    error = SessionResumeError(["claude"], 1, "s1")
    assert isinstance(error, TurnFailedError)
    assert error.kind is FailureKind.OTHER


def test_cli_timeout_error_is_a_turn_timeout_error() -> None:
    error = CliTimeoutError(["claude"], 3.0)
    assert isinstance(error, TurnTimeoutError)
    assert error.timeout == 3.0
    assert error.partial is None
    assert TurnTimeoutError(2.0).timeout == 2.0


def test_continuity_error_names_both_conversations() -> None:
    error = ContinuityError("want", "have")
    assert (error.expected, error.actual) == ("want", "have")
    assert "want" in str(error)
