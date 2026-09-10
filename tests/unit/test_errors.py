"""The exception hierarchy callers catch on."""

from __future__ import annotations

import pytest
from agentshim import (
    AgentShimError,
    CliCheckError,
    CliExitError,
    CliNotFoundError,
    CliTimeoutError,
    McpConfigError,
    ProviderCapabilityError,
    SchemaDialectError,
    SessionResumeError,
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
