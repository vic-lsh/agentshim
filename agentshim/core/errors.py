"""Exception hierarchy. Nothing outside this hierarchy escapes ``turn()``."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class AgentShimError(Exception):
    """Base class for every error agentshim raises."""


class CliNotFoundError(AgentShimError):
    """The provider binary is not on PATH."""

    def __init__(self, binary: str, message: str | None = None) -> None:
        self.binary = binary
        super().__init__(message or f"{binary} binary not found in PATH")


class CliCheckError(AgentShimError):
    """The provider binary was found but its health check failed."""

    def __init__(self, binary_path: str, message: str) -> None:
        self.binary_path = binary_path
        super().__init__(message)


class CliExitError(AgentShimError):
    """The provider CLI exited nonzero."""

    def __init__(
        self,
        argv: Sequence[str],
        returncode: int,
        stdout: str = "",
        stderr: str = "",
        message: str | None = None,
    ) -> None:
        self.argv = tuple(argv)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        binary = self.argv[0] if self.argv else "cli"
        super().__init__(message or f"{binary} exited with code {returncode}: {stderr.strip()}")


class SessionResumeFailed(CliExitError):
    """A resumed turn failed because the provider conversation is gone."""

    def __init__(
        self,
        argv: Sequence[str],
        returncode: int,
        session_id: str,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        self.session_id = session_id
        super().__init__(
            argv,
            returncode,
            stdout,
            stderr,
            message=f"cannot resume session {session_id!r} (exit code {returncode}): {stderr.strip()}",
        )


class CliTimeoutError(AgentShimError):
    """The provider CLI did not finish within the turn timeout."""

    def __init__(self, argv: Sequence[str], timeout: float) -> None:
        self.argv = tuple(argv)
        self.timeout = timeout
        binary = self.argv[0] if self.argv else "cli"
        super().__init__(f"{binary} did not finish within {timeout}s")


class ProviderCapabilityError(AgentShimError):
    """The request asked for something the provider cannot do."""


class SchemaDialectError(ProviderCapabilityError):
    """The output schema uses constructs the provider's dialect rejects."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__("output schema is not expressible in this provider's dialect: " + "; ".join(self.problems))


class McpConfigError(AgentShimError):
    """An MCP config file is unreadable or is not a JSON object."""
