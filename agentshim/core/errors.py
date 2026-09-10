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
        """Record the binary name that the PATH lookup failed to resolve.

        The name is kept as an attribute so a caller can build an install hint
        without re-parsing the message. Passing ``message`` replaces the
        generated text.
        """
        self.binary = binary
        super().__init__(message or f"{binary} binary not found in PATH")


class CliCheckError(AgentShimError):
    """The provider binary was found but its health check failed."""

    def __init__(self, binary_path: str, message: str) -> None:
        """Record the binary that was found but failed its health check.

        ``binary_path`` is the resolved path rather than the name looked up on
        PATH, because the binary that answered may not be the one the caller
        expected to find.
        """
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
        """Capture the whole outcome of a nonzero CLI exit.

        Both streams are retained because a provider may report the real cause
        on either one, and ``argv`` is copied so the error stays accurate after
        the caller mutates its command list. Passing ``message`` replaces the
        generated summary.
        """
        self.argv = tuple(argv)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        binary = self.argv[0] if self.argv else "cli"
        super().__init__(message or f"{binary} exited with code {returncode}: {stderr.strip()}")


class SessionResumeError(CliExitError):
    """A resumed turn failed because the provider conversation is gone."""

    def __init__(
        self,
        argv: Sequence[str],
        returncode: int,
        session_id: str,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        """Attach the conversation id that the provider refused to resume.

        Raised instead of a plain ``CliExitError`` so a caller can tell a dead
        conversation apart from a failed turn and retry from scratch.
        """
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
        """Record the command and the budget it overran.

        ``timeout`` is the budget in seconds that was allowed, not the elapsed
        time, which is unknown once the process has been killed.
        """
        self.argv = tuple(argv)
        self.timeout = timeout
        binary = self.argv[0] if self.argv else "cli"
        super().__init__(f"{binary} did not finish within {timeout}s")


class ProviderCapabilityError(AgentShimError):
    """The request asked for something the provider cannot do."""


class SchemaDialectError(ProviderCapabilityError):
    """The output schema uses constructs the provider's dialect rejects."""

    def __init__(self, problems: Sequence[str]) -> None:
        """Collect every dialect problem found in one schema.

        All problems are reported together so a caller repairing a schema does
        not have to re-run the check once per offending keyword.
        """
        self.problems = list(problems)
        super().__init__(
            "output schema is not expressible in this provider's dialect: "
            + "; ".join(self.problems)
        )


class McpConfigError(AgentShimError):
    """An MCP config file is unreadable or is not a JSON object."""
