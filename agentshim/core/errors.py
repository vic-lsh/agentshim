"""Exception hierarchy. Nothing outside this hierarchy escapes ``turn()``."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .provider import ParsedTurn


class AgentShimError(Exception):
    """Base class for every error agentshim raises."""


class FailureKind(str, Enum):
    """Why a provider turn failed, as far as the provider let agentshim tell.

    The provider package decides the kind from what its CLI reported (a
    structured error field, an HTTP status, or its own error text), so a caller
    can choose a policy without knowing how any CLI phrases its failures.
    """

    #: An overload, rate limit, or server error that waiting may outlast.
    TRANSIENT = "transient"
    #: A usage quota, spend limit, or billing problem: waiting minutes will not help.
    USAGE_LIMIT = "usage_limit"
    #: Missing, expired, or rejected credentials.
    AUTH = "auth"
    #: The provider gave up producing output that matches the requested output
    #: schema. ``detail`` carries the last validation errors it reported, and
    #: the conversation survives: a correction can be sent as the next turn.
    SCHEMA = "schema"
    #: Anything else, including every failure a provider cannot classify.
    OTHER = "other"


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

    def __init__(  # noqa: PLR0913  # one field per part of the outcome; a bundle would hide them
        self,
        argv: Sequence[str],
        returncode: int,
        stdout: str = "",
        stderr: str = "",
        message: str | None = None,
        *,
        kind: FailureKind = FailureKind.OTHER,
        detail: str = "",
    ) -> None:
        """Capture the whole outcome of a nonzero CLI exit.

        Both streams are retained because a provider may report the real cause
        on either one, and ``argv`` is copied so the error stays accurate after
        the caller mutates its command list. ``kind`` is the provider's
        classification of the failure and ``detail`` the error text its stream
        reported; the generated summary carries ``detail`` because a CLI that
        reports its failure in the stream leaves stderr empty. Passing
        ``message`` replaces the generated summary.
        """
        self.argv = tuple(argv)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.kind = kind
        self.detail = detail
        binary = self.argv[0] if self.argv else "cli"
        super().__init__(
            message or f"{binary} exited with code {returncode}: {_describe(detail, stderr)}"
        )


class SessionResumeError(CliExitError):
    """A resumed turn failed because the provider conversation is gone."""

    def __init__(  # noqa: PLR0913  # the parent's fields plus the session id
        self,
        argv: Sequence[str],
        returncode: int,
        session_id: str,
        stdout: str = "",
        stderr: str = "",
        *,
        detail: str = "",
    ) -> None:
        """Attach the conversation id that the provider refused to resume.

        Raised instead of a plain ``CliExitError`` so a caller can tell a dead
        conversation apart from a failed turn and retry from scratch. Its kind
        is always ``OTHER``: a provider that classified the failure as anything
        else has shown that the conversation is not what failed.
        """
        self.session_id = session_id
        super().__init__(
            argv,
            returncode,
            stdout,
            stderr,
            message=(
                f"cannot resume session {session_id!r} (exit code {returncode}): "
                f"{_describe(detail, stderr)}"
            ),
            detail=detail,
        )


class CliTimeoutError(AgentShimError):
    """The provider CLI did not finish within the turn timeout."""

    def __init__(self, argv: Sequence[str], timeout: float) -> None:
        """Record the command and the budget it overran.

        ``timeout`` is the budget in seconds that was allowed, not the elapsed
        time, which is unknown once the process has been killed. ``partial``
        starts empty and is filled in by the session with whatever the stream
        parser read before the budget ran out; it is the only way back to the
        session id of a turn that named its conversation and then timed out.
        """
        self.argv = tuple(argv)
        self.timeout = timeout
        self.partial: ParsedTurn | None = None
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


class ProcessClosedError(AgentShimError):
    """A long-lived process can no longer take input: stdin is closed or it exited."""


class McpConfigError(AgentShimError):
    """An MCP config file is unreadable, unwritable, or is not a JSON object."""


def _describe(detail: str, stderr: str) -> str:
    """Join the stream's error text and stderr, skipping whichever is empty."""
    return "; ".join(part for part in (detail.strip(), stderr.strip()) if part)
