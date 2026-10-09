"""Provider readiness: what a probe reports, and how a provider declares its checks.

A probe answers "could a turn start?" without running one: is the binary
there, which version is it, is it logged in. It never calls a model. The
answer is a ``ProviderStatus``; each provider declares its own checks as a
``ProbeSpec``, and ``agentshim.probe`` runs them through an executor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


class AuthState(str, Enum):
    """Whether the provider CLI is authenticated, as far as its own tooling can say.

    ``UNKNOWN`` is distinct from ``FAILED``: it means the CLI offers no cheap
    way to tell (or the check itself could not run), so a caller must not
    treat it as a problem.
    """

    KNOWN_OK = "known_ok"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProviderStatus:
    """The readiness of one provider CLI on one executor.

    ``version`` is ``None`` when the binary is missing or did not report one.
    ``auth_detail`` explains ``auth`` in words fit for a log or an error
    message, including the fix when authentication failed.
    """

    provider: str
    binary_found: bool
    path: str | None
    version: str | None
    auth: AuthState
    auth_detail: str


@dataclass(frozen=True)
class ProbeOutput:
    """What a status command printed."""

    returncode: int
    stdout: str
    stderr: str

    @property
    def text(self) -> str:
        """Both streams, stdout first."""
        return self.stdout + self.stderr


@dataclass(frozen=True)
class AuthCheck:
    """A provider's reading of its status command's output."""

    state: AuthState
    detail: str


@dataclass(frozen=True)
class AuthProbe:
    """A cheap, non-model command that reports authentication, and how to read it.

    ``argv`` follows the binary. ``read`` receives the output and the
    environment the command ran with (an API key there changes what a failed
    login means) and must return ``UNKNOWN`` for output it does not recognise
    rather than guess.
    """

    argv: tuple[str, ...]
    read: Callable[[ProbeOutput, Mapping[str, str]], AuthCheck]


@dataclass(frozen=True)
class ProbeSpec:
    """A provider's readiness checks.

    ``auth`` is ``None`` for a CLI with no status mechanism; ``auth_note`` then
    says why the state is unknown.
    """

    auth: AuthProbe | None
    auth_note: str = "this CLI has no authentication status command"
    version_argv: tuple[str, ...] = ("--version",)


_VERSION = re.compile(r"\d+(?:\.\d+)+(?:-[0-9A-Za-z.]+)?")


def parse_version(text: str) -> str | None:
    """The first dotted version number in *text*, or ``None``.

    ``--version`` output varies (``2.1.295 (Claude Code)``,
    ``codex-cli 0.160.0``, ``GitHub Copilot CLI 1.0.94.``); the number is the
    only part every CLI shares.
    """
    match = _VERSION.search(text)
    return match.group(0) if match else None
