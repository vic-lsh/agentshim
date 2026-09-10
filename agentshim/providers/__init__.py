"""Provider lookup.

A plain dict of factories, not a mutable registry: importing a provider
module must not change what ``provider_names()`` returns, and a caller can
always construct a provider class directly and pass the instance to
``CliAgent``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from .claude import ClaudeProvider
from .claude import resume_failure_lines as _claude_resume_failure
from .claude import scripted_lines as _claude_scripted
from .codex import CodexProvider
from .codex import resume_failure_lines as _codex_resume_failure
from .codex import scripted_lines as _codex_scripted
from .copilot import CopilotProvider
from .copilot import resume_failure_lines as _copilot_resume_failure
from .copilot import scripted_lines as _copilot_scripted
from .gemini import GeminiProvider
from .gemini import resume_failure_lines as _gemini_resume_failure
from .gemini import scripted_lines as _gemini_scripted
from .opencode import OpencodeProvider
from .opencode import resume_failure_lines as _opencode_resume_failure
from .opencode import scripted_lines as _opencode_scripted

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from agentshim.core.provider import Provider
    from agentshim.core.usage import TokenUsage


class ScriptedLines(Protocol):
    """Builds one turn's stdout in a provider's own stream format."""

    def __call__(
        self,
        *,
        text: str = "",
        session_id: str | None = None,
        usage: TokenUsage | None = None,
        tool_calls: Sequence[tuple[str, Mapping[str, Any], str]] = (),
        structured_output: object | None = None,
    ) -> list[str]:
        """Return the stdout lines of one scripted turn.

        ``tool_calls`` entries are ``(tool, args, output)``;
        ``structured_output`` is any JSON-serializable value the turn should
        report as its structured payload.
        """
        ...


class ResumeFailureLines(Protocol):
    """Builds the stdout, stderr and exit code of a resumed turn gone bad."""

    def __call__(
        self, *, session_id: str | None = None
    ) -> tuple[Sequence[str], Sequence[str], int]:
        """Return ``(stdout, stderr, returncode)`` for a ``FakeRun``.

        ``session_id`` is folded into the scripted message for realism only;
        what a resumed turn's ``classify_exit`` actually reports comes from
        the turn's own argv.
        """
        ...


# To add a provider: implement providers/<name>/ (provider.py, parser.py,
# events.py, scripted.py) and add one entry to each dict below.
_FACTORIES: dict[str, Callable[[], Provider]] = {
    "claude": ClaudeProvider,
    "codex": CodexProvider,
    "copilot": CopilotProvider,
    "gemini": GeminiProvider,
    "opencode": OpencodeProvider,
}

_SCRIPTED: dict[str, ScriptedLines] = {
    "claude": _claude_scripted,
    "codex": _codex_scripted,
    "copilot": _copilot_scripted,
    "gemini": _gemini_scripted,
    "opencode": _opencode_scripted,
}

_RESUME_FAILURES: dict[str, ResumeFailureLines] = {
    "claude": _claude_resume_failure,
    "codex": _codex_resume_failure,
    "copilot": _copilot_resume_failure,
    "gemini": _gemini_resume_failure,
    "opencode": _opencode_resume_failure,
}


def provider_names() -> list[str]:
    """Return the supported provider names, sorted."""
    return sorted(_FACTORIES)


def get_provider(name: str) -> Provider:
    """Construct the provider registered under *name* with its defaults."""
    factory = _FACTORIES.get(name)
    if factory is None:
        msg = f"unknown provider {name!r}; available: {provider_names()}"
        raise ValueError(msg)
    return factory()


def get_scripted_lines(name: str) -> ScriptedLines:
    """Return the test-double stream builder for *name*."""
    scripted = _SCRIPTED.get(name)
    if scripted is None:
        msg = f"no scripted stream for provider {name!r}; available: {sorted(_SCRIPTED)}"
        raise ValueError(msg)
    return scripted


def get_resume_failure_lines(name: str) -> ResumeFailureLines:
    """Return the test-double resume-failure builder for *name*."""
    resume_failure = _RESUME_FAILURES.get(name)
    if resume_failure is None:
        msg = (
            f"no resume-failure stream for provider {name!r}; available: {sorted(_RESUME_FAILURES)}"
        )
        raise ValueError(msg)
    return resume_failure


__all__ = [
    "ClaudeProvider",
    "CodexProvider",
    "CopilotProvider",
    "GeminiProvider",
    "OpencodeProvider",
    "ResumeFailureLines",
    "ScriptedLines",
    "get_provider",
    "get_resume_failure_lines",
    "get_scripted_lines",
    "provider_names",
]
