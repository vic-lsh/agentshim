"""Provider lookup.

A plain dict of factories, not a mutable registry: importing a provider
module must not change what ``provider_names()`` returns, and a caller can
always construct a provider class directly and pass the instance to
``CliAgent``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from .claude import ClaudeProvider
from .claude import scripted_lines as _claude_scripted

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


# To add a provider: implement providers/<name>/ (provider.py, parser.py,
# events.py, scripted.py) and add one entry to each dict below.
_FACTORIES: dict[str, Callable[[], Provider]] = {
    "claude": ClaudeProvider,
}

_SCRIPTED: dict[str, ScriptedLines] = {
    "claude": _claude_scripted,
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


__all__ = [
    "ClaudeProvider",
    "ScriptedLines",
    "get_provider",
    "get_scripted_lines",
    "provider_names",
]
