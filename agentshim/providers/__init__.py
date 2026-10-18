"""Built-in provider catalog and compatibility lookup functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentshim.core.plugin import (
    FailureLines,
    ProviderPlugin,
    ProviderTestSupport,
    ResumeFailureLines,
    ScriptedLines,
    StreamTransportFactory,
)

from .catalog import ProviderCatalog
from .claude import ClaudeProvider, ClaudeStreamTransport
from .claude.plugin import PLUGIN as CLAUDE_PLUGIN
from .codex import CodexProvider
from .codex.app_server import CodexAppServerTransport
from .codex.plugin import PLUGIN as CODEX_PLUGIN
from .copilot import CopilotProvider
from .copilot.plugin import PLUGIN as COPILOT_PLUGIN
from .gemini import GeminiProvider
from .gemini.plugin import PLUGIN as GEMINI_PLUGIN
from .opencode import OpencodeProvider
from .opencode.plugin import PLUGIN as OPENCODE_PLUGIN

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from agentshim.core.clock import Clock
    from agentshim.core.conversation import Transport
    from agentshim.core.ids import IdAllocator
    from agentshim.core.provider import Provider
    from agentshim.core.status import ProbeSpec
    from agentshim.execution.executor import CommandExecutor


DEFAULT_CATALOG = ProviderCatalog(
    (CLAUDE_PLUGIN, CODEX_PLUGIN, COPILOT_PLUGIN, GEMINI_PLUGIN, OPENCODE_PLUGIN)
)


def provider_names(*, catalog: ProviderCatalog = DEFAULT_CATALOG) -> list[str]:
    """List names in *catalog*, sorted."""
    return catalog.names()


def stream_provider_names(*, catalog: ProviderCatalog = DEFAULT_CATALOG) -> list[str]:
    """List providers in *catalog* with long-lived transports."""
    return catalog.stream_names()


def get_provider(name: str, *, catalog: ProviderCatalog = DEFAULT_CATALOG) -> Provider:
    """Construct a provider by name from *catalog*."""
    return catalog.provider(name)


def get_scripted_lines(name: str, *, catalog: ProviderCatalog = DEFAULT_CATALOG) -> ScriptedLines:
    """Return a provider's scripted stream builder."""
    if name not in catalog.names():
        msg = f"no scripted stream for provider {name!r}; available: {catalog.names()}"
        raise ValueError(msg)
    return catalog.test_support(name).scripted_lines


def get_resume_failure_lines(
    name: str, *, catalog: ProviderCatalog = DEFAULT_CATALOG
) -> ResumeFailureLines:
    """Return a provider's resume-failure stream builder."""
    if name not in catalog.names():
        msg = f"no resume-failure stream for provider {name!r}; available: {catalog.names()}"
        raise ValueError(msg)
    return catalog.test_support(name).resume_failure_lines


def get_failure_lines(name: str, *, catalog: ProviderCatalog = DEFAULT_CATALOG) -> FailureLines:
    """Return a provider's classified-failure stream builder."""
    failure = catalog.test_support(name).failure_lines
    if failure is None:
        msg = f"no classified-failure stream for provider {name!r}"
        raise ValueError(msg)
    return failure


def get_probe_spec(name: str, *, catalog: ProviderCatalog = DEFAULT_CATALOG) -> ProbeSpec:
    """Return a provider's readiness checks."""
    return catalog.probe_spec(name)


def get_stream_transport(  # noqa: PLR0913
    name: str,
    *,
    executor: CommandExecutor,
    env: Mapping[str, str] | None,
    clock: Clock,
    ids: IdAllocator,
    check_timeout: float,
    log: Callable[[str], None] | None,
    catalog: ProviderCatalog = DEFAULT_CATALOG,
) -> Transport:
    """Construct a long-lived transport by name."""
    return catalog.stream_transport(
        name,
        executor=executor,
        env=env,
        clock=clock,
        ids=ids,
        check_timeout=check_timeout,
        log=log,
    )


__all__ = [
    "DEFAULT_CATALOG",
    "ClaudeProvider",
    "ClaudeStreamTransport",
    "CodexAppServerTransport",
    "CodexProvider",
    "CopilotProvider",
    "FailureLines",
    "GeminiProvider",
    "OpencodeProvider",
    "ProviderCatalog",
    "ProviderPlugin",
    "ProviderTestSupport",
    "ResumeFailureLines",
    "ScriptedLines",
    "StreamTransportFactory",
    "get_failure_lines",
    "get_probe_spec",
    "get_provider",
    "get_resume_failure_lines",
    "get_scripted_lines",
    "get_stream_transport",
    "provider_names",
    "stream_provider_names",
]
