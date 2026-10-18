"""Scripted readiness checks, delegated to each provider's test adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from agentshim.providers import DEFAULT_CATALOG, ProviderCatalog

if TYPE_CHECKING:
    from agentshim.core.status import AuthState


class ProbeRun(NamedTuple):
    """What a scripted probe command prints."""

    stdout: str = ""
    stderr: str = ""
    returncode: int = 0


def probe_run(
    provider: str,
    args: list[str],
    *,
    version: str,
    auth: AuthState,
    catalog: ProviderCatalog = DEFAULT_CATALOG,
) -> ProbeRun:
    """Replay a provider's version or authentication status command."""
    fixture = catalog.test_support(provider).probe_fixture
    if fixture is None:
        msg = f"no probe fixture for provider {provider!r}"
        raise ValueError(msg)
    return ProbeRun(*fixture(args, version=version, auth=auth))
