"""Readiness probe: is a provider CLI installed, which version, logged in?

Composition layer, next to ``runtime``: it turns a provider name into the
provider's declared checks and runs them on an executor. Everything runs
through the executor and environment a turn would use, so a confinement or a
container is probed where the agent would run. No check calls a model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentshim.core.env import interactive_env
from agentshim.execution.confinement import confine
from agentshim.execution.host import HostCommandExecutor
from agentshim.providers import DEFAULT_CATALOG, ProviderCatalog

if TYPE_CHECKING:
    from collections.abc import Mapping

    from agentshim.core.status import ProviderStatus
    from agentshim.execution.confinement import Confinement
    from agentshim.execution.executor import CommandExecutor


def probe_provider(  # noqa: PLR0913
    provider: str,
    *,
    executor: CommandExecutor | None = None,
    confinement: Confinement | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = 15.0,
    catalog: ProviderCatalog = DEFAULT_CATALOG,
) -> ProviderStatus:
    """Report the readiness of *provider* without running a turn.

    Arguments mean what they mean for ``Agent``: *executor* defaults to the
    local host, *confinement* wraps it and supplies the environment (so *env*
    may not also be given), and *env* otherwise defaults to the login shell's
    environment. *timeout* bounds each of the two commands (``--version`` and
    the provider's status command).

    A missing binary is a result (``binary_found=False``), not an error, so a
    startup check can report every provider in one pass. A command that cannot
    run or times out leaves that field unknown; it never raises for a CLI
    problem. An unknown *provider* raises ``ValueError``.
    """
    base = executor if executor is not None else HostCommandExecutor()
    if confinement is not None:
        if env is not None:
            msg = "env conflicts with confinement: the confinement supplies the agent environment"
            raise ValueError(msg)
        base = confine(base, confinement)
        env = dict(confinement.env)
    return probe_on(
        provider,
        base,
        dict(env) if env is not None else interactive_env(),
        timeout,
        catalog=catalog,
    )


def probe_on(
    provider: str,
    executor: CommandExecutor,
    env: Mapping[str, str],
    timeout: float,
    *,
    catalog: ProviderCatalog = DEFAULT_CATALOG,
) -> ProviderStatus:
    """Probe *provider* on an executor that is already confined, with the final *env*."""
    return catalog.plugin(provider).probe(executor, env, timeout)
