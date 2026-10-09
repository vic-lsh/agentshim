"""Readiness probe: is a provider CLI installed, which version, logged in?

Composition layer, next to ``runtime``: it turns a provider name into the
provider's declared checks and runs them on an executor. Everything runs
through the executor and environment a turn would use, so a confinement or a
container is probed where the agent would run. No check calls a model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentshim.core.env import interactive_env
from agentshim.core.errors import AgentShimError, CliNotFoundError
from agentshim.core.status import (
    AuthCheck,
    AuthState,
    ProbeOutput,
    ProviderStatus,
    parse_version,
)
from agentshim.execution.confinement import confine
from agentshim.execution.executor import CommandRequest, NullSink
from agentshim.execution.host import HostCommandExecutor
from agentshim.providers import get_probe_spec, get_provider

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from agentshim.execution.confinement import Confinement
    from agentshim.execution.executor import CommandExecutor


def probe_provider(
    provider: str,
    *,
    executor: CommandExecutor | None = None,
    confinement: Confinement | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = 15.0,
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
    return probe_on(provider, base, dict(env) if env is not None else interactive_env(), timeout)


def probe_on(
    provider: str, executor: CommandExecutor, env: Mapping[str, str], timeout: float
) -> ProviderStatus:
    """Probe *provider* on an executor that is already confined, with the final *env*."""
    profile = get_provider(provider).profile
    spec = get_probe_spec(provider)
    try:
        path = executor.find_binary(profile.binary, env)
    except CliNotFoundError:
        detail = f"{profile.binary} was not found on PATH"
        return ProviderStatus(
            provider=provider,
            binary_found=False,
            path=None,
            version=None,
            auth=AuthState.UNKNOWN,
            auth_detail=detail,
        )

    version_out = _run(executor, [path, *spec.version_argv], env, timeout)
    version = None if version_out is None else parse_version(version_out.text)
    if spec.auth is None:
        check = AuthCheck(AuthState.UNKNOWN, spec.auth_note)
    else:
        auth_out = _run(executor, [path, *spec.auth.argv], env, timeout)
        if auth_out is None:
            check = AuthCheck(
                AuthState.UNKNOWN,
                f"`{profile.binary} {' '.join(spec.auth.argv)}` did not run or timed out",
            )
        else:
            check = spec.auth.read(auth_out, env)
    return ProviderStatus(
        provider=provider,
        binary_found=True,
        path=path,
        version=version,
        auth=check.state,
        auth_detail=check.detail,
    )


def _run(
    executor: CommandExecutor, argv: Sequence[str], env: Mapping[str, str], timeout: float
) -> ProbeOutput | None:
    """Run one status command; ``None`` when it could not run or timed out."""
    request = CommandRequest(argv=argv, stdin=None, cwd=None, env=env, timeout=timeout)
    try:
        result = executor.run(request, NullSink())
    except (AgentShimError, OSError):
        return None
    return ProbeOutput(result.returncode, result.stdout, result.stderr)
