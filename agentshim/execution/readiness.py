"""Execute a provider-independent static readiness declaration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentshim.core.errors import AgentShimError, CliNotFoundError
from agentshim.core.status import AuthCheck, AuthState, ProbeOutput, ProviderStatus, parse_version
from agentshim.execution.executor import CommandRequest, NullSink

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from agentshim.core.profile import ProviderProfile
    from agentshim.core.status import ProbeSpec
    from agentshim.execution.executor import CommandExecutor


def probe_spec_on(
    profile: ProviderProfile,
    spec: ProbeSpec | None,
    executor: CommandExecutor,
    env: Mapping[str, str],
    timeout: float,
) -> ProviderStatus:
    """Run a static readiness declaration on an executor."""
    provider = profile.name
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

    version_out = None if spec is None else _run(executor, [path, *spec.version_argv], env, timeout)
    version = None if version_out is None else parse_version(version_out.text)
    if spec is None:
        check = AuthCheck(AuthState.UNKNOWN, "no readiness probe supplied")
    elif spec.auth is None:
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
        credential=check.credential,
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
