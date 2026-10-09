"""Confinement through ``docker exec`` into an already running container."""

from __future__ import annotations

import contextlib
import os
import subprocess
from typing import TYPE_CHECKING

from agentshim.core.errors import AgentShimError

from .confinement import PathMap
from .executor import CommandRequest, NullSink

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from .executor import CommandExecutor

#: Set in every process ``wrap`` starts; ``reap`` finds processes by it.
CONFINED_MARKER = "AGENTSHIM_CONFINED"

_REAP_TIMEOUT_S = 60.0

# Kills every process whose environment carries the marker, except this shell
# and its own parents. /proc/<pid>/environ is the process's initial
# environment, so the marker survives however the agent later changes its own.
# Anything without the marker (a nested dockerd, the container's init) is left
# alone. The final ``true`` keeps "nothing matched" a success.
_REAP_SCRIPT = (
    "self=$$; "
    "for d in /proc/[0-9]*; do "
    'pid=${d#/proc/}; [ "$pid" = "$self" ] && continue; '
    f"tr '\\0' '\\n' < \"$d/environ\" 2>/dev/null | grep -qx '{CONFINED_MARKER}=1' "
    '&& kill -9 "$pid" 2>/dev/null; '
    "done; true"
)


class DockerExecConfinement:
    """Run processes with ``docker exec`` in a running container.

    Args:
        container_id: Read on every ``wrap``/``reap``, because the container
            may be replaced during a long run.
        runner: Executor used to run ``docker`` for ``reap``.
        path_map: Host directory to container directory. ``agent_path`` maps
            by longest matching prefix on whole path components.
        env: Environment for the confined process. Each key is passed as
            ``-e KEY`` (name only) and the value rides in the docker client's
            environment (``confine`` merges ``env`` into the launch
            environment), so values never appear in the host process table.
            A key with an empty name or ``=``/NUL in it is rejected.
        user: Passed as ``-u`` when set.
        docker: The docker client binary.
    """

    def __init__(  # noqa: PLR0913 - mirrors the docker flags it models
        self,
        container_id: Callable[[], str],
        *,
        runner: CommandExecutor,
        env: Mapping[str, str],
        path_map: Mapping[str, str] | None = None,
        user: str | None = None,
        docker: str = "docker",
    ) -> None:
        """Validate and freeze the configuration."""
        for key in env:
            if not key or "=" in key or "\x00" in key:
                msg = f"invalid environment variable name: {key!r}"
                raise ValueError(msg)
        self._container_id = container_id
        self._runner = runner
        self._env = dict(env)
        self._path_map = PathMap(path_map or {})
        self._user = user
        self._docker = docker

    @property
    def env(self) -> Mapping[str, str]:
        """The environment the confined process gets (names only on the command line)."""
        return dict(self._env)

    def agent_path(self, host_path: str | os.PathLike[str]) -> str:
        """Map *host_path* through the longest matching ``path_map`` prefix.

        Unmapped paths are returned normalized but otherwise unchanged.
        """
        return self._path_map.map(host_path)

    def wrap(self, argv: Sequence[str], cwd: str | None) -> list[str]:
        """Build ``docker exec -i ... <container> <argv>``; *cwd* is mapped."""
        command = [self._docker, "exec", "-i", "-e", f"{CONFINED_MARKER}=1"]
        if self._user is not None:
            command += ["-u", self._user]
        if cwd is not None:
            command += ["-w", self.agent_path(cwd)]
        for key in self._env:
            command += ["-e", key]
        return [*command, self._container_id(), *argv]

    def reap(self) -> None:
        """Kill every marked process in the container; a missing container is fine."""
        argv = [self._docker, "exec", self._container_id(), "sh", "-c", _REAP_SCRIPT]
        request = CommandRequest(
            argv=argv, stdin=None, cwd=None, env=dict(os.environ), timeout=_REAP_TIMEOUT_S
        )
        # No container, no docker client, or a daemon that is down all mean
        # there is nothing of ours left running to kill.
        with contextlib.suppress(AgentShimError, OSError, subprocess.SubprocessError):
            self._runner.run(request, NullSink())
