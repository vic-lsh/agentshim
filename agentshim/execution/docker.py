"""Confinement through ``docker exec`` into an already running container."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from agentshim.core.errors import ReapError

from .confinement import PathMap
from .executor import CommandRequest, NullSink

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from .executor import CommandExecutor

#: Set in every process ``wrap`` starts; ``reap`` finds processes by it.
CONFINED_MARKER = "AGENTSHIM_CONFINED"

_REAP_TIMEOUT_S = 60.0

# Kills every process whose environment carries the marker. The reaper's own
# shell and its helpers are not marked (``reap`` does not pass the marker to
# ``docker exec``), so they are never candidates; ``$$`` is skipped as well.
# /proc/<pid>/environ is the process's initial environment, so the marker
# survives however the agent later changes its own. Anything without the marker
# (a nested dockerd, the container's init) is left alone. "Nothing matched" is
# success, but a container without ``tr`` or ``grep`` cannot reap at all and
# must say so (exit 127) rather than report a clean sweep.
_REAP_SCRIPT = (
    "command -v tr >/dev/null && command -v grep >/dev/null "
    "|| { echo 'reap needs tr and grep in the container' >&2; exit 127; }; "
    "self=$$; "
    "for d in /proc/[0-9]*; do "
    'pid=${d#/proc/}; [ "$pid" = "$self" ] && continue; '
    f"tr '\\0' '\\n' < \"$d/environ\" 2>/dev/null | grep -qx '{CONFINED_MARKER}=1' "
    '&& kill -9 "$pid" 2>/dev/null; '
    "done; true"
)

# What the docker (or podman) client says when the container is gone or
# stopped: either way none of its processes can still be running.
_NO_CONTAINER = ("no such container", "is not running", "no container with name or id")


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
            A key with an empty name or ``=``/NUL in it, or the reserved
            marker name, is rejected.
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
            if not key or "=" in key or "\x00" in key or key == CONFINED_MARKER:
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
        """Kill every marked process in the container.

        A container that is gone or stopped is nothing to reap. Any other
        failure (no docker client, a daemon error, a timeout, a container
        missing ``sh``/``tr``/``grep``) raises ``ReapError``: the agents may
        still be running.
        """
        argv = [self._docker, "exec", self._container_id(), "sh", "-c", _REAP_SCRIPT]
        request = CommandRequest(
            argv=argv, stdin=None, cwd=None, env=dict(os.environ), timeout=_REAP_TIMEOUT_S
        )
        try:
            result = self._runner.run(request, NullSink())
        except OSError as exc:
            msg = f"could not run {self._docker!r} to reap: {exc}"
            raise ReapError(msg) from exc
        if result.returncode == 0:
            return
        stderr = result.stderr.lower()
        if any(text in stderr for text in _NO_CONTAINER):
            return
        msg = f"reap exited {result.returncode}: {result.stderr.strip()}"
        raise ReapError(msg)
