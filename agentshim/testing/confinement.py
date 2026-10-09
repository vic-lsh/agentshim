"""A recording ``Confinement`` for tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentshim.execution.confinement import PathMap

if TYPE_CHECKING:
    import os
    from collections.abc import Mapping, Sequence


class FakeConfinement:
    """Wraps argv behind a fixed prefix and records what it was asked.

    ``wraps`` holds each ``(argv, cwd)`` seen and ``reaps`` counts ``reap``
    calls. Paths map by ``path_map`` (longest prefix) and are otherwise
    identity.
    """

    def __init__(
        self,
        *,
        path_map: Mapping[str, str] | None = None,
        env: Mapping[str, str] | None = None,
        prefix: Sequence[str] = ("fake-confine",),
    ) -> None:
        """Configure the path mapping, environment and argv prefix."""
        self._path_map = PathMap(path_map or {})
        self._env = dict(env or {})
        self._prefix = list(prefix)
        self.wraps: list[tuple[tuple[str, ...], str | None]] = []
        self.reaps = 0

    @property
    def env(self) -> Mapping[str, str]:
        """The configured environment."""
        return dict(self._env)

    def wrap(self, argv: Sequence[str], cwd: str | None) -> list[str]:
        """Record the call and return the prefix followed by *argv*."""
        self.wraps.append((tuple(argv), cwd))
        return [*self._prefix, *argv]

    def agent_path(self, host_path: str | os.PathLike[str]) -> str:
        """Map *host_path* through ``path_map``."""
        return self._path_map.map(host_path)

    def reap(self) -> None:
        """Count the call."""
        self.reaps += 1
