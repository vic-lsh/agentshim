"""Confinement enforced outside the agent.

A ``Confinement`` bounds what a started process can touch (a container, a VM,
an OS sandbox) without the agent's cooperation. ``confine`` applies one to any
``CommandExecutor``.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol

from .transform import TransformingExecutor

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .executor import CommandExecutor, CommandRequest


class Confinement(Protocol):
    """A boundary around the processes agentshim starts."""

    @property
    def env(self) -> Mapping[str, str]:
        """Environment the confined process needs, which ``confine`` supplies.

        The values travel in the *launcher's* environment (the host process
        that runs the confinement's client), not in its argv, so secrets do not
        appear in the host process table. ``wrap`` references them by name.
        """
        ...

    def wrap(self, argv: Sequence[str], cwd: str | None) -> list[str]:
        """Return the host argv that runs *argv* inside the confinement.

        *cwd* is the directory as the agent sees it (see ``agent_path``). The
        original *argv* is kept verbatim as the tail of the result.
        """
        ...

    def agent_path(self, host_path: str | os.PathLike[str]) -> str:
        """Return the path inside the confinement for *host_path*."""
        ...

    def reap(self) -> None:
        """Kill every process this kind of confinement started for agentshim.

        Includes processes left by an earlier host process. Idempotent; having
        nothing to kill is not an error.
        """
        ...


def confine(executor: CommandExecutor, confinement: Confinement) -> CommandExecutor:
    """Return *executor* with every command run inside *confinement*.

    ``run``, ``spawn`` and the binary health check all go through
    ``confinement.wrap``. The binary lives inside the confinement, not on the
    host ``PATH``, so ``find_binary`` trusts the bare name. The host working
    directory is dropped (the launcher runs wherever the host executor runs);
    the agent's directory is passed to ``wrap`` instead. ``confinement.env`` is
    merged over the request environment, for the launcher to forward by name.
    """

    def transform(request: CommandRequest) -> CommandRequest:
        return replace(
            request,
            argv=confinement.wrap(request.argv, request.cwd),
            cwd=None,
            env={**request.env, **confinement.env},
        )

    return TransformingExecutor(executor, transform, find_binary=lambda name, _env: name)


def normalize_path(path: str | os.PathLike[str]) -> str:
    """Collapse separators and drop ``.`` segments, without touching the filesystem.

    ``..`` is kept: resolving it lexically would be wrong across symlinks, and
    the filesystem is deliberately never consulted.
    """
    text = os.fspath(path)
    joined = "/".join(part for part in text.split("/") if part not in ("", "."))
    if text.startswith("/"):
        return "/" + joined
    return joined or "."


class PathMap:
    """Host directory to confined directory, applied by longest prefix.

    Prefixes match whole path components (``/a/bc`` is not under ``/a/b``).
    A path under no entry maps to itself, normalized.
    """

    def __init__(self, mapping: Mapping[str, str]) -> None:
        """Normalize the entries and order them longest host prefix first."""
        self._entries = sorted(
            ((normalize_path(host), normalize_path(agent)) for host, agent in mapping.items()),
            key=lambda entry: len(entry[0]),
            reverse=True,
        )

    def map(self, host_path: str | os.PathLike[str]) -> str:
        """Return *host_path* as the confined process sees it."""
        path = normalize_path(host_path)
        for host, agent in self._entries:
            if path == host:
                return agent
            prefix = host if host.endswith("/") else host + "/"
            if path.startswith(prefix):
                return normalize_path(agent + "/" + path[len(prefix) :])
        return path
