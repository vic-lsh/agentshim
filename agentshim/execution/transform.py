"""Rewrite every command before an inner executor sees it."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.errors import CliCheckError, CliTimeoutError
from .executor import CommandRequest, NullSink

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from .executor import CommandExecutor, CommandResult, CommandStreamSink


class TransformingExecutor:
    """Wrap an executor and rewrite each ``CommandRequest`` on the way in.

    This is how a caller wraps argv in an OS sandbox or prefixes it with
    ``docker exec`` without agentshim knowing about either. The health check
    goes through ``run``, so the transform applies to it too: a binary that
    only exists inside the container is still checkable.
    """

    def __init__(
        self,
        inner: CommandExecutor,
        transform: Callable[[CommandRequest], CommandRequest],
        *,
        find_binary: Callable[[str, Mapping[str, str]], str] | None = None,
    ) -> None:
        self._inner = inner
        self._transform = transform
        self._find_binary = find_binary

    def find_binary(self, name: str, env: Mapping[str, str]) -> str:
        if self._find_binary is not None:
            return self._find_binary(name, env)
        return self._inner.find_binary(name, env)

    def check_binary(self, path: str, env: Mapping[str, str], *, timeout: float) -> None:
        request = CommandRequest(argv=[path, "--help"], stdin=None, cwd=None, env=env, timeout=timeout)
        try:
            result = self.run(request, NullSink())
        except CliTimeoutError as exc:
            raise CliCheckError(path, f"{path} did not respond to '--help' within {timeout}s") from exc
        if result.returncode != 0:
            raise CliCheckError(
                path,
                f"'{path} --help' exited with code {result.returncode}. Stderr: {result.stderr.strip()}",
            )

    def run(self, request: CommandRequest, sink: CommandStreamSink) -> CommandResult:
        return self._inner.run(self._transform(request), sink)
