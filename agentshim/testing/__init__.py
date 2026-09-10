"""Test doubles shipped for consumers.

Build agents with ``FakeExecutor`` and assert on ``TurnResult`` and the typed
events; never on a provider's internal attributes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agentshim.core.errors import CliNotFoundError, CliTimeoutError
from agentshim.core.usage import TokenUsage
from agentshim.execution.executor import CommandResult
from agentshim.providers import get_scripted_lines

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from agentshim.core.events import AgentEvent
    from agentshim.execution.executor import CommandRequest, CommandStreamSink


@dataclass
class FakeRun:
    """One scripted process run."""

    stdout: Sequence[str] = ()
    stderr: Sequence[str] = ()
    returncode: int = 0
    timeout: bool = False


class FakeCommandHandle:
    """Handle that records the stop calls made against it."""

    def __init__(self) -> None:
        """Start with nothing recorded."""
        self.terminated = False
        self.killed = False

    def terminate(self) -> None:
        """Record a terminate call."""
        self.terminated = True

    def kill(self) -> None:
        """Record a kill call."""
        self.killed = True


class FakeExecutor:
    """``CommandExecutor`` that replays scripted runs instead of spawning.

    ``runs`` is either a list consumed in order (the last one repeats once
    exhausted) or a callable that picks a run per request.
    """

    def __init__(
        self,
        runs: FakeRun | Sequence[FakeRun] | Callable[[CommandRequest], FakeRun],
        *,
        binaries: Mapping[str, str] | None = None,
    ) -> None:
        """Take one run, a sequence of runs, or a callable that picks per request.

        *binaries* pins what ``find_binary`` answers; without it every name
        resolves to a plausible path.
        """
        if isinstance(runs, FakeRun):
            runs = [runs]
        self._runs = runs
        self._index = 0
        self._binaries = dict(binaries) if binaries is not None else {}
        self.requests: list[CommandRequest] = []
        self.handles: list[FakeCommandHandle] = []
        self.checked: list[str] = []

    def find_binary(self, name: str, env: Mapping[str, str]) -> str:
        """Return the pinned path for *name*, or a plausible default one."""
        del env  # part of the executor protocol; the fake ignores it
        if self._binaries:
            path = self._binaries.get(name)
            if path is None:
                raise CliNotFoundError(name)
            return path
        return f"/usr/local/bin/{name}"

    def check_binary(self, path: str, env: Mapping[str, str], *, timeout: float) -> None:
        """Record the check; the fake never inspects the binary."""
        del env, timeout  # part of the executor protocol; the fake ignores them
        self.checked.append(path)

    def run(self, request: CommandRequest, sink: CommandStreamSink) -> CommandResult:
        """Replay the next scripted run, driving *sink* like a real one would."""
        self.requests.append(request)
        handle = FakeCommandHandle()
        self.handles.append(handle)
        sink.started(handle)
        run = self._next(request)
        if run.timeout:
            handle.kill()
            timeout = request.timeout if request.timeout is not None else 0.0
            raise CliTimeoutError(request.argv, timeout)
        for line in run.stdout:
            sink.stdout(line)
        for line in run.stderr:
            sink.stderr(line)
        return CommandResult(
            returncode=run.returncode,
            stdout="".join(run.stdout),
            stderr="".join(run.stderr),
        )

    def _next(self, request: CommandRequest) -> FakeRun:
        runs = self._runs
        if callable(runs):
            return runs(request)
        if not runs:
            return FakeRun()
        index = min(self._index, len(runs) - 1)
        self._index += 1
        return runs[index]


def _no_events() -> list[AgentEvent]:
    return []


@dataclass
class RecordingEventHandler:
    """Collects every event a turn emitted, in order."""

    events: list[AgentEvent] = field(default_factory=_no_events)

    def on_event(self, event: AgentEvent) -> None:
        """Append the event to ``events``."""
        self.events.append(event)

    def of_type(self, kind: type) -> list[AgentEvent]:
        """Return the recorded events that are instances of *kind*."""
        return [event for event in self.events if isinstance(event, kind)]


# Every parameter is an independent, separately documented knob of the test
# double; folding them into a config object would only lengthen call sites.
def scripted_turn(  # noqa: PLR0913
    provider: str,
    *,
    text: str = "",
    session_id: str | None = None,
    usage: TokenUsage | None = None,
    tool_calls: Sequence[tuple[str, Mapping[str, Any], str]] = (),
    structured_output: object | None = None,
    returncode: int = 0,
) -> FakeRun:
    """Build a ``FakeRun`` whose stdout is *provider*'s real stream format."""
    lines = get_scripted_lines(provider)(
        text=text,
        session_id=session_id,
        usage=usage,
        tool_calls=tool_calls,
        structured_output=structured_output,
    )
    return FakeRun(stdout=lines, returncode=returncode)


__all__ = [
    "FakeCommandHandle",
    "FakeExecutor",
    "FakeRun",
    "RecordingEventHandler",
    "TokenUsage",
    "scripted_turn",
]
