"""Test doubles shipped for consumers.

Build agents with ``FakeExecutor`` and assert on ``TurnResult`` and the typed
events; never on a provider's internal attributes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agentshim.core.errors import CliNotFoundError, CliTimeoutError
from agentshim.core.profile import McpMechanism
from agentshim.core.usage import TokenUsage
from agentshim.execution.executor import CommandResult
from agentshim.providers import get_provider, get_resume_failure_lines, get_scripted_lines
from agentshim.providers.claude import provider as _claude
from agentshim.providers.codex.provider import parse_mcp_servers as _parse_codex_mcp_servers
from agentshim.providers.copilot.provider import parse_mcp_servers as _parse_copilot_mcp_servers
from agentshim.providers.gemini import provider as _gemini
from agentshim.providers.opencode import provider as _opencode

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

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


def scripted_resume_failure(provider: str, *, session_id: str | None = None) -> FakeRun:
    """Build a ``FakeRun`` for a resumed turn *provider* cannot continue.

    Serve this to a session that has already ``adopt``-ed a session id: the
    turn's ``resumed`` flag plus *provider*'s own exit-classification rule
    (documented on ``docs/architecture.md``, "Resume diagnosis is
    provider-dependent") is what turns the nonzero exit into
    ``SessionResumeError`` on claude, codex, gemini and opencode. Copilot
    gives no such signal, so the same run there stays a plain
    ``CliExitError``. Served to a *fresh* (non-resumed) turn instead, every
    provider's rule falls back to a plain ``CliExitError`` too.

    Finds the scripted failure through ``providers/<name>/scripted.py``'s
    ``resume_failure_lines``, the same convention ``scripted_turn`` uses for
    ``scripted_lines``, so a new provider cannot ship without one.

    Args:
        provider: A registered provider name.
        session_id: Folded into the scripted failure message for realism;
            it does not change what a raised error reports, which is always
            read from the resumed turn's own argv.

    Returns:
        A ``FakeRun`` with a nonzero exit code.
    """
    stdout, stderr, returncode = get_resume_failure_lines(provider)(session_id=session_id)
    return FakeRun(stdout=list(stdout), stderr=list(stderr), returncode=returncode)


def installed_mcp_servers(
    provider: str, request: CommandRequest, workspace: Path
) -> dict[str, dict[str, Any]]:
    """Return the MCP servers *request* installed for one turn, keyed by name.

    Call this from inside a ``FakeExecutor`` ``run`` callback, while the turn
    that installed the servers is still running: a config-file provider's
    config file exists only for the lifetime of the turn, since the library
    restores it from the session's ``finally`` once the run returns, and a
    CLI-flags provider's servers live only in the one argv the run received.

    Each entry is a plain dict in a canonical shape, regardless of how the
    provider itself renders it: ``command``, ``args`` and ``env`` for a
    server started over stdio, ``url`` and ``transport`` for one reached over
    HTTP.

    Args:
        provider: A registered provider name.
        request: The ``CommandRequest`` the executor's ``run`` callback
            received for this turn.
        workspace: The turn's MCP workspace (its ``cwd``, unless
            ``TurnRequest.mcp_workspace`` overrode it). Unused for a
            CLI-flags provider.

    Returns:
        One canonical entry per installed server.
    """
    profile = get_provider(provider).profile
    if profile.mcp is McpMechanism.NONE:
        return {}
    if profile.mcp is McpMechanism.CLI_FLAGS:
        return _flags_mcp_servers(provider, request.argv)
    return _config_file_mcp_servers(provider, workspace)


def _flags_mcp_servers(provider: str, argv: Sequence[str]) -> dict[str, dict[str, Any]]:
    if provider == "codex":
        return _parse_codex_mcp_servers(argv)
    if provider == "copilot":
        return _parse_copilot_mcp_servers(argv)
    msg = f"no MCP-flags parser registered for provider {provider!r}"
    raise ValueError(msg)


def _config_file_mcp_servers(provider: str, workspace: Path) -> dict[str, dict[str, Any]]:
    if provider == "claude":
        target = workspace / _claude.MCP_CONFIG_FILENAME
        server_key = _claude.MCP_SERVER_KEY
        canonical = _claude_mcp_entry
    elif provider == "gemini":
        target = workspace / _gemini.MCP_CONFIG_FILENAME
        server_key = _gemini.MCP_SERVER_KEY
        canonical = _gemini_mcp_entry
    elif provider == "opencode":
        target = workspace / _opencode.MCP_CONFIG_FILENAME
        server_key = _opencode.MCP_SERVER_KEY
        canonical = _opencode_mcp_entry
    else:
        msg = f"no MCP config-file reader registered for provider {provider!r}"
        raise ValueError(msg)
    if not target.exists():
        return {}
    config = json.loads(target.read_text())
    servers = config.get(server_key, {})
    return {name: canonical(raw) for name, raw in servers.items()}


def _claude_mcp_entry(raw: Mapping[str, Any]) -> dict[str, Any]:
    if "url" in raw:
        return {"url": raw["url"], "transport": raw.get("type", "http")}
    return {
        "command": raw.get("command", ""),
        "args": list(raw.get("args", ())),
        "env": dict(raw.get("env", {})),
    }


def _gemini_mcp_entry(raw: Mapping[str, Any]) -> dict[str, Any]:
    # gemini picks the transport by which key holds the address: ``httpUrl``
    # is streamable HTTP, plain ``url`` is SSE.
    if "httpUrl" in raw:
        return {"url": raw["httpUrl"], "transport": "http"}
    if "url" in raw:
        return {"url": raw["url"], "transport": "sse"}
    return {
        "command": raw.get("command", ""),
        "args": list(raw.get("args", ())),
        "env": dict(raw.get("env", {})),
    }


def _opencode_mcp_entry(raw: Mapping[str, Any]) -> dict[str, Any]:
    if raw.get("type") == "remote":
        # opencode has one remote server type and negotiates the transport
        # itself, so it is not recoverable from the config; "http" matches
        # ``HttpMcpServer``'s own default.
        return {"url": raw["url"], "transport": "http"}
    command = list(raw.get("command", ()))
    return {
        "command": command[0] if command else "",
        "args": command[1:],
        "env": dict(raw.get("environment", {})),
    }


__all__ = [
    "FakeCommandHandle",
    "FakeExecutor",
    "FakeRun",
    "RecordingEventHandler",
    "TokenUsage",
    "installed_mcp_servers",
    "scripted_resume_failure",
    "scripted_turn",
]
