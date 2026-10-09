"""A reactive Codex ``app-server`` for tests.

``CodexScript`` plays the server's side of ``CodexAppServerTransport``: queue the
turns it should run, give ``FakeExecutor`` its ``peer`` factory, and the
transport talks to it over a ``FakeProcess`` exactly as it would to the real
CLI, with no process, no clock and no waiting::

    script = CodexScript()
    script.turn(RunCommand("ls", output="a.txt"), Say("one file"))
    executor = FakeExecutor([], peers=script.peer)
    transport = CodexAppServerTransport(executor=executor, env={"PATH": "/bin"})

The server is reactive: it answers each request as it arrives, runs a turn's
steps up to the first one that needs the client (an approval request waits for
the client's reply; ``Hang`` waits for an interrupt) and carries on when the
client answers. It keeps a rollout store shared by every process it plays, so a
thread started by one process can be resumed by the next, and it records every
message the client sent so a test can assert on requests and replies.

The server holds the client to the protocol: a request before the handshake, an
unknown method, or a reply to a request it never made is recorded in
``script.violations``.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, cast

from agentshim.execution.process import ProcessExited, StderrLine, StdoutLine
from agentshim.providers.codex.app_server import (
    ClientNotification,
    ClientRequest,
    CodexProtocolError,
    ErrorResponse,
    Response,
    parse_client_message,
)
from agentshim.providers.codex.app_server.protocol import (
    ThreadResumeParams,
    ThreadStartParams,
    TurnInterruptParams,
    TurnStartParams,
    UserInputText,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import Any

    from agentshim.execution.process import SpawnRequest
    from agentshim.providers.codex.app_server.protocol import ClientMessage
    from agentshim.testing.process import PeerOutput

_CONTEXT_WINDOW = 258_400
_METHOD_NOT_FOUND = -32601
_INVALID_REQUEST = -32600
_PREVIOUS_TURN = "turn-before-resume"


# -- the steps of a scripted turn


@dataclass(frozen=True)
class Say:
    """The agent says something. ``final_answer`` is the turn's answer, ``commentary`` narration."""

    text: str
    phase: str = "final_answer"


@dataclass(frozen=True)
class Think:
    """The agent reasons."""

    text: str


@dataclass(frozen=True)
class RunCommand:
    """The agent runs a shell command.

    With ``approval`` the command needs the user's approval *when the thread's
    approval policy asks* (anything but ``never``): the server sends an approval
    request and waits for the client's reply. A declined or cancelled command
    does not run.
    """

    command: str
    output: str = ""
    exit_code: int = 0
    approval: bool = False


@dataclass(frozen=True)
class ChangeFile:
    """The agent edits a file; like ``RunCommand``, it may need approval."""

    path: str
    approval: bool = False


@dataclass(frozen=True)
class CallMcp:
    """The agent calls an MCP tool. ``error`` makes the call fail with that message."""

    server: str
    tool: str
    arguments: Mapping[str, object] | None = None
    output: str = ""
    error: str | None = None


class AskKind(Enum):
    """The server-to-client requests ``Ask`` can make."""

    PERMISSIONS = "item/permissions/requestApproval"
    ELICITATION = "mcpServer/elicitation/request"
    USER_INPUT = "item/tool/requestUserInput"
    TOKEN_REFRESH = "account/chatgptAuthTokens/refresh"  # noqa: S105 - a method name
    LEGACY_EXEC = "execCommandApproval"
    UNKNOWN = "item/madeUp/request"


@dataclass(frozen=True)
class Ask:
    """The server asks the client something and waits for the reply."""

    kind: AskKind


@dataclass(frozen=True)
class Spend:
    """The model request used tokens. The thread's running total grows by this much."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_output_tokens: int = 0


@dataclass(frozen=True)
class ReportRateLimits:
    """The server reports the account's rate limits: ``account/rateLimits/updated``.

    ``primary`` and ``secondary`` are ``(used_percent, window_minutes,
    resets_at)`` or ``None`` for a window the server does not report.
    """

    primary: tuple[int, int | None, int | None] | None = (8, 10080, 1791954019)
    secondary: tuple[int, int | None, int | None] | None = None
    limit_id: str | None = "codex"
    reached: str | None = None


@dataclass(frozen=True)
class Retrying:
    """The stream broke and Codex is reconnecting: a non-terminal ``error``."""

    message: str = "stream disconnected - retrying"


@dataclass(frozen=True)
class Fail:
    """The turn fails. ``info`` is the ``codexErrorInfo``: a name or a one-key object."""

    message: str = "the model request failed"
    info: str | Mapping[str, object] = "other"


@dataclass(frozen=True)
class Hang:
    """The turn runs until the client interrupts it.

    With ``ignores_interrupt`` the server acknowledges an interrupt but never
    reports the turn over, as a wedged one would.
    """

    ignores_interrupt: bool = False


@dataclass(frozen=True)
class Crash:
    """The process dies, with a line on stderr."""

    returncode: int = 1
    stderr: str = "codex app-server crashed"


@dataclass(frozen=True)
class Complain:
    """A line on stderr."""

    text: str


CodexStep = (
    Say
    | Think
    | RunCommand
    | ChangeFile
    | CallMcp
    | Ask
    | Spend
    | ReportRateLimits
    | Retrying
    | Fail
    | Hang
    | Crash
    | Complain
)

#: Tokens a turn spends when it says nothing about spending.
DEFAULT_SPEND = Spend(input_tokens=100, output_tokens=10, cached_input_tokens=40)


# -- the script shared by every process


def _zero_totals() -> dict[str, int]:
    return dict.fromkeys(
        (
            "totalTokens",
            "inputTokens",
            "cachedInputTokens",
            "cacheWriteInputTokens",
            "outputTokens",
            "reasoningOutputTokens",
        ),
        0,
    )


def _no_items() -> list[Mapping[str, object]]:
    return []


@dataclass
class _Thread:
    """A thread in the rollout store."""

    thread_id: str
    cwd: str
    turns: int = 0
    totals: dict[str, int] = field(default_factory=_zero_totals)
    replay_usage: bool = False


class CodexScript:
    """What a fake Codex app-server does, and what it was asked.

    Queue turns with ``turn``; a turn that is asked for when none is queued
    answers "ok". ``peer`` is the factory for ``FakeExecutor(peers=...)``: every
    process it makes shares this script's rollout store.

    ``inherited_sandbox`` and ``inherited_approval`` stand for the user's own
    ``config.toml``: they apply to a request that names nothing. With
    ``ignore_requested_permissions`` the server applies them even to a request
    that does name something, to test a client that checks what was applied.
    With ``exits_at_end_of_input=False`` a server ignores the closing of its
    stdin and has to be terminated.
    """

    def __init__(
        self,
        *,
        model: str = "fake-model",
        inherited_sandbox: str = "danger-full-access",
        inherited_approval: str = "never",
        ignore_requested_permissions: bool = False,
        exits_at_end_of_input: bool = True,
    ) -> None:
        """Start with no queued turns, no threads and nothing received."""
        self.model = model
        self.inherited_sandbox = inherited_sandbox
        self.inherited_approval = inherited_approval
        self.ignore_requested_permissions = ignore_requested_permissions
        self.exits_at_end_of_input = exits_at_end_of_input
        self._queued: deque[tuple[CodexStep, ...]] = deque()
        self._threads: dict[str, _Thread] = {}
        self._failing_mcp: dict[str, str] = {}
        self._counts: dict[str, int] = {}
        #: Every message the client sent, in order, across processes.
        self.received: list[ClientMessage] = []
        #: The client's replies to server requests as ``(method, wire)``.
        self.answers: list[tuple[str, object]] = []
        #: Things the client did that the real server would reject.
        self.violations: list[str] = []
        #: How many processes were started.
        self.spawned = 0
        #: How many of them have seen their stdin closed.
        self.closed = 0

    def turn(self, *steps: CodexStep) -> CodexScript:
        """Queue one turn made of *steps*; returns the script for chaining."""
        self._queued.append(steps)
        return self

    def add_thread(self, thread_id: str, *, cwd: str = "/work") -> None:
        """Put a thread in the rollout store, as if an earlier process had made it."""
        self._threads[thread_id] = _Thread(thread_id, cwd, replay_usage=True)

    def forget(self, thread_id: str) -> None:
        """Drop a thread, as if its rollout had been deleted."""
        self._threads.pop(thread_id, None)

    def thread_ids(self) -> list[str]:
        """The threads in the rollout store."""
        return list(self._threads)

    def fail_mcp_server(self, name: str, error: str = "failed to start") -> None:
        """Make the MCP server *name* report a start-up failure on every thread."""
        self._failing_mcp[name] = error

    def peer(self, request: SpawnRequest | None = None) -> CodexAppServerPeer:
        """A fresh server process on this script; pass as ``FakeExecutor(peers=...)``."""
        del request  # the fake serves the same script whatever argv it was started with
        self.spawned += 1
        return CodexAppServerPeer(self)

    def requests(self, method: str) -> list[ClientRequest]:
        """The requests the client sent for *method*, in order."""
        return [
            m for m in self.received if isinstance(m, ClientRequest) and method == m.params.METHOD
        ]

    def next_turn(self) -> tuple[CodexStep, ...]:
        """Take the next queued turn, or a plain "ok"."""
        return self._queued.popleft() if self._queued else (Say("ok"),)

    def counter(self, prefix: str) -> str:
        """The next id for *prefix*: ``thread-1``, ``turn-2``, ..."""
        count = self._counts.get(prefix, 0) + 1
        self._counts[prefix] = count
        return f"{prefix}-{count}"

    def thread(self, thread_id: str) -> _Thread | None:
        """The stored thread, if any."""
        return self._threads.get(thread_id)

    def store(self, thread: _Thread) -> None:
        """Keep *thread* in the rollout store."""
        self._threads[thread.thread_id] = thread

    def mcp_failure(self, name: str) -> str | None:
        """The start-up error scripted for MCP server *name*."""
        return self._failing_mcp.get(name)


# -- the process


@dataclass
class _Waiting:
    """A server request the turn is blocked on."""

    request_id: int
    method: str
    step: CodexStep
    item: Mapping[str, object] | None = None


@dataclass
class _Run:
    """The turn in progress."""

    thread: _Thread
    turn_id: str
    steps: deque[CodexStep]
    approval: str
    items: list[Mapping[str, object]] = field(default_factory=_no_items)
    waiting: _Waiting | None = None
    hanging: bool = False
    deaf: bool = False
    spent: bool = False


class CodexAppServerPeer:
    """One fake ``codex app-server`` process (a ``FakePeer``)."""

    def __init__(self, script: CodexScript) -> None:
        """Serve *script*. Made by ``CodexScript.peer``."""
        self._script = script
        self._buffer = ""
        self._initialized = False
        self._ready = False
        self._dead = False
        self._clock_ms = 1_790_000_000_000
        self._server_ids = 0
        self._threads: dict[str, dict[str, str]] = {}
        self._run: _Run | None = None
        self._abandoned: set[int] = set()

    # -- FakePeer

    def on_start(self) -> Sequence[PeerOutput]:
        """A real server says nothing until spoken to."""
        return ()

    def on_stdin(self, data: str) -> Sequence[PeerOutput]:
        """Answer the lines in *data*."""
        out: list[PeerOutput] = []
        self._buffer += data
        while "\n" in self._buffer and not self._dead:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self._on_line(line, out)
        return out

    def on_stdin_closed(self) -> Sequence[PeerOutput]:
        """Exit cleanly, as the real server does at end of input."""
        self._script.closed += 1
        if self._dead or not self._script.exits_at_end_of_input:
            return ()
        self._dead = True
        return [ProcessExited(0)]

    # -- dispatch

    def _on_line(self, line: str, out: list[PeerOutput]) -> None:
        raw = json.loads(line)
        try:
            message = parse_client_message(raw)
        except CodexProtocolError as error:
            self._script.violations.append(f"unparseable client message: {error}")
            if isinstance(raw, dict):
                members = cast("dict[str, object]", raw)
                if "id" in members and "method" in members:
                    method = members["method"]
                    self._error(out, members["id"], _METHOD_NOT_FOUND, f"unsupported {method}")
            return
        self._script.received.append(message)
        if isinstance(message, ClientRequest):
            self._on_request(message, out)
        elif isinstance(message, ClientNotification):
            self._ready = self._initialized
            if not self._initialized:
                self._script.violations.append("initialized before initialize")
        else:
            self._on_reply(message, raw, out)

    def _on_request(self, request: ClientRequest, out: list[PeerOutput]) -> None:
        params = request.params
        if params.METHOD == "initialize":
            self._initialized = True
            self._result(
                out,
                request.id,
                {
                    "userAgent": "agentshim-fake/0.160.0",
                    "codexHome": "/codex-home",
                    "platformFamily": "unix",
                    "platformOs": "linux",
                },
            )
            return
        if not self._ready:
            self._script.violations.append(f"{params.METHOD} before the handshake finished")
            self._error(out, request.id, _INVALID_REQUEST, "Not initialized")
        elif isinstance(params, ThreadStartParams):
            self._start_thread(request.id, params, out)
        elif isinstance(params, ThreadResumeParams):
            self._resume_thread(request.id, params, out)
        elif isinstance(params, TurnStartParams):
            self._start_turn(request.id, params, out)
        elif isinstance(params, TurnInterruptParams):
            self._interrupt(request.id, params, out)
        else:
            self._error(out, request.id, _METHOD_NOT_FOUND, f"unsupported {params.METHOD}")

    def _on_reply(
        self, message: Response | ErrorResponse, raw: object, out: list[PeerOutput]
    ) -> None:
        run = self._run
        waiting = None if run is None else run.waiting
        if message.id in self._abandoned:
            return  # the turn ended while this was in flight; the client is right to answer it
        if run is None or waiting is None or message.id != waiting.request_id:
            self._script.violations.append(f"reply to a request never made: {raw!r}")
            return
        wire = (
            message.result if isinstance(message, Response) else {"error": message.error.to_wire()}
        )
        self._script.answers.append((waiting.method, wire))
        run.waiting = None
        self._notify(
            out,
            "serverRequest/resolved",
            {"threadId": run.thread.thread_id, "requestId": waiting.request_id},
        )
        self._resolve(run, waiting, wire, out)
        self._advance(run, out)

    # -- threads

    def _start_thread(
        self, request_id: object, params: ThreadStartParams, out: list[PeerOutput]
    ) -> None:
        thread = _Thread(self._script.counter("thread"), params.cwd or "/work")
        self._script.store(thread)
        self._open_thread(request_id, thread, params, out, resumed=False)

    def _resume_thread(
        self, request_id: object, params: ThreadResumeParams, out: list[PeerOutput]
    ) -> None:
        thread = self._script.thread(params.thread_id)
        if thread is None:
            message = f"no rollout found for thread id {params.thread_id}"
            self._error(out, request_id, _INVALID_REQUEST, message)
            return
        self._open_thread(request_id, thread, params, out, resumed=True)

    def _open_thread(
        self,
        request_id: object,
        thread: _Thread,
        params: ThreadStartParams | ThreadResumeParams,
        out: list[PeerOutput],
        *,
        resumed: bool,
    ) -> None:
        sandbox = self._applied_sandbox(params.sandbox.value if params.sandbox else None)
        approval = self._applied_approval(_approval_name(params.approval_policy))
        self._threads[thread.thread_id] = {"sandbox": sandbox, "approval": approval}
        thread.cwd = params.cwd or thread.cwd
        thread.replay_usage = resumed
        wire = self._thread_wire(thread)
        self._result(
            out,
            request_id,
            {
                "thread": wire,
                "model": params.model or self._script.model,
                "modelProvider": "openai",
                "cwd": thread.cwd,
                "approvalPolicy": approval,
                "approvalsReviewer": "user",
                "sandbox": _sandbox_wire(sandbox, thread.cwd),
                "reasoningEffort": None,
            },
        )
        self._notify(out, "thread/started", {"thread": wire})
        self._announce_mcp(thread, params.config, out)

    def _announce_mcp(self, thread: _Thread, config: object, out: list[PeerOutput]) -> None:
        table = (
            cast("dict[str, object]", config).get("mcp_servers")
            if isinstance(config, dict)
            else None
        )
        if not isinstance(table, dict):
            return
        for name in cast("dict[str, object]", table):
            base: dict[str, object] = {
                "threadId": thread.thread_id,
                "name": name,
                "failureReason": None,
            }
            self._notify(
                out,
                "mcpServer/startupStatus/updated",
                {**base, "status": "starting", "error": None},
            )
            error = self._script.mcp_failure(str(name))
            status = (
                {"status": "ready", "error": None}
                if error is None
                else {"status": "failed", "error": error}
            )
            self._notify(out, "mcpServer/startupStatus/updated", {**base, **status})

    def _applied_sandbox(self, requested: str | None) -> str:
        if self._script.ignore_requested_permissions or requested is None:
            return self._script.inherited_sandbox
        return requested

    def _applied_approval(self, requested: str | None) -> str:
        if self._script.ignore_requested_permissions or requested is None:
            return self._script.inherited_approval
        return requested

    def _thread_wire(self, thread: _Thread) -> dict[str, object]:
        return {
            "id": thread.thread_id,
            "sessionId": thread.thread_id,
            "preview": "",
            "ephemeral": False,
            "projectId": None,
            "historyMode": "paginated",
            "modelProvider": "openai",
            "model": self._script.model,
            "createdAt": 1,
            "updatedAt": 1,
            "status": {"type": "idle"},
            "path": f"/codex-home/sessions/rollout-{thread.thread_id}.jsonl",
            "cwd": thread.cwd,
            "cliVersion": "0.160.0",
            "source": "vscode",
            "turns": [],
        }

    # -- turns

    def _start_turn(
        self, request_id: object, params: TurnStartParams, out: list[PeerOutput]
    ) -> None:
        thread = self._script.thread(params.thread_id)
        if thread is None or params.thread_id not in self._threads:
            self._error(out, request_id, _INVALID_REQUEST, f"thread not found: {params.thread_id}")
            return
        if self._run is not None:
            self._error(out, request_id, _INVALID_REQUEST, "a turn is already running")
            return
        settings = self._threads[params.thread_id]
        if params.approval_policy is not None and not self._script.ignore_requested_permissions:
            settings["approval"] = _approval_name(params.approval_policy) or settings["approval"]
        run = _Run(
            thread,
            self._script.counter("turn"),
            deque(self._script.next_turn()),
            settings["approval"],
        )
        self._run = run
        if thread.replay_usage:
            thread.replay_usage = False
            self._notify(
                out, "thread/tokenUsage/updated", self._usage_wire(thread, _PREVIOUS_TURN, Spend())
            )
        turn: dict[str, object] = {
            "id": run.turn_id,
            "items": [],
            "itemsView": "notLoaded",
            "status": "inProgress",
            "error": None,
        }
        self._result(out, request_id, {"turn": turn})
        self._notify(
            out,
            "thread/status/changed",
            {"threadId": thread.thread_id, "status": {"type": "active", "activeFlags": []}},
        )
        self._notify(out, "turn/started", {"threadId": thread.thread_id, "turn": turn})
        first = params.input[0] if params.input else None
        text = first.text if isinstance(first, UserInputText) else ""
        message: dict[str, object] = {
            "type": "userMessage",
            "id": self._script.counter("item"),
            "content": [{"type": "text", "text": text, "text_elements": []}],
        }
        self._item(out, run, "item/started", message)
        self._item(out, run, "item/completed", message)
        self._advance(run, out)

    def _advance(self, run: _Run, out: list[PeerOutput]) -> None:
        while self._run is run and run.waiting is None and not run.hanging:
            if not run.steps:
                self._complete(run, out)
                return
            self._step(run, run.steps.popleft(), out)

    def _step(self, run: _Run, step: CodexStep, out: list[PeerOutput]) -> None:
        handlers: dict[type, Callable[[_Run, Any, list[PeerOutput]], None]] = {
            Say: self._say,
            Think: self._think,
            RunCommand: self._command,
            ChangeFile: self._file_change,
            CallMcp: self._mcp_call,
            Ask: self._ask,
            Spend: self._spend,
            ReportRateLimits: self._report_rate_limits,
            Retrying: self._retrying,
            Fail: self._fail,
            Hang: self._hang,
            Crash: self._crash,
            Complain: self._complain,
        }
        handlers[type(step)](run, step, out)

    def _think(self, run: _Run, step: Think, out: list[PeerOutput]) -> None:
        item = {
            "type": "reasoning",
            "id": self._script.counter("item"),
            "summary": [step.text],
            "content": [],
        }
        self._item(out, run, "item/started", item)
        self._item(out, run, "item/completed", item)

    def _retrying(self, run: _Run, step: Retrying, out: list[PeerOutput]) -> None:
        self._error_notification(out, run, step.message, will_retry=True)

    def _hang(self, run: _Run, step: Hang, out: list[PeerOutput]) -> None:
        del out
        run.hanging = True
        run.deaf = step.ignores_interrupt

    def _complain(self, run: _Run, step: Complain, out: list[PeerOutput]) -> None:
        del run
        out.append(StderrLine(step.text + "\n"))

    def _say(self, run: _Run, step: Say, out: list[PeerOutput]) -> None:
        item_id = self._script.counter("item")
        started = {"type": "agentMessage", "id": item_id, "text": "", "phase": step.phase}
        self._item(out, run, "item/started", started)
        delta = {
            "threadId": run.thread.thread_id,
            "turnId": run.turn_id,
            "itemId": item_id,
            "delta": step.text,
        }
        self._notify(out, "item/agentMessage/delta", delta)
        done = {**started, "text": step.text}
        run.items.append(done)
        self._item(out, run, "item/completed", done)

    def _command(self, run: _Run, step: RunCommand, out: list[PeerOutput]) -> None:
        item: dict[str, object] = {
            "type": "commandExecution",
            "id": self._script.counter("item"),
            "command": step.command,
            "cwd": run.thread.cwd,
            "commandActions": [],
            "source": "agent",
            "status": "inProgress",
            "aggregatedOutput": None,
            "exitCode": None,
        }
        self._item(out, run, "item/started", item)
        if step.approval and run.approval != "never":
            params: dict[str, object] = {
                "kind": "command",
                "threadId": run.thread.thread_id,
                "turnId": run.turn_id,
                "itemId": item["id"],
                "startedAtMs": self._tick(),
                "command": step.command,
                "cwd": run.thread.cwd,
            }
            run.waiting = _Waiting(
                self._request(out, "item/commandExecution/requestApproval", params),
                "item/commandExecution/requestApproval",
                step,
                item,
            )
            return
        self._finish_command(run, step, item, out, ran=True)

    def _finish_command(
        self,
        run: _Run,
        step: RunCommand,
        item: Mapping[str, object],
        out: list[PeerOutput],
        *,
        ran: bool,
    ) -> None:
        if ran:
            done = {
                **item,
                "status": "completed",
                "aggregatedOutput": step.output,
                "exitCode": step.exit_code,
                "durationMs": 5,
            }
        else:
            done = {**item, "status": "declined"}
        self._item(out, run, "item/completed", done)

    def _file_change(self, run: _Run, step: ChangeFile, out: list[PeerOutput]) -> None:
        item = {
            "type": "fileChange",
            "id": self._script.counter("item"),
            "changes": [{"path": step.path, "kind": {"type": "update"}, "diff": ""}],
            "status": "inProgress",
        }
        self._item(out, run, "item/started", item)
        if step.approval and run.approval != "never":
            params = {
                "threadId": run.thread.thread_id,
                "turnId": run.turn_id,
                "itemId": item["id"],
                "startedAtMs": self._tick(),
            }
            run.waiting = _Waiting(
                self._request(out, "item/fileChange/requestApproval", params),
                "item/fileChange/requestApproval",
                step,
                item,
            )
            return
        self._item(out, run, "item/completed", {**item, "status": "completed"})

    def _mcp_call(self, run: _Run, step: CallMcp, out: list[PeerOutput]) -> None:
        item = {
            "type": "mcpToolCall",
            "id": self._script.counter("item"),
            "server": step.server,
            "tool": step.tool,
            "arguments": dict(step.arguments or {}),
            "status": "inProgress",
        }
        self._item(out, run, "item/started", item)
        if step.error is not None:
            done = {**item, "status": "failed", "error": {"message": step.error}, "durationMs": 3}
        else:
            result = {"content": [{"type": "text", "text": step.output}]}
            done = {**item, "status": "completed", "result": result, "durationMs": 3}
        self._item(out, run, "item/completed", done)

    def _ask(self, run: _Run, step: Ask, out: list[PeerOutput]) -> None:
        thread, turn = run.thread.thread_id, run.turn_id
        ask = step.kind
        params: Mapping[str, object]
        if ask is AskKind.PERMISSIONS:
            params = {
                "threadId": thread,
                "turnId": turn,
                "itemId": "ask",
                "startedAtMs": 1,
                "cwd": run.thread.cwd,
                "permissions": {},
                "reason": "needs more access",
            }
        elif ask is AskKind.ELICITATION:
            params = {"threadId": thread, "turnId": turn, "serverName": "docs"}
        elif ask is AskKind.USER_INPUT:
            question = {"header": "Q", "id": "q1", "question": "Which one?"}
            params = {
                "threadId": thread,
                "turnId": turn,
                "itemId": "ask",
                "isBlocking": True,
                "questions": [question],
            }
        elif ask is AskKind.LEGACY_EXEC:
            params = {
                "callId": "c1",
                "conversationId": thread,
                "command": ["rm", "-rf", "x"],
                "cwd": run.thread.cwd,
                "parsedCmd": [],
            }
        elif ask is AskKind.TOKEN_REFRESH:
            params = {"reason": "unauthorized"}
        else:
            params = {}
        run.waiting = _Waiting(self._request(out, ask.value, params), ask.value, step)

    def _spend(self, run: _Run, step: Spend, out: list[PeerOutput]) -> None:
        run.spent = True
        totals = run.thread.totals
        totals["inputTokens"] += step.input_tokens
        totals["cachedInputTokens"] += step.cached_input_tokens
        totals["outputTokens"] += step.output_tokens
        totals["reasoningOutputTokens"] += step.reasoning_output_tokens
        totals["totalTokens"] += step.input_tokens + step.output_tokens
        self._notify(
            out, "thread/tokenUsage/updated", self._usage_wire(run.thread, run.turn_id, step)
        )

    def _report_rate_limits(self, run: _Run, step: ReportRateLimits, out: list[PeerOutput]) -> None:
        del run

        def window(spec: tuple[int, int | None, int | None] | None) -> object:
            if spec is None:
                return None
            used, minutes, resets = spec
            return {"usedPercent": used, "windowDurationMins": minutes, "resetsAt": resets}

        snapshot = {
            "limitId": step.limit_id,
            "limitName": None,
            "primary": window(step.primary),
            "secondary": window(step.secondary),
            "credits": None,
            "rateLimitReachedType": step.reached,
        }
        self._notify(out, "account/rateLimits/updated", {"rateLimits": snapshot})

    def _usage_wire(self, thread: _Thread, turn_id: str, last: Spend) -> Mapping[str, object]:
        last_wire = {
            "totalTokens": last.input_tokens + last.output_tokens,
            "inputTokens": last.input_tokens,
            "cachedInputTokens": last.cached_input_tokens,
            "cacheWriteInputTokens": 0,
            "outputTokens": last.output_tokens,
            "reasoningOutputTokens": last.reasoning_output_tokens,
        }
        usage = {
            "total": dict(thread.totals),
            "last": last_wire,
            "modelContextWindow": _CONTEXT_WINDOW,
        }
        return {"threadId": thread.thread_id, "turnId": turn_id, "tokenUsage": usage}

    def _fail(self, run: _Run, step: Fail, out: list[PeerOutput]) -> None:
        error = {"message": step.message, "codexErrorInfo": step.info, "additionalDetails": None}
        self._notify(
            out,
            "error",
            {
                "error": error,
                "willRetry": False,
                "threadId": run.thread.thread_id,
                "turnId": run.turn_id,
            },
        )
        self._finish(run, "failed", out, error=error)

    def _crash(self, run: _Run, step: Crash, out: list[PeerOutput]) -> None:
        del run
        self._dead = True
        self._run = None
        out.append(StderrLine(step.stderr + "\n"))
        out.append(ProcessExited(step.returncode))

    def _error_notification(
        self, out: list[PeerOutput], run: _Run, message: str, *, will_retry: bool
    ) -> None:
        error = {"message": message, "codexErrorInfo": "other", "additionalDetails": None}
        self._notify(
            out,
            "error",
            {
                "error": error,
                "willRetry": will_retry,
                "threadId": run.thread.thread_id,
                "turnId": run.turn_id,
            },
        )

    def _complete(self, run: _Run, out: list[PeerOutput]) -> None:
        if not run.spent:
            self._spend(run, DEFAULT_SPEND, out)
        self._finish(run, "completed", out)

    def _finish(
        self, run: _Run, status: str, out: list[PeerOutput], *, error: object = None
    ) -> None:
        run.thread.turns += 1
        self._run = None
        if run.waiting is not None:
            self._abandoned.add(run.waiting.request_id)
            run.waiting = None
        self._notify(
            out,
            "thread/status/changed",
            {"threadId": run.thread.thread_id, "status": {"type": "idle"}},
        )
        items = [
            item
            for item in run.items
            if status == "completed" and item.get("phase") == "final_answer"
        ]
        turn = {
            "id": run.turn_id,
            "items": items,
            "itemsView": "summary" if items else "notLoaded",
            "status": status,
            "error": error,
            "durationMs": 7,
        }
        self._notify(out, "turn/completed", {"threadId": run.thread.thread_id, "turn": turn})

    def _resolve(self, run: _Run, waiting: _Waiting, wire: object, out: list[PeerOutput]) -> None:
        """Carry on after the client answered the request the turn was blocked on."""
        decision = _decision(wire)
        accepted = decision in ("accept", "acceptForSession")
        if isinstance(waiting.step, RunCommand) and waiting.item is not None:
            self._finish_command(run, waiting.step, waiting.item, out, ran=accepted)
        elif isinstance(waiting.step, ChangeFile) and waiting.item is not None:
            status = "completed" if accepted else "declined"
            self._item(out, run, "item/completed", {**waiting.item, "status": status})
        if decision == "cancel":
            run.steps.clear()
            self._finish(run, "interrupted", out)

    def _interrupt(
        self, request_id: object, params: TurnInterruptParams, out: list[PeerOutput]
    ) -> None:
        run = self._run
        if run is None or run.turn_id != params.turn_id:
            self._error(out, request_id, _INVALID_REQUEST, "no such running turn")
            return
        self._result(out, request_id, {})
        if run.deaf:
            return
        run.steps.clear()
        self._finish(run, "interrupted", out)

    # -- wire helpers

    def _item(
        self, out: list[PeerOutput], run: _Run, method: str, item: Mapping[str, object]
    ) -> None:
        stamp = "startedAtMs" if method == "item/started" else "completedAtMs"
        self._notify(
            out,
            method,
            {
                "item": item,
                "threadId": run.thread.thread_id,
                "turnId": run.turn_id,
                stamp: self._tick(),
            },
        )

    def _request(self, out: list[PeerOutput], method: str, params: Mapping[str, object]) -> int:
        request_id = self._server_ids
        self._server_ids += 1
        self._emit(out, {"method": method, "id": request_id, "params": params})
        return request_id

    def _notify(self, out: list[PeerOutput], method: str, params: Mapping[str, object]) -> None:
        self._emit(out, {"method": method, "params": params, "emittedAtMs": self._tick()})

    def _result(
        self, out: list[PeerOutput], request_id: object, result: Mapping[str, object]
    ) -> None:
        self._emit(out, {"id": request_id, "result": result})

    def _error(self, out: list[PeerOutput], request_id: object, code: int, message: str) -> None:
        self._emit(out, {"error": {"code": code, "message": message}, "id": request_id})

    @staticmethod
    def _emit(out: list[PeerOutput], message: Mapping[str, object]) -> None:
        out.append(StdoutLine(json.dumps(message, separators=(",", ":")) + "\n"))

    def _tick(self) -> int:
        self._clock_ms += 1
        return self._clock_ms


def _approval_name(policy: object) -> str | None:
    """The wire name of an approval policy (an enum member's value, or a bare string)."""
    value = getattr(policy, "value", policy)
    return value if isinstance(value, str) else None


def _sandbox_wire(mode: str, cwd: str) -> Mapping[str, object]:
    if mode == "read-only":
        return {"type": "readOnly", "networkAccess": False}
    if mode == "workspace-write":
        return {
            "type": "workspaceWrite",
            "writableRoots": [cwd],
            "networkAccess": False,
            "excludeSlashTmp": False,
            "excludeTmpdirEnvVar": False,
        }
    return {"type": "dangerFullAccess"}


def _decision(wire: object) -> str | None:
    """The ``decision`` of a client's reply to an approval request."""
    if isinstance(wire, dict):
        value = cast("dict[str, object]", wire).get("decision")
        return value if isinstance(value, str) else None
    return None
