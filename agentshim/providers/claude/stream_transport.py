"""``ClaudeStreamTransport``: Claude Code over one long-lived ``stream-json`` process.

One ``claude`` process serves a whole conversation. Prompts are written to its
stdin as user messages and each turn ends at the next ``result`` frame; the
process keeps its context between turns, so a turn costs one model round trip
and not a process start plus a transcript reload.

Choices that follow from how the CLI behaves (verified against the recordings
in ``tests/fixtures/claude_stream``):

* ``system/init`` is repeated at the start of every turn, so readiness is the
  reply to the ``initialize`` request, not ``init``.
* ``total_cost_usd`` is cumulative over the process while ``usage`` is per turn,
  so the cost of a turn is the difference from the previous result.
* A failed API request arrives as a ``result`` with subtype ``success`` and
  ``is_error`` true; a failure is classified on ``is_error`` first.
* ``--json-schema``, MCP servers, effort and extra arguments are flags of the
  process, while the schema is chosen per turn. A turn that needs different
  ones restarts the process with ``--resume`` of the same conversation, which
  keeps the conversation id (and so the session's continuity) unchanged.
* A refused ``--resume`` makes the CLI write one error ``result`` and exit
  before it answers ``initialize``; that is a ``SessionResumeError`` from
  ``open``.
"""

from __future__ import annotations

import contextlib
import json
import threading
from collections import deque
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast

from agentshim.core.clock import SystemClock
from agentshim.core.env import interactive_env
from agentshim.core.errors import (
    AgentShimError,
    CliExitError,
    FailureKind,
    ProcessClosedError,
    ProviderCapabilityError,
    SchemaDialectError,
    SessionResumeError,
    SessionStateError,
    TurnCancelledError,
    TurnFailedError,
    TurnTimeoutError,
)
from agentshim.core.events import ApprovalDenied, TurnInterrupted
from agentshim.core.ids import RandomIds
from agentshim.core.permissions import ApprovalPolicy, NativeMode
from agentshim.core.profile import SchemaDialect
from agentshim.core.schema import compact_json, dialect_problems
from agentshim.core.skills import SkillTracker
from agentshim.core.stream import parse_json_object
from agentshim.core.turn import TurnResult
from agentshim.execution.host import HostCommandExecutor
from agentshim.execution.process import ProcessExited, SpawnRequest, StderrLine, StdoutLine

from .events import ResultFrame, parse_frame
from .failures import classify_failure
from .parser import ClaudeStreamParser
from .provider import PROFILE
from .stream_argv import ProcessConfig, stream_argv, stream_env

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from agentshim.core.clock import Clock
    from agentshim.core.conversation import Conversation, ConversationSpec
    from agentshim.core.events import AgentEvent
    from agentshim.core.ids import IdAllocator
    from agentshim.core.profile import ProviderProfile
    from agentshim.core.provider import ParsedTurn
    from agentshim.core.turn import TurnRequest
    from agentshim.execution.executor import CommandExecutor
    from agentshim.execution.process import Process, ProcessOutput

    from .user_hooks import ClaudeHook

#: How often an idle pull loop looks at the clock again.
POLL_S = 0.05

#: Stderr lines kept for the error of a process that dies.
STDERR_TAIL_LINES = 40

#: What the CLI is told when a permission request is refused.
DENY_MESSAGE = "denied by policy: this agent runs without a person to approve"

#: Claude Code's reason when an interrupt ended the turn; ``aborted_streaming``
#: while the model was talking, ``aborted_tools`` while a tool was running.
_ABORTED_PREFIX = "aborted"

_NO_CONVERSATION = "No conversation found"

#: Native permission modes this transport enforces: ``READ_ONLY`` is absent
#: because Claude Code's sandbox always grants the working directory (and
#: temp directories) to commands, so "may read but not write" cannot be
#: promised for what the agent runs.
STREAM_PERMISSION_MODES = frozenset({NativeMode.BYPASS, NativeMode.WORKSPACE_WRITE})

STREAM_PROFILE: ProviderProfile = replace(PROFILE, native_permission_modes=STREAM_PERMISSION_MODES)


@dataclass(frozen=True)
class _Runtime:
    """Everything a conversation borrows from its transport."""

    executor: CommandExecutor
    binary: str
    env: Mapping[str, str]
    clock: Clock
    ids: IdAllocator
    hooks: tuple[ClaudeHook, ...]
    startup_timeout_s: float
    interrupt_grace_s: float
    close_grace_s: float


class ClaudeStreamTransport:
    """A ``Transport`` over a long-lived ``claude --input-format stream-json`` process.

    Construction resolves the binary on *executor* and runs its health check
    once, like ``CliAgent``, so a broken install fails here. Only
    ``NativeMode.BYPASS`` and ``NativeMode.WORKSPACE_WRITE`` without network
    are supported (see ``STREAM_PERMISSION_MODES``). *hooks* are Claude Code
    hooks added to the settings of every process.
    """

    # Each argument is an independent documented option.
    def __init__(  # noqa: PLR0913
        self,
        *,
        executor: CommandExecutor | None = None,
        env: Mapping[str, str] | None = None,
        clock: Clock | None = None,
        ids: IdAllocator | None = None,
        hooks: Sequence[ClaudeHook] = (),
        check_timeout: float = 15.0,
        startup_timeout_s: float = 60.0,
        interrupt_grace_s: float = 10.0,
        close_grace_s: float = 5.0,
        log: Callable[[str], None] | None = None,
    ) -> None:
        """Check the install of ``claude`` on *executor* (the local host by default).

        ``startup_timeout_s`` bounds the wait for a new process to answer
        ``initialize``; ``interrupt_grace_s`` how long a timed-out turn gets to
        wind down after its interrupt before the process is killed;
        ``close_grace_s`` each step of a shutdown (EOF, terminate, kill).
        """
        executor = executor if executor is not None else HostCommandExecutor()
        base_env = dict(env) if env is not None else interactive_env()
        binary = executor.find_binary(PROFILE.binary, base_env)
        executor.check_binary(binary, base_env, timeout=check_timeout)
        if log is not None:
            log(f"{PROFILE.display_name} ready at {binary}")
        self._runtime = _Runtime(
            executor=executor,
            binary=binary,
            env=base_env,
            clock=clock if clock is not None else SystemClock(),
            ids=ids if ids is not None else RandomIds(),
            hooks=tuple(hooks),
            startup_timeout_s=startup_timeout_s,
            interrupt_grace_s=interrupt_grace_s,
            close_grace_s=close_grace_s,
        )

    @property
    def profile(self) -> ProviderProfile:
        """Claude Code's profile, with the permission modes this transport enforces."""
        return STREAM_PROFILE

    def open(self, spec: ConversationSpec) -> Conversation:
        """Start the process, resuming ``spec.resume_id``, and wait until it is ready.

        Raises ``ProviderCapabilityError`` for a permission mode outside
        ``STREAM_PERMISSION_MODES`` or for network access, ``SessionResumeError``
        when the conversation to resume is gone, and ``TurnFailedError`` (a
        ``CliExitError``) when the process dies before it is ready.
        """
        _check_spec(spec)
        conversation = _StreamConversation(self._runtime, spec)
        try:
            conversation.start()
        except BaseException:
            conversation.close()
            raise
        return conversation


def _check_spec(spec: ConversationSpec) -> None:
    permissions = spec.permissions
    if permissions.mode not in STREAM_PERMISSION_MODES:
        msg = (
            f"{PROFILE.name} cannot enforce native permission mode {permissions.mode.value!r}; "
            f"supported: {sorted(mode.value for mode in STREAM_PERMISSION_MODES)}"
        )
        raise ProviderCapabilityError(msg)
    if permissions.network:
        msg = "claude cannot enforce network access for its sandboxed commands"
        raise ProviderCapabilityError(msg)


def _lines() -> deque[str]:
    return deque(maxlen=STDERR_TAIL_LINES)


def _names() -> set[str]:
    return set()


@dataclass
class _TurnRun:
    """What one running turn needs while its output is pulled."""

    parser: ClaudeStreamParser
    emit: Callable[[AgentEvent], None]
    timeout_s: float | None
    deadline: float | None
    #: The turn ran out of time; its interrupt is sent and its result awaited.
    expired: bool = False
    grace_deadline: float = 0.0
    #: Tool calls already reported as denied (through a request or the result).
    denied: set[str] = field(default_factory=_names)
    #: Set when a permission request ended the turn (``ApprovalPolicy.FAIL_TURN``).
    approval_failure: str | None = None


class _StreamConversation:
    """One conversation: a process kept across turns, restarted when its flags must change."""

    def __init__(self, runtime: _Runtime, spec: ConversationSpec) -> None:
        self._rt = runtime
        self._spec = spec
        self._id: str | None = spec.resume_id
        self._lock = threading.Lock()
        #: Serializes writes to stdin, which a turn and an interrupt both make.
        self._write_lock = threading.Lock()
        self._process: Process | None = None
        self._argv: tuple[str, ...] = ()
        self._config = ProcessConfig()
        self._cost_baseline = 0.0
        self._stderr_tail: deque[str] = _lines()
        self._closed = False
        self._running = False
        self._turn_seq = 0
        self._prompt_sent = False
        self._interrupt_requested = False

    # -- Conversation protocol

    @property
    def conversation_id(self) -> str | None:
        """The session id the CLI named, or the one being resumed."""
        return self._id

    def start(self) -> None:
        """Spawn the first process for the spec alone, before any turn."""
        self._launch(self._base_config())

    def turn(self, request: TurnRequest, emit: Callable[[AgentEvent], None]) -> TurnResult:
        """Write the prompt and pump output until this turn's ``result`` frame."""
        self._begin_turn()
        try:
            return self._run_turn(request, emit)
        finally:
            self._end_turn()

    def interrupt(self) -> None:
        """Ask the CLI to stop the running turn; the process and conversation survive."""
        with self._lock:
            if not self._running or self._closed:
                return
            self._interrupt_requested = True
            seq, sent = self._turn_seq, self._prompt_sent
        if sent:
            self._send_interrupt(seq)

    def close(self) -> None:
        """End stdin, wait a bounded time, then terminate and kill. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            process, self._process = self._process, None
        if process is not None:
            self._stop(process)

    # -- turn lifecycle

    def _begin_turn(self) -> None:
        with self._lock:
            if self._closed:
                msg = "conversation is closed"
                raise SessionStateError(msg)
            if self._running:
                msg = "a turn is already running in this conversation"
                raise SessionStateError(msg)
            self._running = True
            self._turn_seq += 1
            self._prompt_sent = False
            self._interrupt_requested = False

    def _end_turn(self) -> None:
        with self._lock:
            self._running = False

    def _run_turn(self, request: TurnRequest, emit: Callable[[AgentEvent], None]) -> TurnResult:
        config = self._config_for(request)
        skills = SkillTracker(STREAM_PROFILE)

        def relay(event: AgentEvent) -> None:
            skills.on_event(event)
            emit(event)

        clock = self._rt.clock
        started = clock.monotonic()
        resumed = self._id is not None
        self._drain_idle()
        self._ensure_process(config)
        parser = ClaudeStreamParser(
            relay,
            expect_structured=request.output_schema is not None,
            cost_baseline_usd=self._cost_baseline,
        )
        deadline = None if request.timeout is None else started + request.timeout
        run = _TurnRun(parser, relay, request.timeout, deadline)
        self._send_prompt(request.prompt)
        frame = self._read_result(run)
        parsed = self._settle(run, frame)
        duration_ms = int((clock.monotonic() - started) * 1000)
        result = TurnResult(
            text=parsed.text,
            structured_output=parsed.structured_output,
            session_id=self._id,
            resumed=resumed,
            usage=parsed.usage,
            cost_usd=parsed.cost_usd,
            duration_ms=duration_ms,
            exit_code=0,
            skills=skills.summary(),
            interrupted=_interrupted(frame),
        )
        if result.interrupted:
            relay(TurnInterrupted())
        return result

    # -- configuration and process management

    def _base_config(self) -> ProcessConfig:
        spec = self._spec
        return ProcessConfig(
            reasoning_effort=spec.reasoning_effort,
            mcp_servers=tuple(spec.mcp_servers),
        )

    def _config_for(self, request: TurnRequest) -> ProcessConfig:
        """The process flags this turn needs, after checking what it asks for."""
        spec = self._spec
        if request.cwd is not None and request.cwd != spec.cwd:
            msg = (
                f"a conversation's working directory is fixed at {spec.cwd!r}; "
                f"the turn asked for {request.cwd!r}"
            )
            raise ProviderCapabilityError(msg)
        schema = None
        if request.output_schema is not None:
            problems = dialect_problems(request.output_schema.schema, SchemaDialect.OPEN)
            if problems:
                raise SchemaDialectError(problems)
            schema = compact_json(request.output_schema.schema)
        servers = tuple(request.mcp_servers) or tuple(spec.mcp_servers)
        return ProcessConfig(
            schema=schema,
            reasoning_effort=request.reasoning_effort or spec.reasoning_effort,
            mcp_servers=servers,
            extra_args=tuple(request.extra_args),
            env=tuple(sorted((request.env or {}).items())),
        )

    def _ensure_process(self, config: ProcessConfig) -> None:
        """Make sure a live process runs with *config*, restarting (resuming) if not."""
        with self._lock:
            process = self._process
            same = process is not None and config == self._config
        if same:
            return
        if process is not None:
            with self._lock:
                self._process = None
            self._stop(process)
        self._launch(config)

    def _launch(self, config: ProcessConfig) -> None:
        """Spawn a process with *config*, resuming the conversation when it has an id."""
        rt = self._rt
        argv = stream_argv(rt.binary, self._spec, config, resume_id=self._id, hooks=rt.hooks)
        env = stream_env(rt.env, self._spec, config)
        process = rt.executor.spawn(SpawnRequest(argv=argv, cwd=self._spec.cwd, env=env))
        self._stderr_tail.clear()
        try:
            self._handshake(process, argv)
        except BaseException:
            process.kill()
            raise
        with self._lock:
            if self._closed:
                self._stop(process)
                msg = "conversation closed while its process started"
                raise TurnCancelledError(msg)
            self._process = process
            self._argv = tuple(argv)
            self._config = config
            self._cost_baseline = 0.0

    def _handshake(self, process: Process, argv: Sequence[str]) -> None:
        """Send ``initialize`` and wait for its answer, or explain why none came."""
        clock = self._rt.clock
        request_id = self._rt.ids.new_id("init")
        deadline = clock.monotonic() + self._rt.startup_timeout_s
        stdout: list[str] = []
        refusal: ResultFrame | None = None
        # A closed pipe is explained by what the process wrote before it died.
        with contextlib.suppress(ProcessClosedError):
            self._write(process, _control_request(request_id, {"subtype": "initialize"}))
        while True:
            item = self._poll(process)
            if item is None:
                if clock.monotonic() >= deadline:
                    msg = f"claude did not answer initialize within {self._rt.startup_timeout_s}s"
                    raise TurnTimeoutError(self._rt.startup_timeout_s, msg)
            elif isinstance(item, StderrLine):
                self._note_stderr(item.text)
            elif isinstance(item, ProcessExited):
                raise self._startup_failure(argv, item.returncode, stdout, refusal)
            else:
                stdout.append(item.text)
                refusal = self._startup_line(item.text) or refusal
                if refusal is None and self._answered(item.text, request_id):
                    return

    def _startup_line(self, text: str) -> ResultFrame | None:
        """The error ``result`` a starting process wrote instead of answering, if this is it."""
        data = parse_json_object(text)
        if data is None or data.get("type") != "result":
            return None
        frame = parse_frame(data)
        return frame if isinstance(frame, ResultFrame) else None

    def _answered(self, text: str, request_id: str) -> bool:
        data = parse_json_object(text)
        if data is None or data.get("type") != "control_response":
            return False
        response = data.get("response")
        if not isinstance(response, dict):
            return False
        body = cast("dict[str, Any]", response)
        if body.get("request_id") != request_id:
            return False
        if body.get("subtype") == "success":
            return True
        detail = str(body.get("error") or "initialize failed")
        msg = f"claude refused initialize: {detail}"
        raise TurnFailedError(msg, detail=detail)

    def _startup_failure(
        self,
        argv: Sequence[str],
        returncode: int,
        stdout: Sequence[str],
        refusal: ResultFrame | None,
    ) -> TurnFailedError:
        """Explain a process that exited before it was ready."""
        stderr = "\n".join(self._stderr_tail)
        out = "".join(stdout)
        reported = "; ".join(refusal.errors) or refusal.text if refusal is not None else ""
        detail = reported or stderr
        resume_id = self._id
        gone = _NO_CONVERSATION in detail or _NO_CONVERSATION in stderr
        if resume_id is not None and gone:
            return SessionResumeError(argv, returncode, resume_id, out, stderr, detail=detail)
        kind = classify_failure(
            subtype=refusal.subtype if refusal is not None else None,
            api_error=None,
            status=refusal.api_error_status if refusal is not None else None,
            text=detail,
        )
        return CliExitError(argv, returncode, out, stderr, kind=kind, detail=detail)

    def _stop(self, process: Process) -> None:
        """End a process politely: EOF, then terminate, then kill, each bounded."""
        grace = self._rt.close_grace_s
        process.close_stdin()
        if process.wait(grace) is not None:
            return
        process.terminate()
        if process.wait(grace) is not None:
            return
        process.kill()
        process.wait(grace)

    # -- stdin

    def _write(self, process: Process, line: dict[str, Any]) -> None:
        with self._write_lock:
            process.write(json.dumps(line) + "\n")

    def _send_prompt(self, prompt: str) -> None:
        """Write the user message; a pipe the process already closed ends the turn."""
        process = self._live_process()
        envelope = {
            "type": "user",
            "message": {"role": "user", "content": prompt},
            "parent_tool_use_id": None,
            "session_id": "default",
        }
        try:
            self._write(process, envelope)
        except ProcessClosedError as error:
            self._forget(process)
            raise self._died_error(None, process_gone=str(error)) from error
        with self._lock:
            self._prompt_sent = True
            seq, pending = self._turn_seq, self._interrupt_requested
        if pending:
            self._send_interrupt(seq)

    def _send_interrupt(self, seq: int) -> None:
        """Write an interrupt, unless the turn it was meant for has already ended."""
        request = _control_request(self._rt.ids.new_id("req"), {"subtype": "interrupt"})
        with self._write_lock:
            with self._lock:
                process = self._process
                current = self._running and seq == self._turn_seq
            if process is None or not current:
                return
            try:
                process.write(json.dumps(request) + "\n")
            except ProcessClosedError:
                return  # the pump sees the exit

    def _live_process(self) -> Process:
        with self._lock:
            process = self._process
            closed = self._closed
        if closed:
            msg = "conversation is closed"
            raise SessionStateError(msg)
        if process is None:  # pragma: no cover - _ensure_process always leaves one
            msg = "conversation has no process"
            raise SessionStateError(msg)
        return process

    def _forget(self, process: Process) -> None:
        """Drop *process*; the next turn starts a new one resuming this conversation."""
        with self._lock:
            if self._process is process:
                self._process = None

    # -- stdout

    def _poll(self, process: Process) -> ProcessOutput | None:
        """The next output item, or ``None`` after the clock let a little time pass."""
        item = process.next_output(0.0)
        if item is None:
            self._rt.clock.wait(POLL_S)
        return item

    def _note_stderr(self, text: str) -> None:
        stripped = text.rstrip("\n")
        if stripped:
            self._stderr_tail.append(stripped)

    def _drain_idle(self) -> None:
        """Read what an idle process left unread (late acks, stderr, or its exit).

        A process that exited between turns (after an error, say) is dropped,
        so the turn starts a new one resuming the conversation.
        """
        with self._lock:
            process = self._process
        if process is None:
            return
        while True:
            item = process.next_output(0.0)
            if item is None:
                return
            if isinstance(item, StderrLine):
                self._note_stderr(item.text)
            elif isinstance(item, ProcessExited):
                self._forget(process)
                return

    def _read_result(self, run: _TurnRun) -> ResultFrame:
        """Pump output until this turn's ``result`` frame, or fail the turn."""
        process = self._live_process()
        while True:
            self._check_deadline(run, process)
            item = self._poll(process)
            if item is None:
                continue
            if isinstance(item, StdoutLine):
                frame = self._on_stdout(item.text, run)
                if frame is not None:
                    return frame
            elif isinstance(item, StderrLine):
                self._note_stderr(item.text)
                run.parser.feed_stderr(item.text)
            else:
                self._forget(process)
                raise self._died_error(run, returncode=item.returncode)

    def _check_deadline(self, run: _TurnRun, process: Process) -> None:
        """Interrupt a turn that ran out of time; kill the process if it will not stop."""
        if run.deadline is None:
            return
        now = self._rt.clock.monotonic()
        if not run.expired:
            if now >= run.deadline:
                run.expired = True
                run.grace_deadline = now + self._rt.interrupt_grace_s
                self._send_interrupt(self._turn_seq)
        elif now >= run.grace_deadline:
            self._forget(process)
            process.kill()
            raise TurnTimeoutError(run.timeout_s or 0.0)

    def _on_stdout(self, text: str, run: _TurnRun) -> ResultFrame | None:
        """Handle one stdout line; return the frame if it ends the turn."""
        data = parse_json_object(text)
        kind = data.get("type") if data is not None else None
        if data is not None and kind == "control_request":
            self._answer(data, run)
            return None
        if data is not None and kind in ("control_response", "control_cancel_request"):
            return None
        if data is not None and kind == "system":
            self._learn_id(data.get("session_id"))
        run.parser.feed_stdout(text)
        if data is not None and kind == "result":
            frame = parse_frame(data)
            if isinstance(frame, ResultFrame):
                return frame
        return None

    def _learn_id(self, session_id: object) -> None:
        """Remember the conversation id as soon as the CLI names it.

        A turn that then fails or times out has still created the conversation,
        and the next process must resume it.
        """
        if isinstance(session_id, str) and session_id and self._id is None:
            self._id = session_id

    def _answer(self, data: Mapping[str, Any], run: _TurnRun) -> None:
        """Answer a request from the CLI at once: it blocks until it gets a reply."""
        request_id = str(data.get("request_id"))
        raw = data.get("request")
        request = cast("dict[str, Any]", raw) if isinstance(raw, dict) else {}
        subtype = request.get("subtype")
        if subtype == "can_use_tool":
            self._deny(request_id, request, run)
        else:
            self._reply(
                {
                    "subtype": "error",
                    "request_id": request_id,
                    "error": f"agentshim does not handle {subtype!r} requests",
                },
            )

    def _deny(self, request_id: str, request: Mapping[str, Any], run: _TurnRun) -> None:
        fail = self._spec.approvals is ApprovalPolicy.FAIL_TURN
        tool = str(request.get("tool_name") or "tool")
        detail = _describe_input(request.get("input"))
        tool_use_id = request.get("tool_use_id")
        if isinstance(tool_use_id, str):
            run.denied.add(tool_use_id)
        run.emit(ApprovalDenied(kind=tool, detail=detail))
        if fail and run.approval_failure is None:
            run.approval_failure = f"permission request for {tool} refused: {detail}"
        self._reply(
            {
                "subtype": "success",
                "request_id": request_id,
                "response": {"behavior": "deny", "message": DENY_MESSAGE, "interrupt": fail},
            },
        )

    def _reply(self, response: dict[str, Any]) -> None:
        with self._lock:
            process = self._process
        if process is None:
            return
        try:
            self._write(process, {"type": "control_response", "response": response})
        except ProcessClosedError:
            return  # the pump sees the exit

    # -- the end of a turn

    def _settle(self, run: _TurnRun, frame: ResultFrame) -> ParsedTurn:
        """Turn a ``result`` frame into the turn's outcome, or the error it stands for."""
        parser = run.parser
        parsed = parser.finish()
        if parser.total_cost_usd is not None:
            self._cost_baseline = parser.total_cost_usd
        self._learn_id(parsed.session_id)
        self._report_denials(run, frame)
        if run.expired:
            raise TurnTimeoutError(run.timeout_s or 0.0)
        if run.approval_failure is not None:
            raise TurnFailedError(run.approval_failure, detail=run.approval_failure)
        if frame.is_error and not _interrupted(frame):
            self._settle_after_error()
            raise self._turn_error(frame, parsed)
        return parsed

    def _report_denials(self, run: _TurnRun, frame: ResultFrame) -> None:
        """Report the tool calls the CLI refused on its own (``--permission-prompts none``)."""
        for denial in frame.permission_denials:
            tool_use_id = denial.get("tool_use_id")
            if isinstance(tool_use_id, str):
                if tool_use_id in run.denied:
                    continue
                run.denied.add(tool_use_id)
            detail = _describe_input(denial.get("tool_input"))
            run.emit(ApprovalDenied(kind=str(denial.get("tool_name") or "tool"), detail=detail))
            if self._spec.approvals is ApprovalPolicy.FAIL_TURN and run.approval_failure is None:
                tool = denial.get("tool_name")
                run.approval_failure = f"permission request for {tool} refused: {detail}"

    def _settle_after_error(self) -> None:
        """Notice a process that exits after an error result (an API error may end it)."""
        with self._lock:
            process = self._process
        if process is None:
            return
        clock = self._rt.clock
        until = clock.monotonic() + 1.0
        while clock.monotonic() < until:
            item = self._poll(process)
            if isinstance(item, ProcessExited):
                self._forget(process)
                return
            if isinstance(item, StderrLine):
                self._note_stderr(item.text)

    def _turn_error(self, frame: ResultFrame, parsed: ParsedTurn) -> TurnFailedError:
        detail = parsed.error or ""
        if self._id is not None and frame.num_turns == 0 and _NO_CONVERSATION in detail:
            return SessionResumeError(self._argv, 1, self._id, detail=detail)
        return TurnFailedError(
            f"claude reported an error ({parsed.error_kind.value}): {detail}",
            kind=parsed.error_kind,
            detail=detail,
        )

    def _died_error(
        self, run: _TurnRun | None, *, returncode: int | None = None, process_gone: str = ""
    ) -> AgentShimError:
        """The error of a process that ended before the turn's ``result``."""
        with self._lock:
            closed = self._closed
        if closed:
            msg = "conversation closed during the turn"
            return TurnCancelledError(msg)
        stderr = "\n".join(self._stderr_tail)
        kind = FailureKind.OTHER
        detail = stderr or process_gone
        if run is not None:
            parsed = run.parser.finish()
            kind = parsed.error_kind
            detail = parsed.error or detail
        code = returncode if returncode is not None else -1
        return CliExitError(self._argv, code, "", stderr, kind=kind, detail=detail)


def _interrupted(frame: ResultFrame) -> bool:
    return (
        frame.is_error
        and frame.terminal_reason is not None
        and frame.terminal_reason.startswith(_ABORTED_PREFIX)
    )


def _control_request(request_id: str, request: Mapping[str, Any]) -> dict[str, Any]:
    return {"type": "control_request", "request_id": request_id, "request": dict(request)}


def _describe_input(value: object) -> str:
    return json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
