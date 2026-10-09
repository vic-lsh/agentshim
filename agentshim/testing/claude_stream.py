"""Scripted stand-ins for a long-lived ``claude --input-format stream-json`` process.

``ClaudeStreamPeers`` is the factory a ``FakeExecutor`` takes as ``peers``: it
builds one ``ClaudeStreamPeer`` per spawned process and keeps what lives
beyond a process, namely which conversations the "CLI" still has (so a resume
of an unknown id is refused exactly like the real one) and the script of turns
shared by all processes. A peer reacts to what the process is sent, the way
the CLI does: it answers ``initialize``, runs a scripted turn per user
message, answers ``interrupt`` and the other control requests, and blocks on
the permission requests it sends until they are answered.

``ClaudeRecordedPeer`` instead replays the stdout of a real recorded run,
turn by turn, so the real transport can be checked against real frames.

Everything is deterministic: no threads, no clock, no randomness.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from agentshim.execution.process import ProcessExited, StderrLine, StdoutLine

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from agentshim.execution.process import SpawnRequest

    from .process import PeerOutput

#: What the real CLI says when asked to resume a conversation it does not have.
NO_CONVERSATION = "No conversation found with session ID: "


@dataclass(frozen=True)
class ClaudeApiError:
    """A turn that ends in an API error, the way Claude Code reports one.

    The CLI writes a synthetic assistant message with an ``error`` kind and
    then a ``result`` with subtype ``success`` and ``is_error`` true.
    ``ends_process`` makes the process exit 1 afterwards (what one-shot mode
    does on purpose; whether a stream-json process survives is unverified).
    """

    status: int = 529
    kind: str = "overloaded"
    text: str = (
        'API Error: 529 {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}'
    )
    ends_process: bool = False


@dataclass(frozen=True)
class ClaudeCrash:
    """The process dies in the middle of a turn, after writing *stderr*."""

    returncode: int = 1
    stderr: str = "claude: fatal error\n"


@dataclass(frozen=True)
class ClaudePeerTurn:
    """One scripted turn: what the CLI does for one user message.

    ``structured_output`` is reported only when the process was started with
    ``--json-schema``, like the real CLI. ``control_requests`` are requests the
    CLI sends mid-turn (``can_use_tool``, ``hook_callback``, ...); the turn
    does not finish until every one is answered. ``permission_denials`` are
    listed in the result, as the CLI does for what it refused on its own.
    ``stall`` makes the turn run until it is interrupted (``ignores_interrupt``
    makes it ignore even that). ``fail_subtype`` ends the turn with an error
    ``result`` of that subtype (``error_max_structured_output_retries``,
    ``error_during_execution``, ...) whose ``errors`` hold ``text``.
    """

    text: str = "ok"
    tool_calls: Sequence[tuple[str, Mapping[str, Any], str]] = ()
    structured_output: object | None = None
    input_tokens: int = 10
    output_tokens: int = 5
    cost_usd: float = 0.01
    skills_invoked: Sequence[str] = ()
    api_error: ClaudeApiError | None = None
    control_requests: Sequence[Mapping[str, Any]] = ()
    permission_denials: Sequence[Mapping[str, Any]] = ()
    stall: bool = False
    ignores_interrupt: bool = False
    fail_subtype: str | None = None
    crash: ClaudeCrash | None = None
    #: The ``rate_limit_info`` of a ``rate_limit_event`` frame written just
    #: before the turn's result, as the real CLI does; ``None`` writes none.
    rate_limit: Mapping[str, Any] | None = None


class ClaudeStreamPeers:
    """Builds the peers of a fake ``claude`` and remembers its conversations.

    *script* holds one entry per user message, consumed across all processes
    in order; once spent, every turn is a plain ``ClaudePeerTurn()``.
    *known_sessions* are conversations the CLI already has (resumable); every
    conversation a peer starts is added. *skills* is the list the ``init``
    frame offers (``None`` offers no list).
    """

    def __init__(
        self,
        script: Sequence[ClaudePeerTurn] = (),
        *,
        known_sessions: Collection[str] = (),
        skills: Sequence[str] | None = None,
    ) -> None:
        """Bind the script and the conversations that already exist."""
        self._script = list(script)
        self.known_sessions: set[str] = set(known_sessions)
        self.skills = None if skills is None else tuple(skills)
        self.peers: list[ClaudeStreamPeer] = []
        self._sessions = 0

    def build(self, request: SpawnRequest) -> ClaudeStreamPeer:
        """The far end of the process *request* spawns (pass as ``FakeExecutor(peers=...)``)."""
        peer = ClaudeStreamPeer(self, list(request.argv))
        self.peers.append(peer)
        return peer

    def next_turn(self) -> ClaudePeerTurn:
        """Take the next scripted turn, or a plain one once the script is spent."""
        return self._script.pop(0) if self._script else ClaudePeerTurn()

    def new_session_id(self) -> str:
        """Name a conversation: ``session-1``, ``session-2``, ..."""
        self._sessions += 1
        return f"session-{self._sessions}"


class ClaudeStreamPeer:
    """One fake ``claude`` process. Records everything it is sent."""

    def __init__(self, world: ClaudeStreamPeers, argv: Sequence[str]) -> None:
        """Read the flags that change behaviour from *argv*."""
        self._world = world
        self.argv = tuple(argv)
        self.resume_id = _flag_value(self.argv, "--resume")
        self.schema = _flag_value(self.argv, "--json-schema")
        #: Every JSON message received on stdin, in order.
        self.received: list[dict[str, Any]] = []
        #: The text of every user message received.
        self.prompts: list[str] = []
        #: The control responses the client sent (answers to CLI requests).
        self.answers: list[dict[str, Any]] = []
        self.session_id: str | None = None
        self._cumulative_cost = 0.0
        self._turn: ClaudePeerTurn | None = None
        self._pending: list[str] = []
        self._request_count = 0
        self._result_index = 0
        self._dead = False

    # -- FakePeer protocol

    def on_start(self) -> Sequence[PeerOutput]:
        """Refuse a resume of a conversation the CLI does not have; otherwise stay quiet."""
        if self.resume_id is None:
            self.session_id = None
            return ()
        if self.resume_id not in self._world.known_sessions:
            self._dead = True
            message = f"{NO_CONVERSATION}{self.resume_id}"
            result = _result(
                self.resume_id,
                subtype="error_during_execution",
                is_error=True,
                num_turns=0,
                errors=[message],
            )
            return [StdoutLine(_line(result)), StderrLine(message + "\n"), ProcessExited(1)]
        self.session_id = self.resume_id
        return ()

    def on_stdin(self, data: str) -> Sequence[PeerOutput]:
        """React to each JSON line in *data*."""
        out: list[PeerOutput] = []
        for raw in data.splitlines():
            if not raw.strip() or self._dead:
                continue
            message = json.loads(raw)
            self.received.append(message)
            out.extend(self._react(message))
        return out

    def on_stdin_closed(self) -> Sequence[PeerOutput]:
        """Exit cleanly at end of input."""
        self._dead = True
        return [ProcessExited(0)]

    # -- reactions

    def _react(self, message: Mapping[str, Any]) -> list[PeerOutput]:
        kind = message.get("type")
        if kind == "control_request":
            return self._on_control_request(message)
        if kind == "control_response":
            return self._on_answer(message)
        if kind == "user":
            return self._on_user(message)
        return []

    def _on_control_request(self, message: Mapping[str, Any]) -> list[PeerOutput]:
        request_id = message.get("request_id")
        subtype = _obj(message.get("request")).get("subtype")
        if subtype == "initialize":
            return [_stdout(_ok(request_id, {"pid": 1, "session_state": "idle"}))]
        if subtype == "interrupt":
            reply = _stdout(_ok(request_id, {"still_queued": []}))
            if self._turn is not None and self._turn.ignores_interrupt:
                return [reply]
            return [reply, *self._abort()]
        error = f"Unsupported control request subtype: {subtype}"
        return [_stdout(_error(request_id, error))]

    def _abort(self) -> list[PeerOutput]:
        """End the running turn the way an interrupt does."""
        if self._turn is None:
            return []
        reason = "aborted_tools" if self._pending else "aborted_streaming"
        turn, self._turn, self._pending = self._turn, None, []
        interrupted = {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "[Request interrupted by user]"}],
            },
        }
        result = self._result_frame(
            turn, subtype="error_during_execution", is_error=True, terminal_reason=reason, text=""
        )
        return [_stdout(interrupted), _stdout(result)]

    def _on_answer(self, message: Mapping[str, Any]) -> list[PeerOutput]:
        response = _obj(message.get("response"))
        self.answers.append(response)
        request_id = response.get("request_id")
        if request_id in self._pending:
            self._pending.remove(request_id)
        body = _obj(response.get("response"))
        if body.get("interrupt") is True:
            return self._abort_after_answer()
        if self._turn is not None and not self._pending:
            return self._finish(self._turn)
        return []

    def _abort_after_answer(self) -> list[PeerOutput]:
        self._pending = ["answered"]  # the turn was waiting on a tool, not on the model
        return self._abort()

    def _on_user(self, message: Mapping[str, Any]) -> list[PeerOutput]:
        content = _obj(message.get("message")).get("content", "")
        self.prompts.append(content if isinstance(content, str) else json.dumps(content))
        turn = self._world.next_turn()
        self._turn = turn
        if self.session_id is None:
            self.session_id = self._world.new_session_id()
        self._world.known_sessions.add(self.session_id)
        out: list[PeerOutput] = [_stdout(self._init_frame())]
        out.extend(_stdout(frame) for frame in self._body_frames(turn))
        if turn.crash is not None:
            self._dead = True
            return [*out, StderrLine(turn.crash.stderr), ProcessExited(turn.crash.returncode)]
        if turn.stall:
            return out
        for request in turn.control_requests:
            self._request_count += 1
            request_id = f"cli-{self._request_count}"
            self._pending.append(request_id)
            out.append(_stdout(_control_request(request_id, request)))
        if self._pending:
            return out
        return [*out, *self._finish(turn)]

    def _finish(self, turn: ClaudePeerTurn) -> list[PeerOutput]:
        out = self._conclude(turn)
        if turn.rate_limit is None:
            return out
        frame = {"type": "rate_limit_event", "rate_limit_info": dict(turn.rate_limit)}
        return [_stdout(frame), *out]

    def _conclude(self, turn: ClaudePeerTurn) -> list[PeerOutput]:
        self._turn = None
        self._cumulative_cost += turn.cost_usd
        if turn.fail_subtype is not None:
            frame = self._result_frame(turn, subtype=turn.fail_subtype, is_error=True, text="")
            frame["errors"] = [turn.text or turn.fail_subtype]
            return [_stdout(frame)]
        if turn.api_error is not None:
            error = turn.api_error
            out: list[PeerOutput] = [
                _stdout(_assistant_error(error)),
                _stdout(
                    self._result_frame(
                        turn,
                        subtype="success",
                        is_error=True,
                        text=error.text,
                        api_error_status=error.status,
                    )
                ),
            ]
            if error.ends_process:
                self._dead = True
                out.append(ProcessExited(1))
            return out
        frame = self._result_frame(turn, subtype="success", is_error=False, text=turn.text)
        if turn.structured_output is not None and self.schema is not None:
            frame["structured_output"] = turn.structured_output
            frame["result"] = json.dumps(turn.structured_output)
        frame["permission_denials"] = [dict(d) for d in turn.permission_denials]
        return [_stdout(frame)]

    # -- frames

    def _init_frame(self) -> dict[str, Any]:
        frame: dict[str, Any] = {
            "type": "system",
            "subtype": "init",
            "session_id": self.session_id,
            "cwd": "/work",
            "model": "fake-model",
        }
        if self._world.skills is not None:
            frame["skills"] = list(self._world.skills)
        return frame

    def _body_frames(self, turn: ClaudePeerTurn) -> list[dict[str, Any]]:
        calls = [
            *(
                ("Skill", {"skill": name}, f"Launching skill: {name}")
                for name in turn.skills_invoked
            ),
            *turn.tool_calls,
        ]
        frames: list[dict[str, Any]] = []
        if turn.structured_output is not None and self.schema is not None:
            calls.append(("StructuredOutput", {"value": turn.structured_output}, "accepted"))
        blocks: list[dict[str, Any]] = [{"type": "text", "text": turn.text}] if turn.text else []
        blocks.extend(
            {"type": "tool_use", "id": f"toolu_{i}", "name": tool, "input": dict(args)}
            for i, (tool, args, _output) in enumerate(calls)
        )
        if blocks:
            usage = {"input_tokens": turn.input_tokens, "output_tokens": turn.output_tokens}
            message = {"role": "assistant", "content": blocks, "usage": usage}
            frames.append({"type": "assistant", "message": message})
        frames.extend(
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": f"toolu_{i}", "content": output}
                    ],
                },
            }
            for i, (_tool, _args, output) in enumerate(calls)
        )
        return frames

    def _result_frame(  # noqa: PLR0913  # the independent fields of a result frame
        self,
        turn: ClaudePeerTurn,
        *,
        subtype: str,
        is_error: bool,
        text: str,
        terminal_reason: str = "completed",
        api_error_status: int | None = None,
    ) -> dict[str, Any]:
        usage = {
            "input_tokens": turn.input_tokens,
            "output_tokens": turn.output_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        frame = _result(
            self.session_id,
            subtype=subtype,
            is_error=is_error,
            num_turns=1,
            text=text,
            terminal_reason=terminal_reason,
            total_cost_usd=self._cumulative_cost,
            usage=usage,
        )
        frame["result_index"] = self._result_index
        self._result_index += 1
        if api_error_status is not None:
            frame["api_error_status"] = api_error_status
        return frame


class ClaudeRecordedPeer:
    """Replays the stdout of a recorded run, one turn per user message.

    The recording is split at its ``result`` frames. The answer to
    ``initialize`` is replayed with the request id the client actually used. A
    recording that begins with a ``result`` is a process that died at start (a
    refused resume): everything, and *stderr*, is emitted at once, then exit 1.
    Control requests other than ``initialize`` are not answered: a recording
    holds the answers it was made with, in place.
    """

    def __init__(self, stdout: str, stderr: str = "") -> None:
        """Split *stdout* (the recorded JSON lines) into its turns."""
        lines = [line for line in stdout.splitlines() if line.strip()]
        self._stderr = stderr
        self._dies_at_start = bool(lines) and json.loads(lines[0]).get("type") == "result"
        self._initialize: dict[str, Any] | None = None
        self._turns: list[list[str]] = []
        self._next = 0
        #: Every JSON message received on stdin.
        self.received: list[dict[str, Any]] = []
        if self._dies_at_start:
            self._turns = [lines]
            return
        current: list[str] = []
        for line in lines:
            data = json.loads(line)
            if self._initialize is None and data.get("type") == "control_response":
                self._initialize = data
                continue
            current.append(line)
            if data.get("type") == "result":
                self._turns.append(current)
                current = []
        if current:
            self._turns.append(current)

    def on_start(self) -> Sequence[PeerOutput]:
        """Replay a recording of a process that died at start."""
        if not self._dies_at_start:
            return ()
        out: list[PeerOutput] = [StdoutLine(line + "\n") for line in self._turns[0]]
        if self._stderr:
            out.append(StderrLine(self._stderr))
        out.append(ProcessExited(1))
        return out

    def on_stdin(self, data: str) -> Sequence[PeerOutput]:
        """Answer ``initialize`` and replay the next recorded turn per user message."""
        out: list[PeerOutput] = []
        for raw in data.splitlines():
            if raw.strip() and not self._dies_at_start:
                out.extend(self._react(json.loads(raw)))
        return out

    def on_stdin_closed(self) -> Sequence[PeerOutput]:
        """Exit cleanly at end of input."""
        return [ProcessExited(0)]

    def _react(self, message: Mapping[str, Any]) -> list[PeerOutput]:
        self.received.append(dict(message))
        kind = message.get("type")
        request = _obj(message.get("request"))
        if (
            kind == "control_request"
            and request.get("subtype") == "initialize"
            and self._initialize is not None
        ):
            reply = json.loads(json.dumps(self._initialize))
            reply["response"]["request_id"] = message.get("request_id")
            return [StdoutLine(_line(reply))]
        if kind == "user" and self._next < len(self._turns):
            lines = self._turns[self._next]
            self._next += 1
            return [StdoutLine(line + "\n") for line in lines]
        return []


# -- wire helpers


def _flag_value(argv: Sequence[str], flag: str) -> str | None:
    """The value of *flag* given as ``--flag value`` or ``--flag=value``."""
    for index, arg in enumerate(argv):
        if arg == flag and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith(flag + "="):
            return arg[len(flag) + 1 :]
    return None


def _obj(value: object) -> dict[str, Any]:
    """*value* when it is a JSON object, else an empty one."""
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def _line(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload) + "\n"


def _stdout(payload: Mapping[str, Any]) -> StdoutLine:
    return StdoutLine(_line(payload))


def _ok(request_id: object, body: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "control_response",
        "response": {"subtype": "success", "request_id": request_id, "response": dict(body)},
    }


def _error(request_id: object, error: str) -> dict[str, Any]:
    return {
        "type": "control_response",
        "response": {"subtype": "error", "request_id": request_id, "error": error},
    }


def _control_request(request_id: str, request: Mapping[str, Any]) -> dict[str, Any]:
    return {"type": "control_request", "request_id": request_id, "request": dict(request)}


def _assistant_error(error: ClaudeApiError) -> dict[str, Any]:
    message = {"role": "assistant", "content": [{"type": "text", "text": error.text}]}
    return {"type": "assistant", "message": message, "error": error.kind}


def _result(  # noqa: PLR0913  # the independent fields of a result frame
    session_id: str | None,
    *,
    subtype: str,
    is_error: bool,
    num_turns: int,
    text: str = "",
    errors: Sequence[str] = (),
    terminal_reason: str | None = None,
    total_cost_usd: float = 0,
    usage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    frame: dict[str, Any] = {
        "type": "result",
        "subtype": subtype,
        "is_error": is_error,
        "result": text,
        "session_id": session_id,
        "num_turns": num_turns,
        "duration_ms": 10,
        "total_cost_usd": total_cost_usd,
        "usage": dict(usage) if usage is not None else {"input_tokens": 0, "output_tokens": 0},
        "permission_denials": [],
    }
    if errors:
        frame["errors"] = list(errors)
    if terminal_reason is not None:
        frame["terminal_reason"] = terminal_reason
    return frame
