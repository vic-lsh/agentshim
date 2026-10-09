"""``CodexScript``, the fake Codex server, holds a client to the protocol."""

from __future__ import annotations

import json

from agentshim import ProcessExited, StdoutLine
from agentshim.testing import CodexScript, FakeProcess, Say


def _send(process: FakeProcess, **message: object) -> None:
    process.write(json.dumps(message) + "\n")


def _lines(process: FakeProcess) -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    while (item := process.next_output(0)) is not None:
        if isinstance(item, StdoutLine):
            found.append(json.loads(item.text))
    return found


def _handshake(process: FakeProcess) -> None:
    _send(process, id=1, method="initialize", params={"clientInfo": {"name": "t", "version": "0"}})
    _send(process, method="initialized")
    _lines(process)


def test_a_request_before_the_handshake_is_refused_and_recorded() -> None:
    script = CodexScript()
    process = FakeProcess(script.peer())
    _send(process, id=1, method="thread/start", params={})
    (reply,) = _lines(process)
    assert reply["error"]["code"] == -32600  # type: ignore[index]
    assert script.violations == ["thread/start before the handshake finished"]


def test_an_unknown_method_gets_method_not_found() -> None:
    script = CodexScript()
    process = FakeProcess(script.peer())
    _handshake(process)
    _send(process, id=5, method="thread/delete", params={})
    (reply,) = _lines(process)
    assert reply["id"] == 5
    assert reply["error"]["code"] == -32601  # type: ignore[index]
    assert script.violations  # the client did something the protocol subset does not have


def test_a_reply_to_a_request_never_made_is_recorded() -> None:
    script = CodexScript()
    process = FakeProcess(script.peer())
    _handshake(process)
    _send(process, id=99, result={"decision": "accept"})
    assert any("never made" in v for v in script.violations)


def test_threads_outlive_the_process_that_made_them() -> None:
    script = CodexScript()
    first = FakeProcess(script.peer())
    _handshake(first)
    _send(first, id=2, method="thread/start", params={"cwd": "/w"})
    started = [m for m in _lines(first) if m.get("id") == 2]
    thread_id = started[0]["result"]["thread"]["id"]  # type: ignore[index]
    first.close_stdin()
    assert first.next_output(0) is not None
    second = FakeProcess(script.peer())
    _handshake(second)
    _send(second, id=2, method="thread/resume", params={"threadId": thread_id})
    resumed = [m for m in _lines(second) if m.get("id") == 2]
    assert resumed[0]["result"]["thread"]["id"] == thread_id  # type: ignore[index]
    assert script.spawned == 2
    assert script.thread_ids() == [thread_id]


def test_closing_stdin_exits_zero_once() -> None:
    script = CodexScript()
    process = FakeProcess(script.peer())
    process.close_stdin()
    assert process.next_output(0) == ProcessExited(0)
    assert script.closed == 1


def test_queued_turns_run_in_order_and_then_every_turn_says_ok() -> None:
    script = CodexScript()
    script.turn(Say("first")).turn(Say("second"))
    texts: list[str] = []
    process = FakeProcess(script.peer())
    _handshake(process)
    _send(process, id=2, method="thread/start", params={"cwd": "/w"})
    thread = next(m for m in _lines(process) if m.get("id") == 2)["result"]["thread"]["id"]  # type: ignore[index]
    for index in range(3):
        _send(
            process,
            id=10 + index,
            method="turn/start",
            params={
                "threadId": thread,
                "input": [{"type": "text", "text": "x", "text_elements": []}],
            },
        )
        for message in _lines(process):
            item = message.get("params", {}).get("item", {})  # type: ignore[union-attr]
            if message.get("method") == "item/completed" and item.get("type") == "agentMessage":
                texts.append(item["text"])
    assert texts == ["first", "second", "ok"]
