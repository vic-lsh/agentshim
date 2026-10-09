"""``FakeProcess``: reactive peers, gates, ordering."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from agentshim import (
    ProcessClosedError,
    ProcessExited,
    SpawnRequest,
    StderrLine,
    StdoutLine,
)
from agentshim.testing import (
    EchoPeer,
    FakeExecutor,
    FakeProcess,
    GateMarker,
    ReplayGates,
    SilentPeer,
)
from hypothesis import given
from hypothesis import strategies as st

if TYPE_CHECKING:
    from collections.abc import Sequence

    from agentshim.testing.process import PeerOutput

_OUTPUTS = st.one_of(
    st.text(max_size=5).map(lambda t: StdoutLine(t + "\n")),
    st.text(max_size=5).map(lambda t: StderrLine(t + "\n")),
)


class ScriptedStartPeer:
    def __init__(self, items: Sequence[PeerOutput]) -> None:
        self._items = items

    def on_start(self) -> Sequence[PeerOutput]:
        return self._items

    def on_stdin(self, data: str) -> Sequence[PeerOutput]:
        del data
        return ()

    def on_stdin_closed(self) -> Sequence[PeerOutput]:
        return ()


class JsonRpcPeer:
    """Answers each request line with a response carrying the same id."""

    def on_start(self) -> Sequence[PeerOutput]:
        return ()

    def on_stdin(self, data: str) -> Sequence[PeerOutput]:
        request = json.loads(data)
        return [StdoutLine(json.dumps({"id": request["id"], "result": "ok"}) + "\n")]

    def on_stdin_closed(self) -> Sequence[PeerOutput]:
        return [ProcessExited(0)]


def _read_all(process: FakeProcess) -> list[object]:
    items: list[object] = []
    while (item := process.next_output(0.0)) is not None:
        items.append(item)
        if isinstance(item, ProcessExited):
            break
    return items


@given(st.lists(_OUTPUTS, max_size=20))
def test_items_come_out_in_the_order_the_peer_produced_them(items: list[PeerOutput]) -> None:
    process = FakeProcess(ScriptedStartPeer([*items, ProcessExited(0)]))
    assert _read_all(process) == [*items, ProcessExited(0)]


@given(st.lists(_OUTPUTS, max_size=10))
def test_nothing_follows_the_exit(items: list[PeerOutput]) -> None:
    process = FakeProcess(ScriptedStartPeer([ProcessExited(3), *items]))
    assert _read_all(process) == [ProcessExited(3)]
    assert process.next_output(None) == ProcessExited(3)


def test_a_peer_can_answer_by_request_id() -> None:
    process = FakeProcess(JsonRpcPeer())
    process.write(json.dumps({"id": 41}) + "\n")
    process.write(json.dumps({"id": 42}) + "\n")
    answers = [json.loads(item.text)["id"] for item in _read_all(process)]  # type: ignore[union-attr]
    assert answers == [41, 42]


def test_an_empty_buffer_is_a_simulated_timeout() -> None:
    assert FakeProcess(SilentPeer()).next_output(5.0) is None


def test_a_closed_gate_holds_back_output_until_opened() -> None:
    gates = ReplayGates()
    gates.close("mid")
    peer = ScriptedStartPeer(
        [StdoutLine("a\n"), GateMarker("mid"), StdoutLine("b\n"), ProcessExited(0)]
    )
    process = FakeProcess(peer, gates=gates)

    assert process.next_output(0.0) == StdoutLine("a\n")
    assert process.next_output(0.0) is None
    assert process.next_output(0.0) is None
    assert not gates.is_open("mid")

    gates.open("mid")
    assert process.next_output(0.0) == StdoutLine("b\n")
    assert process.next_output(0.0) == ProcessExited(0)


def test_gate_markers_are_never_visible_to_the_consumer() -> None:
    process = FakeProcess(ScriptedStartPeer([GateMarker("open-gate"), StdoutLine("x\n")]))
    assert process.next_output(0.0) == StdoutLine("x\n")


def test_a_kill_overrides_a_closed_gate_and_discards_pending_output() -> None:
    gates = ReplayGates()
    gates.close("g")
    process = FakeProcess(ScriptedStartPeer([GateMarker("g"), StdoutLine("never\n")]), gates=gates)
    process.kill()
    assert process.next_output(0.0) == ProcessExited(-9)
    assert process.kill_calls == 1


def test_terminate_and_kill_are_recorded_and_idempotent() -> None:
    process = FakeProcess(SilentPeer())
    process.terminate()
    process.terminate()
    process.kill()
    assert (process.terminate_calls, process.kill_calls) == (2, 1)
    assert process.next_output(0.0) == ProcessExited(-15)
    assert process.wait(0.0) == -15


def test_writes_are_recorded_and_refused_after_exit() -> None:
    process = FakeProcess(EchoPeer())
    process.write("a\n")
    process.close_stdin()
    assert process.writes == ["a\n"]
    with pytest.raises(ProcessClosedError):
        process.write("b\n")


def test_a_natural_exit_is_not_overwritten_by_a_later_kill() -> None:
    process = FakeProcess(ScriptedStartPeer([ProcessExited(0)]))
    process.kill()
    assert process.next_output(0.0) == ProcessExited(0)


def test_the_executor_spawns_through_its_peer_factory_and_records_requests() -> None:
    executor = FakeExecutor([], peers=lambda _request: EchoPeer())
    request = SpawnRequest(argv=["codex", "app-server"], cwd="/w", env={"A": "1"})
    process = executor.spawn(request)
    assert executor.spawns == [request]
    assert executor.processes == [process]


def test_spawning_without_a_peer_factory_is_an_error() -> None:
    with pytest.raises(ValueError, match="peers"):
        FakeExecutor([]).spawn(SpawnRequest(argv=["x"], cwd=None, env={}))


def test_the_executors_gates_reach_every_spawned_process() -> None:
    executor = FakeExecutor(
        [], peers=lambda _r: ScriptedStartPeer([GateMarker("g"), StdoutLine("x\n")])
    )
    executor.gates.close("g")
    process = executor.spawn(SpawnRequest(argv=["x"], cwd=None, env={}))
    assert process.next_output(0.0) is None
    executor.gates.open("g")
    assert process.next_output(0.0) == StdoutLine("x\n")
