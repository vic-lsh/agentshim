"""What a Codex app-server conversation lets a caller watch: ids, deltas, idle polling, launch flags."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from agentshim import (
    Agent,
    ApprovalPolicy,
    AssistantText,
    FrameDirection,
    NativePermissions,
    ProviderCapabilityError,
    RawFrame,
    Reasoning,
    TextDelta,
    TextKind,
    TransportKind,
    TurnRequest,
)
from agentshim.testing import (
    CodexScript,
    FakeClock,
    FakeExecutor,
    Leave,
    RecordingEventHandler,
    RunCommand,
    Say,
    SequentialIds,
    Think,
)

if TYPE_CHECKING:
    from agentshim.execution.process import SpawnRequest

ENV = {"PATH": "/usr/bin:/bin", "HOME": "/home/tester"}


class _World:
    def __init__(self, script: CodexScript, **options: object) -> None:
        self.script = script
        self.argvs: list[tuple[str, ...]] = []

        def build(request: SpawnRequest):  # noqa: ANN202 - a test helper
            self.argvs.append(tuple(request.argv))
            return script.peer(request)

        arguments: dict[str, object] = {
            "transport": TransportKind.STREAM,
            "model": "fake-model",
            "executor": FakeExecutor([], peers=build),
            "env": dict(ENV),
            "permissions": NativePermissions.bypass(),
            "approvals": ApprovalPolicy.DENY,
            "clock": FakeClock(),
            "ids": SequentialIds(),
        }
        self.agent = Agent("codex", **arguments)  # type: ignore[arg-type]
        self.options = options

    def session(self, **options: object):  # noqa: ANN202 - a test helper
        return self.agent.session("/work", **options)  # type: ignore[arg-type]


def _run(session, events: RecordingEventHandler, prompt: str = "go"):  # noqa: ANN001, ANN202
    return session.run(
        session.prepare_turn(TurnRequest(prompt=prompt, timeout=60.0)), on_event=events
    )


def test_text_items_are_named_by_the_provider_and_deltas_name_the_same_items() -> None:
    world = _World(CodexScript().turn(Think("pondering"), Say("hello")))
    session = world.session(partial_text=True)
    events = RecordingEventHandler()
    _run(session, events)
    deltas = [e for e in events.events if isinstance(e, TextDelta)]
    done = {e.item_id: e for e in events.events if isinstance(e, (AssistantText, Reasoning))}
    assert {d.item_id for d in deltas} <= set(done)
    assert all(d.item_id is not None for d in deltas)
    by_item: dict[str | None, str] = {}
    for d in deltas:
        by_item[d.item_id] = by_item.get(d.item_id, "") + d.text
    for item_id, streamed in by_item.items():
        assert done[item_id].text.startswith(streamed)
    assert any(d.kind is TextKind.REASONING for d in deltas)
    assert any(d.kind is TextKind.ASSISTANT for d in deltas)


def test_without_the_option_there_are_no_deltas() -> None:
    world = _World(CodexScript().turn(Think("pondering"), Say("hello")))
    session = world.session()
    events = RecordingEventHandler()
    _run(session, events)
    assert not [e for e in events.events if isinstance(e, TextDelta)]


def test_command_output_streams_as_tool_output() -> None:
    world = _World(CodexScript().turn(RunCommand("ls", output="a\nb\n"), Say("ok")))
    session = world.session(partial_text=True)
    events = RecordingEventHandler()
    _run(session, events)
    out = "".join(
        e.text for e in events.events if isinstance(e, TextDelta) and e.kind is TextKind.TOOL_OUTPUT
    )
    assert out == "a\nb\n"


def test_an_item_that_never_completes_leaves_the_turn_finishing_all_the_same() -> None:
    world = _World(CodexScript().turn(Leave("sleep 99"), Say("moving on")))
    session = world.session()
    events = RecordingEventHandler()
    result = _run(session, events)
    assert [e.text for e in events.events if isinstance(e, AssistantText)] == ["moving on"]
    assert result is not None


def test_subagents_are_turned_off_with_the_feature_flag_before_the_other_config() -> None:
    world = _World(CodexScript().turn(Say("hi")))
    _run(world.session(disable_subagents=True), RecordingEventHandler())
    argv = world.argvs[0]
    at = argv.index("--disable")
    assert argv[at : at + 2] == ("--disable", "multi_agent")
    assert argv[at - 1] == "app-server"


def test_subagents_stay_on_by_default() -> None:
    world = _World(CodexScript().turn(Say("hi")))
    _run(world.session(), RecordingEventHandler())
    assert "--disable" not in world.argvs[0]


def test_codex_names_its_own_threads_so_a_chosen_id_is_refused() -> None:
    world = _World(CodexScript().turn(Say("hi")))
    with pytest.raises(ProviderCapabilityError):
        world.session(new_conversation_id="11111111-1111-4111-8111-111111111111").warm()


def test_warm_opens_the_thread_before_the_first_turn_and_the_turn_reuses_it() -> None:
    script = CodexScript().turn(Say("hi"))
    world = _World(script)
    session = world.session()
    thread_id = session.warm()
    assert thread_id
    assert len(world.argvs) == 1
    assert session.warm() == thread_id
    _run(session, RecordingEventHandler())
    assert len(world.argvs) == 1
    assert session.conversation_id == thread_id


def test_every_line_in_both_directions_reaches_the_raw_sink() -> None:
    world = _World(CodexScript().turn(Say("hi")))
    seen: list[RawFrame] = []
    _run(world.session(raw_sink=seen.append), RecordingEventHandler())
    directions = {f.direction for f in seen}
    assert directions == {FrameDirection.TO_PROVIDER, FrameDirection.FROM_PROVIDER}
    assert all(not f.text.endswith("\n") for f in seen)
    assert any(
        '"method":"initialize"' in f.text for f in seen if f.direction is FrameDirection.TO_PROVIDER
    )


def test_polling_between_turns_with_nothing_to_read_keeps_reporting_alive() -> None:
    world = _World(CodexScript().turn(Say("hi")))
    session = world.session()
    session.warm()
    events = RecordingEventHandler()
    assert session.poll_idle(events) is True
    _run(session, RecordingEventHandler())
    assert session.poll_idle(events) is True
    assert events.events == []
    session.close()
    assert session.poll_idle(events) is False
