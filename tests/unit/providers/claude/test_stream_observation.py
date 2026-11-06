"""What a stream conversation lets a caller watch: item ids, attribution, deltas, the raw pipe.

The frames are written by hand, so each case states the exact stream it depends on.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest
from agentshim import (
    ApprovalPolicy,
    AssistantText,
    ClaudeStreamTransport,
    ConversationSpec,
    FrameDirection,
    IdleConversation,
    NativePermissions,
    OneShotTransport,
    ProviderCapabilityError,
    RawFrame,
    Reasoning,
    TextDelta,
    TextKind,
    ToolCall,
    ToolResult,
    TurnRequest,
)
from agentshim.execution.process import ProcessExited
from agentshim.testing import (
    ClaudeFramePeer,
    FakeClock,
    FakeExecutor,
    RecordingEventHandler,
    ReplayGates,
    SequentialIds,
)
from agentshim.testing.process import GateMarker
from hypothesis import given
from hypothesis import strategies as st

if TYPE_CHECKING:
    from collections.abc import Sequence

    from agentshim import Conversation
    from agentshim.core.events import AgentEvent
    from agentshim.execution.process import SpawnRequest

SESSION = "11111111-1111-4111-8111-111111111111"
INIT: dict[str, Any] = {"type": "system", "subtype": "init", "session_id": SESSION, "model": "m"}


def _result(text: str = "done") -> dict[str, Any]:
    return {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": text,
        "num_turns": 1,
        "session_id": SESSION,
        "usage": {"input_tokens": 3, "output_tokens": 2},
        "total_cost_usd": 0.01,
    }


def _assistant(
    message_id: str, block: dict[str, Any], *, parent: str | None = None
) -> dict[str, Any]:
    return {
        "type": "assistant",
        "message": {"id": message_id, "role": "assistant", "model": "m", "content": [block]},
        "parent_tool_use_id": parent,
        "session_id": SESSION,
    }


def _text(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _partial(event: dict[str, Any], *, parent: str | None = None) -> dict[str, Any]:
    return {
        "type": "stream_event",
        "event": event,
        "parent_tool_use_id": parent,
        "session_id": SESSION,
    }


def _start(message_id: str, *, parent: str | None = None) -> dict[str, Any]:
    return _partial({"type": "message_start", "message": {"id": message_id}}, parent=parent)


def _delta(
    index: int, text: str, *, thinking: bool = False, parent: str | None = None
) -> dict[str, Any]:
    delta = (
        {"type": "thinking_delta", "thinking": text}
        if thinking
        else {"type": "text_delta", "text": text}
    )
    return _partial({"type": "content_block_delta", "index": index, "delta": delta}, parent=parent)


def _spec(**changes: object) -> ConversationSpec:
    base = ConversationSpec(
        cwd="/work",
        model=None,
        permissions=NativePermissions.bypass(),
        approvals=ApprovalPolicy.DENY,
    )
    return replace(base, **changes)  # type: ignore[arg-type]


class _World:
    """A transport over a frame peer, with the argv of every process it spawned."""

    def __init__(self, turns: Sequence[Sequence[Any]]) -> None:
        self.peer = ClaudeFramePeer(turns)
        self.gates = ReplayGates()
        self.argvs: list[tuple[str, ...]] = []

        def build(request: SpawnRequest) -> ClaudeFramePeer:
            self.argvs.append(tuple(request.argv))
            return self.peer

        self.transport = ClaudeStreamTransport(
            executor=FakeExecutor([], peers=build, gates=self.gates),
            env={},
            clock=FakeClock(),
            ids=SequentialIds(),
        )

    def run(
        self, spec: ConversationSpec, prompt: str = "go"
    ) -> tuple[Conversation, list[AgentEvent]]:
        conversation = self.transport.open(spec)
        events = RecordingEventHandler()
        conversation.turn(TurnRequest(prompt=prompt), events.on_event)
        return conversation, events.events


class TestItemIdsAndAttribution:
    def test_each_block_of_a_message_is_named_by_the_message_and_its_index(self) -> None:
        world = _World(
            [
                [
                    INIT,
                    _assistant("msg_1", {"type": "thinking", "thinking": "hmm"}),
                    _assistant("msg_1", _text("hello")),
                    _assistant(
                        "msg_1", {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}}
                    ),
                    _result(),
                ]
            ]
        )
        _, events = world.run(_spec())
        assert [
            (type(e).__name__, getattr(e, "item_id", getattr(e, "tool_id", None)))
            for e in events
            if isinstance(e, (Reasoning, AssistantText, ToolCall))
        ] == [
            ("Reasoning", "msg_1:0"),
            ("AssistantText", "msg_1:1"),
            ("ToolCall", "toolu_1"),
        ]

    def test_a_second_message_starts_its_indexes_again(self) -> None:
        world = _World(
            [[INIT, _assistant("msg_1", _text("a")), _assistant("msg_2", _text("b")), _result()]]
        )
        _, events = world.run(_spec())
        ids = [e.item_id for e in events if isinstance(e, AssistantText)]
        assert ids == ["msg_1:0", "msg_2:0"]

    def test_a_subagents_events_name_the_call_that_started_it(self) -> None:
        world = _World(
            [
                [
                    INIT,
                    _assistant("msg_s", _text("from the subagent"), parent="toolu_task"),
                    {
                        "type": "user",
                        "message": {
                            "role": "user",
                            "content": [
                                {"type": "tool_result", "tool_use_id": "toolu_x", "content": "out"}
                            ],
                        },
                        "parent_tool_use_id": "toolu_task",
                    },
                    _assistant("msg_m", _text("from the lead")),
                    _result(),
                ]
            ]
        )
        _, events = world.run(_spec())
        assert [e.agent for e in events if isinstance(e, AssistantText)] == ["toolu_task", None]
        assert [e.agent for e in events if isinstance(e, ToolResult)] == ["toolu_task"]

    def test_an_event_equals_one_that_says_the_same_without_the_metadata(self) -> None:
        assert AssistantText("x", "id", "agent") == AssistantText("x")
        assert TextDelta("x", TextKind.ASSISTANT, "a") != TextDelta("x", TextKind.ASSISTANT, "b")


class TestPartialText:
    def test_deltas_are_named_like_the_message_that_completes_them(self) -> None:
        world = _World(
            [
                [
                    INIT,
                    _start("msg_1"),
                    _delta(0, "hel", thinking=True),
                    _delta(1, "he"),
                    _delta(1, "llo"),
                    _assistant("msg_1", {"type": "thinking", "thinking": "hel"}),
                    _assistant("msg_1", _text("hello")),
                    _result(),
                ]
            ]
        )
        _, events = world.run(_spec(partial_text=True))
        deltas = [e for e in events if isinstance(e, TextDelta)]
        assert deltas == [
            TextDelta("hel", TextKind.REASONING, "msg_1:0"),
            TextDelta("he", TextKind.ASSISTANT, "msg_1:1"),
            TextDelta("llo", TextKind.ASSISTANT, "msg_1:1"),
        ]
        whole = {e.item_id: e.text for e in events if isinstance(e, (AssistantText, Reasoning))}
        assert whole == {"msg_1:0": "hel", "msg_1:1": "hello"}

    def test_without_the_option_a_partial_frame_is_ignored_and_the_flag_is_not_sent(self) -> None:
        world = _World(
            [[INIT, _start("m"), _delta(0, "x"), _assistant("m", _text("x")), _result()]]
        )
        _, events = world.run(_spec())
        assert not [e for e in events if isinstance(e, TextDelta)]
        assert "--include-partial-messages" not in world.argvs[0]

    def test_the_option_adds_the_flag(self) -> None:
        world = _World([[INIT, _result()]])
        world.run(_spec(partial_text=True))
        assert "--include-partial-messages" in world.argvs[0]

    def test_a_subagent_streaming_beside_the_lead_keeps_its_own_message(self) -> None:
        world = _World(
            [
                [
                    INIT,
                    _start("msg_lead"),
                    _start("msg_sub", parent="toolu_task"),
                    _delta(0, "a", parent="toolu_task"),
                    _delta(0, "b"),
                    _result(),
                ]
            ]
        )
        _, events = world.run(_spec(partial_text=True))
        assert [e for e in events if isinstance(e, TextDelta)] == [
            TextDelta("a", TextKind.ASSISTANT, "msg_sub:0", "toolu_task"),
            TextDelta("b", TextKind.ASSISTANT, "msg_lead:0", None),
        ]

    @given(st.lists(st.text(min_size=1, max_size=8), min_size=1, max_size=12))
    def test_the_deltas_of_a_block_always_add_up_to_the_text_that_completes_it(
        self, chunks: list[str]
    ) -> None:
        text = "".join(chunks)
        world = _World(
            [
                [
                    INIT,
                    _start("msg_p"),
                    *[_delta(0, chunk) for chunk in chunks],
                    _assistant("msg_p", _text(text)),
                    _result(),
                ]
            ]
        )
        _, events = world.run(_spec(partial_text=True))
        streamed = "".join(
            e.text for e in events if isinstance(e, TextDelta) and e.item_id == "msg_p:0"
        )
        done = [e for e in events if isinstance(e, AssistantText)]
        assert streamed == text
        assert [(e.item_id, e.text) for e in done] == [("msg_p:0", text)]


class TestLaunchOptions:
    def test_subagents_are_denied_by_name_and_the_flag_is_followed_by_another_flag(self) -> None:
        world = _World([[INIT, _result()]])
        world.run(_spec(disable_subagents=True))
        argv = world.argvs[0]
        at = argv.index("--disallowedTools")
        denied = argv[at + 1].split(",")
        assert {"Agent", "Task", "SendMessage", "ListAgents"} <= set(denied)
        assert argv[at + 2].startswith("--")

    @pytest.mark.parametrize("subagents", [False, True])
    @pytest.mark.parametrize("scheduling", [False, True])
    def test_each_denial_is_asked_for_alone_and_the_two_share_one_list(
        self, *, subagents: bool, scheduling: bool
    ) -> None:
        world = _World([[INIT, _result()]])
        world.run(_spec(disable_subagents=subagents, disable_self_scheduling=scheduling))
        argv = world.argvs[0]
        if not (subagents or scheduling):
            assert "--disallowedTools" not in argv
            return
        assert argv.count("--disallowedTools") == 1
        at = argv.index("--disallowedTools")
        denied = set(argv[at + 1].split(","))
        assert argv[at + 2].startswith("--")
        assert ({"Agent", "Task"} <= denied) is subagents
        assert ({"ScheduleWakeup", "CronCreate", "CronDelete", "CronList"} <= denied) is scheduling
        assert (denied & {"ScheduleWakeup", "CronCreate"} == set()) is not scheduling

    def test_no_denial_is_sent_by_default(self) -> None:
        world = _World([[INIT, _result()]])
        world.run(_spec())
        assert "--disallowedTools" not in world.argvs[0]

    def test_a_chosen_id_is_a_session_id_until_the_first_prompt_and_a_resume_after(self) -> None:
        world = _World([[INIT, _result()], [_result()]])
        conversation = world.transport.open(_spec(new_conversation_id=SESSION))
        assert conversation.conversation_id == SESSION
        assert f"--session-id={SESSION}" in world.argvs[0]
        assert not any(arg.startswith("--resume") for arg in world.argvs[0])
        first = conversation.turn(TurnRequest(prompt="a"), RecordingEventHandler().on_event)
        assert first.resumed is False
        # A config change restarts the process; the conversation now exists, so it is resumed.
        conversation.turn(
            TurnRequest(prompt="b", extra_args=("--max-turns", "2")),
            RecordingEventHandler().on_event,
        )
        assert f"--resume={SESSION}" in world.argvs[1]
        assert not any(arg.startswith("--session-id") for arg in world.argvs[1])

    def test_the_one_shot_transport_refuses_what_only_a_stream_can_do(self) -> None:
        transport = OneShotTransport("claude", executor=FakeExecutor([]), env={})
        for option in (
            {"partial_text": True},
            {"disable_subagents": True},
            {"disable_self_scheduling": True},
            {"raw_sink": print},
        ):
            with pytest.raises(ProviderCapabilityError):
                transport.open(_spec(**option))


class TestRawFrames:
    def test_every_line_in_either_direction_reaches_the_sink_in_order(self) -> None:
        frames = [INIT, _assistant("m", _text("x")), _result()]
        world = _World([frames])
        seen: list[RawFrame] = []
        world.run(_spec(raw_sink=seen.append), prompt="hello")
        theirs = [f.text for f in seen if f.direction is FrameDirection.FROM_PROVIDER]
        ours = [f.text for f in seen if f.direction is FrameDirection.TO_PROVIDER]
        assert [
            json.loads(t)["type"] for t in theirs if json.loads(t)["type"] != "control_response"
        ] == ["system", "assistant", "result"]
        assert all(not t.endswith("\n") for t in theirs + ours)
        assert json.loads(ours[0])["request"]["subtype"] == "initialize"
        assert json.loads(ours[-1])["message"]["content"] == "hello"
        # The first thing agentshim wrote precedes the first thing the provider answered.
        assert seen[0].direction is FrameDirection.TO_PROVIDER


class TestIdleActivity:
    def _world(self) -> _World:
        wake = [
            INIT,
            _assistant("msg_w", _text("a background task finished")),
            _result("woke"),
        ]
        return _World([[INIT, _result("first"), GateMarker("wake"), *wake]])

    def test_what_the_cli_does_between_turns_is_reported_when_polled(self) -> None:
        world = self._world()
        world.gates.close("wake")
        conversation, _ = world.run(_spec())
        assert isinstance(conversation, IdleConversation)
        idle = RecordingEventHandler()
        assert conversation.poll_idle(idle.on_event) is True
        assert idle.events == []
        world.gates.open("wake")
        assert conversation.poll_idle(idle.on_event) is True
        texts = [e.text for e in idle.events if isinstance(e, AssistantText)]
        assert texts == ["a background task finished"]
        assert conversation.poll_idle(idle.on_event) is True  # nothing is repeated
        assert len([e for e in idle.events if isinstance(e, AssistantText)]) == 1

    def test_a_process_that_exits_between_turns_is_reported_as_gone(self) -> None:
        world = _World([[INIT, _result(), GateMarker("die"), ProcessExited(1)]])
        world.gates.close("die")
        conversation, _ = world.run(_spec())
        assert isinstance(conversation, IdleConversation)
        ignored = RecordingEventHandler()
        assert conversation.poll_idle(ignored.on_event) is True
        world.gates.open("die")
        assert conversation.poll_idle(ignored.on_event) is False

    def test_polling_while_a_turn_runs_reads_nothing(self) -> None:
        world = _World([[INIT, _result()]])
        conversation = world.transport.open(_spec())
        assert isinstance(conversation, IdleConversation)
        seen: list[bool] = []
        nothing = RecordingEventHandler()

        def during(_event: AgentEvent) -> None:
            seen.append(conversation.poll_idle(nothing.on_event))

        conversation.turn(TurnRequest(prompt="go"), during)
        assert seen
        assert all(seen)
        assert nothing.events == []
