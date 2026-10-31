"""Session files as the CLIs write them: streamed messages, and files that move.

The tests beside this one build idealised input (a repeat is a copy of the line before it, a file
stays where it is). These hold the invariants over ``shapes.py``, which makes the shapes real files
have. Each one failed on 0.24.0, which counted a streamed message at its first line and a moved
session file twice.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from agentshim import UsageSpeed, read_transcript_usage
from hypothesis import given, settings
from hypothesis import strategies as st

from .builders import (
    BIG_BUDGET,
    SINCE,
    Collected,
    claude_line,
    claude_root,
    codex_context,
    codex_count,
    codex_meta,
    codex_root,
    codex_usage,
    collect,
    tree,
)
from .shapes import (
    ADVISOR_MODEL,
    LIVE,
    Moving,
    Totals,
    claude_sessions,
    cut_points,
    file_moves,
    pieces,
    rollout,
    total_of,
)

if TYPE_CHECKING:
    from .shapes import ClaudeSession

SESSION = Path(__file__).resolve().parents[2] / "fixtures" / "claude_session"
NAME = "projects/-work-project/sess-1.jsonl"
# Three requests: output 8 8 152, then 40 40, then 6 6 6 311. The CLI wrote the first line of each
# message when the stream had reached only part of the reply.
FINAL = {
    "claude-opus-5-5": Totals(
        uncached_input=3 + 1 + 5,
        cache_read=15_000 + 15_400 + 15_440,
        cache_write=400 + 0 + 200,
        cache_write_1h=400 + 0 + 200,
        output=152 + 40 + 311,
        turns=3,
    )
}


def _flat(collected: Collected) -> list:
    return [record for records in collected.records.values() for record in records]


class TestAStreamedMessage:
    @pytest.mark.parametrize("budget", [1, 700, BIG_BUDGET])
    def test_a_file_shaped_like_a_real_one_counts_each_message_at_its_finished_output(
        self, budget: int
    ) -> None:
        with tree() as state:
            state.write(NAME, (SESSION / "streamed_message.jsonl").read_bytes())
            records = _flat(collect(claude_root(state), budget=budget))
        assert total_of(records) == FINAL

    def test_the_lines_that_added_nothing_add_no_record(self) -> None:
        with tree() as state:
            state.write(NAME, (SESSION / "streamed_message.jsonl").read_bytes())
            records = _flat(collect(claude_root(state)))
        # 3 first lines, plus the lines that grew: 8 -> 152, 6 -> 311.
        assert [record.tokens.output_tokens for record in records] == [8, 144, 40, 6, 305]
        assert [record.tokens.turns for record in records] == [1, 0, 1, 1, 0]

    @settings(max_examples=80)
    @given(
        session=claude_sessions(),
        data=st.data(),
        budget=st.sampled_from([1, 300, 4096, BIG_BUDGET]),
    )
    def test_the_records_of_a_session_read_in_any_pieces_add_up_to_its_final_usage(
        self, session: ClaudeSession, data: st.DataObject, budget: int
    ) -> None:
        raw = session.text.encode()
        with tree() as state:
            root = claude_root(state)
            collected = None
            for index, piece in enumerate(pieces(raw, data.draw(cut_points(len(raw))))):
                (state.write if index == 0 else state.append)(NAME, piece)
                collected = collect(root, collected, budget=budget)
            whole = collect(root)  # read again from nothing, the file now complete
        assert collected is not None
        assert total_of(_flat(collected)) == session.final()
        assert _flat(collected) == _flat(whole)

    def test_an_advisor_call_that_shows_only_on_the_last_line_is_counted(self) -> None:
        advisor = {"type": "advisor_message", "model": ADVISOR_MODEL, "output_tokens": 30}
        first = claude_line(0, message="m", request="r", output=5)
        last = claude_line(1, message="m", request="r", output=60, iterations=[advisor])
        with tree() as state:
            state.write(NAME, first + "\n" + last + "\n")
            records = _flat(collect(claude_root(state)))
        totals = total_of(records)
        assert totals[ADVISOR_MODEL].output == 30
        assert totals["claude-opus-5-5"].output == 60

    def test_a_position_saved_by_0_24_0_is_refused_and_the_file_read_again(self) -> None:
        # 0.24.0 saved only the keys it had seen, not what it counted for them.
        old = json.dumps({"keys": ["msg_0:"], "malformed": 0}).encode()
        with tree() as state:
            root = claude_root(state)
            path = state.write(NAME, claude_line(0, output=3) + "\n")
            position = collect(root).positions[path]
            state.append(NAME, claude_line(1, message="msg_0", output=90) + "\n")
            scan = read_transcript_usage(
                root,
                known={path: dataclasses.replace(position, state=old)},
                since=SINCE,
                byte_budget=BIG_BUDGET,
            )
        (file,) = scan.files
        assert file.reset
        assert total_of(list(file.records))["claude-opus-5-5"].output == 90

    def test_an_ultrafast_request_is_read_as_ultrafast(self) -> None:
        with tree() as state:
            state.write(NAME, claude_line(0, speed="ultrafast") + "\n")
            (record,) = _flat(collect(claude_root(state)))
        assert record.speed is UsageSpeed.ULTRAFAST


class TestAWorkingDirectoryIsAPersonalPath:
    def test_records_carry_none_unless_the_caller_asks(self) -> None:
        with tree() as state:
            path = state.write(
                NAME, claude_line(0, cwd="/work/x") + "\n" + claude_line(1, cwd="/work/x")
            )
            quiet = collect(claude_root(state))
            asked = collect(claude_root(state), include_cwd=True)
        assert [r.cwd for r in quiet.records[path]] == [None]
        assert [r.cwd for r in quiet.tails[path]] == [None]
        assert [r.cwd for r in asked.records[path]] == ["/work/x"]
        assert [r.cwd for r in asked.tails[path]] == ["/work/x"]

    def test_a_codex_record_carries_none_unless_the_caller_asks(self) -> None:
        usage = codex_usage(10)
        text = "\n".join(
            [codex_meta(0, cwd="/work/y"), codex_context(1, "m"), codex_count(2, usage, usage)]
        )
        with tree() as state:
            path = state.write(LIVE, text + "\n")
            quiet = collect(codex_root(state))
            asked = collect(codex_root(state), include_cwd=True)
        assert [r.cwd for r in quiet.records[path]] == [None]
        assert [r.cwd for r in asked.records[path]] == ["/work/y"]


class TestASessionFileThatMoves:
    def test_a_session_archived_after_it_was_read_is_not_counted_again(self) -> None:
        lines = rollout(3)
        with tree() as state:
            moving = Moving(state, lines)
            moving.append(len(lines))
            held = collect(codex_root(state, 0))
            moving.move()
            collect(codex_root(state, 1), held)
            collect(codex_root(state, 0), held)
        assert sorted(r.tokens.input_tokens for r in _flat(held)) == [100, 101, 102]

    def test_a_line_written_after_the_move_adds_only_itself(self) -> None:
        lines = rollout(3)
        with tree() as state:
            moving = Moving(state, lines)
            moving.append(len(lines) - 1)
            held = collect(codex_root(state, 0))
            moving.move()
            moving.append(1)
            collect(codex_root(state, 1), held)
        assert sorted(r.tokens.input_tokens for r in _flat(held)) == [100, 101, 102]

    def test_a_copy_left_behind_that_is_older_than_what_was_read_holds_nothing_new(self) -> None:
        lines = rollout(4)
        with tree() as state:
            moving = Moving(state, lines)
            moving.append(len(lines) - 2)
            moving.copy_aside()
            moving.append(2)
            held = collect(codex_root(state, 0))
            collect(codex_root(state, 1), held)
        assert sorted(r.tokens.input_tokens for r in _flat(held)) == [100, 101, 102, 103]

    @settings(max_examples=120)
    @given(steps=file_moves(), requests=st.integers(min_value=1, max_value=10))
    def test_a_session_file_that_grows_moves_and_is_copied_is_counted_once(
        self, steps: list, requests: int
    ) -> None:
        lines = rollout(requests)
        with tree() as state:
            moving = Moving(state, lines)
            held = Collected()
            for step in steps:
                if step.kind == "append":
                    moving.append(step.amount)
                elif step.kind == "read":
                    collect(codex_root(state, step.amount), held)
                elif step.kind == "move":
                    moving.move()
                elif step.kind == "copy":
                    moving.copy_aside()
                else:
                    moving.drop_copy()
            moving.append(len(lines))
            for _ in range(2):
                for root in (0, 1):
                    collect(codex_root(state, root), held)
        assert sorted(r.tokens.input_tokens for r in _flat(held)) == [
            100 + number for number in range(requests)
        ]
        assert held.resets == []
