"""Claude Code's session files, read through the public transcript API."""

from __future__ import annotations

import json

import pytest
from agentshim import TokenUsage, TranscriptRecord, UsageSpeed
from agentshim.providers.claude.transcripts import KEY_WINDOW
from hypothesis import given, settings
from hypothesis import strategies as st

from .builders import (
    BIG_BUDGET,
    claude_line,
    claude_root,
    collect,
    stamp,
    tree,
)

NAME = "projects/-work-project/sess-1.jsonl"


def read(*lines: str, name: str = NAME, newline: str = "\n") -> list[TranscriptRecord]:
    with tree() as state:
        path = state.write(name, "".join(line + newline for line in lines))
        return collect(claude_root(state)).records.get(path, [])


class TestRecords:
    def test_a_line_becomes_one_record_with_the_model_session_and_directory(self) -> None:
        (record,) = read(claude_line(7, input_tokens=3, output=9, cwd="/work/x", session="s-9"))
        assert record.model == "claude-opus-5-5"
        assert (record.session_id, record.cwd, record.speed) == (
            "s-9",
            "/work/x",
            UsageSpeed.STANDARD,
        )
        assert record.timestamp.isoformat() == "2026-10-20T00:00:07.250000+00:00"
        assert record.reported_usd is None

    def test_input_counts_cache_reads_and_writes_as_agentshim_does_everywhere(self) -> None:
        (record,) = read(
            claude_line(0, input_tokens=3, cache_read=100, cache_write=40, one_hour=25, output=9)
        )
        assert record.tokens == TokenUsage(
            input_tokens=143,
            cache_read_input_tokens=100,
            cache_write_input_tokens=40,
            cache_write_1h_input_tokens=25,
            output_tokens=9,
            turns=1,
        )
        assert record.tokens.uncached_input_tokens == 3

    def test_a_one_hour_share_larger_than_the_writes_is_clamped(self) -> None:
        (record,) = read(claude_line(0, cache_write=10, one_hour=99))
        assert record.tokens.cache_write_1h_input_tokens == 10

    def test_fast_speed_is_read_and_anything_else_is_standard(self) -> None:
        fast, other, absent = read(
            claude_line(0, speed="fast"), claude_line(1, speed="warp"), claude_line(2)
        )
        assert (fast.speed, other.speed, absent.speed) == (
            UsageSpeed.FAST,
            UsageSpeed.STANDARD,
            UsageSpeed.STANDARD,
        )

    def test_a_cost_the_file_states_is_not_trusted(self) -> None:
        (record,) = read(claude_line(0, extra={"costUSD": 12.5}))
        assert record.reported_usd is None

    def test_a_negative_or_non_numeric_count_reads_as_zero(self) -> None:
        line = json.loads(claude_line(0))
        line["message"]["usage"].update(
            input_tokens=-5, output_tokens="9", cache_read_input_tokens=None
        )
        (record,) = read(json.dumps(line))
        assert record.tokens == TokenUsage(turns=1)

    def test_unknown_keys_are_ignored(self) -> None:
        line = json.loads(claude_line(0))
        line["brand_new"] = {"x": 1}
        line["message"]["usage"]["future_field"] = 3
        assert len(read(json.dumps(line))) == 1

    def test_lines_that_carry_no_billed_call_are_skipped_without_being_called_malformed(
        self,
    ) -> None:
        with tree() as state:
            synthetic = claude_line(1, model="<synthetic>")
            unnamed = json.loads(claude_line(2))
            del unnamed["message"]["model"]
            user = json.dumps({"type": "user", "timestamp": stamp(3), "message": {"content": "x"}})
            no_usage = json.dumps({"type": "assistant", "message": {"model": "m", "content": []}})
            path = state.write(
                NAME,
                "\n".join([claude_line(0), synthetic, json.dumps(unnamed), user, no_usage]) + "\n",
            )
            result = collect(claude_root(state))
        assert len(result.records[path]) == 1
        assert result.malformed[path] == 0

    def test_lines_with_the_marker_that_are_not_json_or_have_no_time_are_counted(self) -> None:
        no_time = json.loads(claude_line(0))
        del no_time["timestamp"]
        with tree() as state:
            path = state.write(
                NAME,
                "\n".join(['{"usage": 1', json.dumps(no_time), "plain text", claude_line(1)])
                + "\n",
            )
            result = collect(claude_root(state))
        assert len(result.records[path]) == 1
        assert result.malformed[path] == 2

    def test_a_windows_line_ending_is_not_part_of_the_line(self) -> None:
        assert len(read(claude_line(0), claude_line(1), newline="\r\n")) == 2

    def test_non_ascii_text_in_a_line_is_fine(self) -> None:
        line = json.loads(claude_line(0))
        line["message"]["content"][0]["text"] = "héllo 世界 \U0001f600"
        assert len(read(json.dumps(line, ensure_ascii=False))) == 1


class TestRepeatedBlocks:
    def test_every_content_block_of_a_message_repeats_its_usage_and_counts_once(self) -> None:
        same = claude_line(0, message="m1", request="r1")
        records = read(same, same, claude_line(1, message="m1", request="r1"), claude_line(2))
        assert len(records) == 2

    def test_a_message_and_a_request_id_are_each_enough_to_tell_two_messages_apart(self) -> None:
        records = read(
            claude_line(0, message="m1", request="r1"),
            claude_line(1, message="m1", request="r2"),
            claude_line(2, message="m2", request="r1"),
        )
        assert len(records) == 3

    def test_lines_naming_neither_are_all_counted(self) -> None:
        line = json.loads(claude_line(0))
        del line["message"]["id"]
        text = json.dumps(line)
        assert len(read(text, text)) == 2

    def test_a_repeat_after_more_than_the_window_of_other_messages_counts_again(self) -> None:
        first = claude_line(0, message="m0", request="r0")
        others = [
            claude_line(i, message=f"m{i}", request=f"r{i}") for i in range(1, KEY_WINDOW + 1)
        ]
        assert len(read(first, *others, first)) == KEY_WINDOW + 2

    def test_a_repeat_within_the_window_is_dropped_even_across_a_resume(self) -> None:
        with tree() as state:
            root = claude_root(state)
            path = state.write(NAME, claude_line(0, message="m1", request="r1") + "\n")
            result = collect(root)
            state.append(NAME, claude_line(1, message="m1", request="r1") + "\n")
            collect(root, result)
        assert len(result.records[path]) == 1


class TestAdvisors:
    def test_an_advisor_call_is_its_own_record_on_its_own_model(self) -> None:
        advisor = {
            "type": "advisor_message",
            "model": "claude-fable-5-1",
            "input_tokens": 7,
            "cache_read_input_tokens": 100,
            "cache_creation_input_tokens": 0,
            "output_tokens": 30,
        }
        other = {"type": "message", "model": "claude-opus-5-5", "input_tokens": 1_000_000}
        executor, call = read(claude_line(0, speed="fast", iterations=[other, advisor]))
        assert call.model == "claude-fable-5-1"
        assert call.tokens.uncached_input_tokens == 7
        assert call.tokens.cache_read_input_tokens == 100
        assert call.tokens.output_tokens == 30
        assert (call.session_id, call.cwd, call.speed) == (
            executor.session_id,
            executor.cwd,
            UsageSpeed.FAST,
        )
        assert call.tokens.turns == 0
        assert executor.tokens.turns == 1

    @pytest.mark.parametrize("model", [None, "", "<synthetic>", 5])
    def test_an_advisor_call_with_no_usable_model_is_skipped(self, model: object) -> None:
        advisor = {"type": "advisor_message", "model": model, "output_tokens": 3}
        assert len(read(claude_line(0, iterations=[advisor, "junk", 4]))) == 1  # type: ignore[list-item]

    def test_repeats_of_the_message_do_not_repeat_its_advisor_calls(self) -> None:
        advisor = {"type": "advisor_message", "model": "claude-fable-5-1", "output_tokens": 3}
        line = claude_line(0, message="m1", request="r1", iterations=[advisor])
        assert len(read(line, line)) == 2


class TestSubagents:
    def test_a_subagent_file_takes_the_session_of_its_parent_from_the_path(self) -> None:
        sub = "projects/-work-project/parent-session/subagents/agent-1.jsonl"
        (record,) = read(claude_line(0, session="its-own"), name=sub)
        assert record.session_id == "parent-session"

    def test_a_file_whose_lines_name_no_session_is_named_by_the_file(self) -> None:
        (record,) = read(claude_line(0, session=None))
        assert record.session_id == "sess-1"

    def test_a_missing_working_directory_is_none(self) -> None:
        (record,) = read(claude_line(0, cwd=None))
        assert record.cwd is None


def _plain(draw: st.DrawFn, index: int) -> str:
    return claude_line(
        index,
        model=draw(st.sampled_from(["claude-opus-5-5", "claude-haiku-4-5", "claude-sonnet-5-5"])),
        message=f"m{draw(st.integers(0, 4))}",
        request=draw(st.sampled_from([None, "r1", "r2"])),
        input_tokens=draw(st.integers(0, 1000)),
        cache_read=draw(st.integers(0, 1000)),
        cache_write=draw(st.integers(0, 100)),
        output=draw(st.integers(0, 1000)),
        speed=draw(st.sampled_from([None, "fast"])),
    )


@st.composite
def claude_files(draw: st.DrawFn) -> list[str]:
    lines: list[str] = []
    for index in range(draw(st.integers(0, 14))):
        kind = draw(st.sampled_from(["usage", "usage", "usage", "repeat", "user", "bad", "plain"]))
        if kind == "repeat" and lines:
            lines.append(lines[-1])
        elif kind == "user":
            lines.append(json.dumps({"type": "user", "timestamp": stamp(index), "message": {}}))
        elif kind == "bad":
            lines.append('{"usage": truncated')
        elif kind == "plain":
            lines.append("héllo")
        else:
            lines.append(_plain(draw, index))
    return lines


def _whole(
    lines: list[str], tail: str
) -> tuple[list[TranscriptRecord], int, tuple[TranscriptRecord, ...]]:
    with tree() as state:
        path = state.write(NAME, "".join(line + "\n" for line in lines) + tail)
        result = collect(claude_root(state))
        return result.records[path], result.malformed[path], result.tails[path]


def _cut(data: bytes, cuts: list[int]) -> list[bytes]:
    points = sorted({cut % (len(data) + 1) for cut in cuts} | {len(data)})
    return [data[a:b] for a, b in zip([0, *points[:-1]], points, strict=True)]


@settings(max_examples=120)
@given(
    claude_files(),
    st.sampled_from(["", '{"type":"assistant","time']),
    st.lists(st.integers(0, 100_000), max_size=6),
    st.sampled_from([1, 40, 300, BIG_BUDGET]),
)
def test_a_file_that_grows_in_any_pieces_reads_as_the_whole_file_does(
    lines: list[str], tail: str, cuts: list[int], budget: int
) -> None:
    data = ("".join(line + "\n" for line in lines) + tail).encode()
    expected = _whole(lines, tail)
    with tree() as state:
        root = claude_root(state)
        collected = None
        path = state.root / NAME
        for index, piece in enumerate(_cut(data, cuts)):
            if index == 0:
                state.write(NAME, piece)
            else:
                state.append(NAME, piece)
            collected = collect(root, collected, budget=budget)
        assert collected is not None
        assert collected.records[path] == expected[0]
        assert collected.malformed[path] == expected[1]
        assert collected.tails[path] == expected[2]
        assert collected.resets == []


@settings(max_examples=60)
@given(claude_files(), st.integers(1, 2000))
def test_any_byte_budget_finishes_and_finds_the_same_records(lines: list[str], budget: int) -> None:
    expected = _whole(lines, "")
    with tree() as state:
        path = state.write(NAME, "".join(line + "\n" for line in lines))
        collected = collect(claude_root(state), budget=budget)
        assert collected.records[path] == expected[0]
        total = len(state.written[path])
        assert collected.calls <= total + 2


def test_an_unterminated_last_line_is_provisional_until_its_newline_arrives() -> None:
    first, second = claude_line(0, message="a"), claude_line(1, message="b")
    with tree() as state:
        root = claude_root(state)
        path = state.write(NAME, first + "\n" + second)
        before = collect(root)
        assert len(before.records[path]) == 1
        assert len(before.tails[path]) == 1
        assert before.positions[path].offset == len(first) + 1
        state.append(NAME, "\n")
        after = collect(root, before)
        assert len(after.records[path]) == 2
        assert after.tails[path] == ()
        assert after.resets == []
