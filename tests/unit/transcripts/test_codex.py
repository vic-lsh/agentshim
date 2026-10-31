"""Codex's session files, read through the public transcript API."""

from __future__ import annotations

import json

import pytest
from agentshim import TokenUsage, TranscriptRecord, UsageSpeed, read_transcript_usage
from hypothesis import given, settings
from hypothesis import strategies as st

from .builders import (
    BIG_BUDGET,
    SINCE,
    add_usage,
    codex_context,
    codex_count,
    codex_meta,
    codex_root,
    codex_tier,
    codex_usage,
    collect,
    stamp,
    tree,
)

NAME = "sessions/2026/10/20/rollout-1-abc.jsonl"


def read(*lines: str, name: str = NAME) -> list[TranscriptRecord]:
    with tree() as state:
        path = state.write(name, "".join(line + "\n" for line in lines))
        return collect(codex_root(state)).records.get(path, [])


def header(model: str = "gpt-6-luna", tier: str | None = None) -> list[str]:
    lines = [codex_meta(0), codex_context(1, model)]
    if tier is not None:
        lines.append(codex_tier(2, tier))
    return lines


class TestRecords:
    def test_a_request_carries_the_model_session_directory_and_tier_from_earlier_lines(
        self,
    ) -> None:
        one = codex_usage(1000, cached=600, write=100, out=50, reasoning=20)
        (record,) = read(*header(tier="priority"), codex_count(5, one, one))
        assert record.model == "gpt-6-luna"
        assert (record.session_id, record.cwd) == ("codex-sess", "/work/project")
        assert record.speed is UsageSpeed.FAST
        assert record.reported_usd is None
        assert record.tokens == TokenUsage(
            input_tokens=1000,
            cache_read_input_tokens=600,
            cache_write_input_tokens=100,
            output_tokens=50,
            reasoning_output_tokens=20,
            turns=1,
        )
        assert record.timestamp.isoformat() == "2026-10-20T00:00:05+00:00"

    def test_a_tier_and_a_model_change_apply_from_their_line_on(self) -> None:
        first = codex_usage(100)
        second = add_usage(first, codex_usage(200))
        third = add_usage(second, codex_usage(300))
        a, b, c = read(
            *header(),
            codex_count(5, first, first),
            codex_tier(6, "priority"),
            codex_count(7, codex_usage(200), second),
            codex_context(8, "gpt-6-sol"),
            codex_tier(9, None),
            codex_count(10, codex_usage(300), third),
        )
        assert [(r.model, r.speed) for r in (a, b, c)] == [
            ("gpt-6-luna", UsageSpeed.STANDARD),
            ("gpt-6-luna", UsageSpeed.FAST),
            ("gpt-6-sol", UsageSpeed.STANDARD),
        ]

    @pytest.mark.parametrize(
        ("tier", "speed"),
        [
            ("priority", UsageSpeed.FAST),
            ("fast", UsageSpeed.FAST),
            ("ultrafast", UsageSpeed.ULTRAFAST),
            ("flex", UsageSpeed.STANDARD),
            (None, UsageSpeed.STANDARD),
        ],
    )
    def test_service_tiers_map_to_speeds(self, tier: str | None, speed: UsageSpeed) -> None:
        usage = codex_usage(10)
        (record,) = read(*header(tier=tier or "none"), codex_count(5, usage, usage))
        assert record.speed is (UsageSpeed.STANDARD if tier is None else speed)

    def test_usage_before_any_model_is_known_is_dropped(self) -> None:
        usage = codex_usage(10)
        assert read(codex_meta(0), codex_count(1, usage, usage)) == []

    def test_a_turn_context_directory_replaces_the_sessions(self) -> None:
        usage = codex_usage(10)
        (record,) = read(
            codex_meta(0), codex_context(1, "m", cwd="/elsewhere"), codex_count(2, usage, usage)
        )
        assert record.cwd == "/elsewhere"

    def test_only_the_first_session_meta_names_the_session(self) -> None:
        usage = codex_usage(10)
        (record,) = read(
            codex_meta(0, session="child"),
            codex_meta(1, session="copied-parent"),
            codex_context(2, "m"),
            codex_count(3, usage, usage),
        )
        assert record.session_id == "child"

    def test_a_file_with_no_session_meta_is_named_by_the_file(self) -> None:
        usage = codex_usage(10)
        (record,) = read(codex_context(0, "m"), codex_count(1, usage, usage))
        assert record.session_id == "rollout-1-abc"

    def test_a_session_may_name_itself_session_id(self) -> None:
        usage = codex_usage(10)
        meta = json.dumps(
            {
                "type": "session_meta",
                "timestamp": stamp(0),
                "payload": {"session_id": "by-other-key"},
            }
        )
        (record,) = read(meta, codex_context(1, "m"), codex_count(2, usage, usage))
        assert record.session_id == "by-other-key"
        assert record.cwd is None


class TestRepeatsAndTotals:
    def test_a_line_written_twice_counts_once(self) -> None:
        usage = codex_usage(100)
        line = codex_count(5, usage, usage)
        assert len(read(*header(), line, line)) == 1

    def test_requests_are_told_apart_by_the_running_totals(self) -> None:
        a = codex_usage(100)
        b = add_usage(a, codex_usage(100))
        records = read(
            *header(),
            codex_count(5, a, a),
            codex_count(6, codex_usage(100), b),
            codex_count(7, codex_usage(100), b),
        )
        assert len(records) == 2

    def test_without_running_totals_a_repeat_of_the_last_request_is_not_a_new_one(self) -> None:
        usage = codex_usage(100)
        other = codex_usage(40)
        lines = [
            codex_count(5, usage),
            codex_count(6, usage),
            codex_count(7, other),
            codex_count(8, usage),
        ]
        assert len(read(*header(), *lines)) == 3

    def test_a_context_size_marker_is_not_a_request(self) -> None:
        zero = codex_usage(0, out=0)
        real = codex_usage(50)
        records = read(*header(), codex_count(5, zero, zero), codex_count(6, real, real))
        assert len(records) == 1

    def test_a_request_whose_total_does_not_add_up_is_counted_as_malformed(self) -> None:
        usage = codex_usage(100)
        bad = {**usage, "total_tokens": 5}
        with tree() as state:
            path = state.write(NAME, "\n".join([*header(), codex_count(5, bad, bad)]) + "\n")
            result = collect(codex_root(state))
        assert result.records[path] == []
        assert result.malformed[path] == 1

    @pytest.mark.parametrize(
        "broken",
        [
            {"input_tokens": 10, "output_tokens": 1, "cached_input_tokens": None},
            {"input_tokens": 10, "output_tokens": 1, "cached_input_tokens": 20},
            {"input_tokens": 10, "output_tokens": 1, "reasoning_output_tokens": 5},
            {"input_tokens": -1, "output_tokens": 1},
            {"input_tokens": 1.5, "output_tokens": 1},
            {"input_tokens": "10", "output_tokens": 1},
            {"output_tokens": 1},
            [],
        ],
    )
    def test_a_usage_that_contradicts_itself_is_malformed_not_billed(self, broken: object) -> None:
        line = json.dumps(
            {
                "type": "event_msg",
                "timestamp": stamp(5),
                "payload": {"type": "token_count", "info": {"last_token_usage": broken}},
            }
        )
        with tree() as state:
            path = state.write(NAME, "\n".join([*header(), line]) + "\n")
            result = collect(codex_root(state))
        assert result.records[path] == []
        assert result.malformed[path] == 1

    def test_running_totals_that_are_not_the_sum_of_what_was_seen_are_malformed(self) -> None:
        a = codex_usage(100)
        skipped = add_usage(add_usage(a, codex_usage(500)), codex_usage(100))
        with tree() as state:
            path = state.write(
                NAME,
                "\n".join(
                    [*header(), codex_count(5, a, a), codex_count(6, codex_usage(100), skipped)]
                )
                + "\n",
            )
            result = collect(codex_root(state))
        assert len(result.records[path]) == 1
        assert result.malformed[path] == 1

    def test_a_whole_number_written_with_a_decimal_point_is_accepted(self) -> None:
        line = json.dumps(
            {
                "type": "event_msg",
                "timestamp": stamp(5),
                "payload": {
                    "type": "token_count",
                    "info": {"last_token_usage": {"input_tokens": 10.0, "output_tokens": 2.0}},
                },
            }
        )
        (record,) = read(*header(), line)
        assert record.tokens.input_tokens == 10

    def test_a_session_resumed_in_a_new_process_restarts_its_totals_at_that_request(self) -> None:
        first = codex_usage(100)
        restarted = codex_usage(70)
        records = read(
            *header(), codex_count(5, first, first), codex_count(6, restarted, restarted)
        )
        assert [r.tokens.input_tokens for r in records] == [100, 70]

    def test_the_same_restart_written_twice_is_a_replay(self) -> None:
        first = codex_usage(100)
        restarted = codex_usage(70)
        replay = codex_count(6, restarted, restarted)
        assert len(read(*header(), codex_count(5, first, first), replay, replay)) == 2


class TestForks:
    def test_history_copied_into_a_fork_in_a_burst_is_not_billed_again(self) -> None:
        parent = codex_usage(100)
        both = add_usage(parent, codex_usage(100))
        child = codex_usage(40)
        lines = [
            codex_meta(0, session="child", forked_from_id="parent"),
            codex_context(0, "m"),
            codex_count(0, parent, parent, ms=1),
            codex_count(0, codex_usage(100), both, ms=20),
            codex_count(5, child, add_usage(both, child)),
        ]
        records = read(*lines)
        assert [r.tokens.input_tokens for r in records] == [40]
        assert records[0].session_id == "child"

    def test_a_subagent_thread_is_a_fork_by_its_spawn_source(self) -> None:
        parent = codex_usage(100)
        source = {"subagent": {"thread_spawn": {"parent_thread_id": "p"}}}
        lines = [
            codex_meta(0, source=source),
            codex_context(0, "m"),
            codex_count(0, parent, parent, ms=5),
            codex_count(9, codex_usage(30), add_usage(parent, codex_usage(30))),
        ]
        assert [r.tokens.input_tokens for r in read(*lines)] == [30]

    def test_a_session_that_is_not_a_fork_bills_a_burst(self) -> None:
        a = codex_usage(100)
        lines = [
            codex_meta(0),
            codex_context(0, "m"),
            codex_count(0, a, a, ms=1),
            codex_count(0, codex_usage(100), add_usage(a, codex_usage(100)), ms=20),
        ]
        assert len(read(*lines)) == 2


class TestRoots:
    def test_sessions_and_archived_sessions_are_each_a_root(self) -> None:
        usage = codex_usage(10)
        with tree() as state:
            live = state.write(
                "sessions/a/rollout-1.jsonl",
                "\n".join([*header(), codex_count(5, usage, usage)]) + "\n",
            )
            old = state.write(
                "archived_sessions/rollout-2.jsonl",
                "\n".join([*header(), codex_count(5, usage, usage)]) + "\n",
            )
            sessions = collect(codex_root(state, 0))
            archived = collect(codex_root(state, 1))
        assert list(sessions.records) == [live]
        assert list(archived.records) == [old]

    def test_a_root_marks_that_one_session_may_appear_in_several_files(self) -> None:
        with tree() as state:
            assert codex_root(state).shared_sessions


@st.composite
def codex_files(draw: st.DrawFn) -> list[str]:
    lines = [codex_meta(0, **({"forked_from_id": "p"} if draw(st.booleans()) else {}))]
    lines.append(codex_context(0, draw(st.sampled_from(["gpt-6-luna", "gpt-6-sol"]))))
    total = codex_usage(0, out=0)
    previous = ""
    for index in range(1, draw(st.integers(1, 16))):
        kind = draw(
            st.sampled_from(
                [
                    "request",
                    "request",
                    "request",
                    "replay",
                    "legacy",
                    "marker",
                    "tier",
                    "model",
                    "junk",
                    "bad",
                ]
            )
        )
        if kind == "request":
            last = codex_usage(draw(st.integers(1, 500)), out=draw(st.integers(0, 50)))
            total = add_usage(total, last)
            previous = codex_count(index, last, total, ms=draw(st.sampled_from([0, 5, 900])))
            lines.append(previous)
        elif kind == "replay" and previous:
            lines.append(previous)
        elif kind == "legacy":
            previous = codex_count(index, codex_usage(draw(st.integers(1, 500))))
            lines.append(previous)
        elif kind == "marker":
            lines.append(codex_count(index, codex_usage(0, out=0), codex_usage(0, out=0)))
        elif kind == "tier":
            lines.append(codex_tier(index, draw(st.sampled_from(["priority", "flex", None]))))
        elif kind == "model":
            lines.append(codex_context(index, draw(st.sampled_from(["gpt-6-luna", "gpt-5.5"]))))
        elif kind == "bad":
            lines.append('{"payload": {"type": "token_count", broken')
        else:
            lines.append(json.dumps({"type": "event_msg", "payload": {"type": "agent_message"}}))
    return lines


def _whole(
    lines: list[str], tail: str
) -> tuple[list[TranscriptRecord], int, tuple[TranscriptRecord, ...]]:
    with tree() as state:
        path = state.write(NAME, "".join(line + "\n" for line in lines) + tail)
        result = collect(codex_root(state))
        return result.records[path], result.malformed[path], result.tails[path]


@settings(max_examples=120)
@given(
    codex_files(),
    st.sampled_from(["", '{"type":"event_msg","payload":{"type":"token_']),
    st.lists(st.integers(0, 100_000), max_size=6),
    st.sampled_from([1, 60, 400, BIG_BUDGET]),
)
def test_a_rollout_that_grows_in_any_pieces_reads_as_the_whole_one_does(
    lines: list[str], tail: str, cuts: list[int], budget: int
) -> None:
    data = ("".join(line + "\n" for line in lines) + tail).encode()
    expected = _whole(lines, tail)
    points = sorted({cut % (len(data) + 1) for cut in cuts} | {len(data)})
    with tree() as state:
        root = codex_root(state)
        path = state.root / NAME
        collected = None
        for index, (a, b) in enumerate(zip([0, *points[:-1]], points, strict=True)):
            (state.write if index == 0 else state.append)(NAME, data[a:b])
            collected = collect(root, collected, budget=budget)
        assert collected is not None
        assert collected.records[path] == expected[0]
        assert collected.malformed[path] == expected[1]
        assert collected.tails[path] == expected[2]
        assert collected.resets == []


@settings(max_examples=60)
@given(codex_files(), st.integers(1, 3000))
def test_any_byte_budget_finishes_and_finds_the_same_records(lines: list[str], budget: int) -> None:
    expected = _whole(lines, "")
    with tree() as state:
        path = state.write(NAME, "".join(line + "\n" for line in lines))
        collected = collect(codex_root(state), budget=budget)
        assert collected.records[path] == expected[0]


@settings(max_examples=40)
@given(
    st.dictionaries(
        st.sampled_from(
            [
                "model",
                "speed",
                "session_id",
                "cwd",
                "last_signature",
                "last_cumulative",
                "sequence_start",
                "malformed",
                "saw_meta",
                "suppressing",
                "anchor_ms",
            ]
        ),
        st.one_of(
            st.none(),
            st.booleans(),
            st.integers(-5, 5),
            st.text(max_size=5),
            st.just({}),
            st.just({"input_tokens": 1}),
        ),
    )
)
def test_a_saved_state_from_nowhere_is_refused_and_the_file_read_whole(
    state_fields: dict[str, object],
) -> None:
    from agentshim import FilePosition  # noqa: PLC0415

    usage = codex_usage(10)
    with tree() as state:
        path = state.write(NAME, "\n".join([*header(), codex_count(5, usage, usage)]) + "\n")
        root = codex_root(state)
        expected = collect(root).records[path]
        position = FilePosition(
            size=0,
            mtime_ns=0,
            offset=0,
            guard_length=0,
            guard_hash=0,
            state=json.dumps(state_fields).encode(),
            complete=False,
        )
        scan = read_transcript_usage(
            root, known={path: position}, since=SINCE, byte_budget=BIG_BUDGET
        )
    assert scan.files[0].records == tuple(expected)
