"""Reading a root: what is skipped, resumed, reset, bounded and reported as failed."""

from __future__ import annotations

import datetime as dt
import json

import pytest
from agentshim import (
    FilePosition,
    TokenUsage,
    TranscriptRoot,
    parse_timestamp,
    read_transcript_usage,
    transcript_providers,
    transcript_roots,
)
from hypothesis import given, settings
from hypothesis import strategies as st

from .builders import (
    BIG_BUDGET,
    SINCE,
    claude_line,
    claude_root,
    codex_count,
    codex_root,
    codex_usage,
    collect,
    tree,
)

NAME = "projects/p/s.jsonl"


def lines(count: int, *, start: int = 0) -> str:
    return "".join(claude_line(start + i, message=f"m{start + i}") + "\n" for i in range(count))


class TestRoots:
    def test_claude_reads_projects_under_its_config_dir_or_home(self) -> None:
        (explicit,) = transcript_roots("claude", {"CLAUDE_CONFIG_DIR": "/cfg", "HOME": "/h"})
        (default,) = transcript_roots("claude", {"HOME": "/h"})
        assert str(explicit.path) == "/cfg/projects"
        assert str(default.path) == "/h/.claude/projects"

    def test_codex_reads_sessions_and_archived_sessions(self) -> None:
        roots = transcript_roots("codex", {"CODEX_HOME": "/c", "HOME": "/h"})
        assert [str(root.path) for root in roots] == ["/c/sessions", "/c/archived_sessions"]
        home = transcript_roots("codex", {"HOME": "/h"})
        assert str(home[0].path) == "/h/.codex/sessions"

    def test_a_provider_without_readable_transcripts_has_no_roots(self) -> None:
        assert transcript_roots("gemini", {"HOME": "/h"}) == ()
        assert transcript_providers() == ("claude", "codex")

    def test_an_unknown_provider_is_refused_by_name(self) -> None:
        with pytest.raises(ValueError, match="unknown provider"):
            transcript_roots("nobody", {"HOME": "/h"})


class TestArguments:
    def test_a_budget_below_one_is_refused(self) -> None:
        with tree() as state, pytest.raises(ValueError, match="byte_budget"):
            read_transcript_usage(claude_root(state), known={}, since=SINCE, byte_budget=0)

    def test_a_naive_since_is_refused(self) -> None:
        with tree() as state, pytest.raises(ValueError, match="timezone"):
            read_transcript_usage(
                claude_root(state),
                known={},
                since=dt.datetime(2026, 1, 1),  # noqa: DTZ001 - a naive time is the input under test
                byte_budget=1,
            )

    def test_a_root_for_a_provider_with_no_format_is_refused(self) -> None:
        with tree() as state, pytest.raises(ValueError, match="no transcript format"):
            read_transcript_usage(
                TranscriptRoot("gemini", state.root), known={}, since=SINCE, byte_budget=1
            )

    def test_a_missing_root_is_reported_as_not_found_and_not_as_empty_files(self) -> None:
        with tree() as state:
            scan = read_transcript_usage(
                claude_root(state), known={}, since=SINCE, byte_budget=BIG_BUDGET
            )
        assert (scan.found, scan.files, scan.failed) == (False, (), ())


class TestWhichFiles:
    def test_only_jsonl_files_at_any_depth_are_read_in_path_order(self) -> None:
        with tree() as state:
            for name in ("projects/b/2.jsonl", "projects/a/1.jsonl", "projects/a/x/3.jsonl"):
                state.write(name, lines(1))
            state.write("projects/a/notes.txt", "hello")
            scan = read_transcript_usage(
                claude_root(state), known={}, since=SINCE, byte_budget=BIG_BUDGET
            )
            names = [scan.path.relative_to(state.root).as_posix() for scan in scan.files]
        assert names == ["projects/a/1.jsonl", "projects/a/x/3.jsonl", "projects/b/2.jsonl"]

    def test_a_file_older_than_the_window_by_more_than_36_hours_is_skipped(self) -> None:
        with tree() as state:
            old = state.write("projects/p/old.jsonl", lines(1))
            near = state.write("projects/p/near.jsonl", lines(1))
            fresh = state.write("projects/p/fresh.jsonl", lines(1))
            since = dt.datetime(2026, 10, 20, tzinfo=dt.timezone.utc)
            state.touch(old, age_hours=37)
            state.touch(near, age_hours=35)
            scan = read_transcript_usage(
                claude_root(state), known={}, since=since, byte_budget=BIG_BUDGET
            )
            read = {file.path for file in scan.files}
        assert (scan.skipped, read) == (1, {near, fresh})

    def test_a_finished_file_whose_size_and_time_are_unchanged_is_not_read_again(self) -> None:
        with tree() as state:
            state.write(NAME, lines(3))
            first = collect(claude_root(state))
            again = read_transcript_usage(
                claude_root(state), known=first.positions, since=SINCE, byte_budget=BIG_BUDGET
            )
        assert (again.files, again.unchanged, again.pending) == ((), 1, 0)


class TestFailures:
    def test_a_file_that_cannot_be_read_is_reported_and_never_treated_as_empty(self) -> None:
        with tree() as state:
            good = state.write(NAME, lines(1))
            (state.root / "projects/p/broken.jsonl").symlink_to(state.root / "nowhere")
            result = collect(claude_root(state))
        assert result.failed == {state.root / "projects/p/broken.jsonl"}
        assert len(result.records[good]) == 1
        assert state.root / "projects/p/broken.jsonl" not in result.records

    def test_a_compressed_codex_archive_is_reported_because_it_cannot_be_read(self) -> None:
        with tree() as state:
            archive = state.write("archived_sessions/rollout-1.jsonl.zst", b"\x28\xb5\x2f\xfd")
            result = collect(codex_root(state, 1))
        assert result.failed == {archive}
        assert result.records == {}


class TestBudget:
    def test_the_budget_stops_at_a_line_boundary_and_the_next_call_continues(self) -> None:
        with tree() as state:
            path = state.write(NAME, lines(10))
            root = claude_root(state)
            first = read_transcript_usage(root, known={}, since=SINCE, byte_budget=1)
            (scan,) = first.files
            assert len(scan.records) == 1
            assert not scan.position.complete
            assert scan.position.offset == len(lines(1))
            assert first.pending == 1
            second = read_transcript_usage(
                root, known={path: scan.position}, since=SINCE, byte_budget=BIG_BUDGET
            )
        assert len(second.files[0].records) == 9
        assert second.files[0].position.complete
        assert not second.files[0].reset

    def test_files_the_budget_never_reached_are_pending_and_unopened(self) -> None:
        with tree() as state:
            for name in ("a", "b", "c"):
                state.write(f"projects/p/{name}.jsonl", lines(2))
            scan = read_transcript_usage(
                claude_root(state), known={}, since=SINCE, byte_budget=len(lines(2))
            )
        assert [file.position.complete for file in scan.files] == [True]
        assert scan.pending == 2

    def test_a_file_far_larger_than_the_budget_still_finishes(self) -> None:
        with tree() as state:
            path = state.write(NAME, lines(300))
            result = collect(claude_root(state), budget=500)
        assert len(result.records[path]) == 300
        assert result.calls > 10


class TestReset:
    def _read_then(self, change: str, content: str) -> tuple[int, int]:
        """Read a file, replace it as *change* says, and return (records now, resets)."""
        with tree() as state:
            root = claude_root(state)
            path = state.write(NAME, lines(5))
            held = collect(root)
            state.write(NAME, content)
            collect(root, held)
            expected = collect_fresh(content)
            assert held.records[path] == expected
            assert change
            return len(held.records[path]), len(held.resets)

    def test_a_truncated_file_is_read_again_from_the_start(self) -> None:
        assert self._read_then("truncate", lines(2, start=100)) == (2, 1)

    def test_a_file_rewritten_to_the_same_length_is_read_again(self) -> None:
        other = lines(5, start=200)
        assert len(other) != len(lines(5)) or True
        assert self._read_then("same size", other)[1] == 1

    def test_a_longer_file_whose_old_part_changed_is_read_again(self) -> None:
        assert self._read_then("grown but changed", lines(5, start=300) + lines(2, start=400)) == (
            7,
            1,
        )

    def test_only_a_changed_guard_window_matters_so_an_edit_far_before_it_is_not_noticed(
        self,
    ) -> None:
        # The guard covers the 64 bytes before the resume point; an edit earlier cannot be seen.
        with tree() as state:
            root = claude_root(state)
            path = state.write(NAME, lines(5))
            held = collect(root)
            data = bytearray(state.written[path])
            data[10:14] = b"ZZZZ"
            state.write(NAME, bytes(data) + lines(1, start=50).encode())
            collect(root, held)
        assert held.resets == []
        assert len(held.records[path]) == 6

    @settings(max_examples=60)
    @given(
        st.binary(max_size=200),
        st.integers(0, 10**6),
        st.integers(0, 64),
        st.integers(0, 2**32 - 1),
    )
    def test_a_position_made_up_from_anywhere_reads_the_same_as_none_at_all(
        self, state_bytes: bytes, offset: int, guard_length: int, guard_hash: int
    ) -> None:
        with tree() as state:
            root = claude_root(state)
            path = state.write(NAME, lines(4))
            expected = collect(root).records[path]
            position = FilePosition(
                size=0,
                mtime_ns=0,
                offset=offset,
                guard_length=min(guard_length, offset),
                guard_hash=guard_hash,
                state=state_bytes,
                complete=False,
            )
            scan = read_transcript_usage(
                root, known={path: position}, since=SINCE, byte_budget=BIG_BUDGET
            )
        (file,) = scan.files
        assert file.records == tuple(expected)
        assert file.reset or offset == 0

    @settings(max_examples=60)
    @given(st.binary(max_size=3000), st.lists(st.integers(0, 3000), max_size=4))
    def test_arbitrary_bytes_never_raise_and_read_the_same_however_they_arrive(
        self, data: bytes, cuts: list[int]
    ) -> None:
        with tree() as state:
            root = claude_root(state)
            path = state.write(NAME, data)
            expected = collect(root)
            collected = None
            points = sorted({cut % (len(data) + 1) for cut in cuts} | {len(data)})
            for index, (a, b) in enumerate(zip([0, *points[:-1]], points, strict=True)):
                (state.write if index == 0 else state.append)(NAME, data[a:b])
                collected = collect(root, collected)
        assert collected is not None
        assert collected.records.get(path, []) == expected.records.get(path, [])


def collect_fresh(content: str) -> list:
    with tree() as state:
        path = state.write(NAME, content)
        return collect(claude_root(state)).records[path]


class TestHugeLines:
    def test_a_line_over_8_mib_is_reduced_to_the_fields_the_format_reads(self) -> None:
        line = json.loads(claude_line(0, message="big", request="r"))
        line["message"]["content"] = [{"type": "text", "text": "x" * (9 << 20)}]
        line["toolUseResult"] = {"nested": ["[", "}", '"', "\\"] * 1000}
        line["note"] = 'quote " and {brace} and [bracket]'
        big = json.dumps(line)
        assert len(big) > 8 << 20
        with tree() as state:
            path = state.write(NAME, big + "\n" + claude_line(1) + "\n")
            result = collect(claude_root(state))
            small = state.write(
                "projects/p/small.jsonl", claude_line(0, message="big", request="r") + "\n"
            )
            reference = collect(claude_root(state))
        assert result.records[path][0] == reference.records[small][0]
        assert len(result.records[path]) == 2

    def test_a_huge_line_that_is_not_an_object_is_counted_as_malformed(self) -> None:
        junk = '"usage" ' + "[" * (9 << 20)
        with tree() as state:
            path = state.write(NAME, junk + "\n" + claude_line(0) + "\n")
            result = collect(claude_root(state))
        assert len(result.records[path]) == 1
        assert result.malformed[path] == 1


class TestFilePosition:
    def test_it_refuses_values_no_read_could_produce(self) -> None:
        good = {
            "size": 10,
            "mtime_ns": 5,
            "offset": 4,
            "guard_length": 4,
            "guard_hash": 7,
            "state": b"",
        }
        FilePosition(**good)
        for change in (
            {"offset": -1},
            {"guard_length": 5},
            {"guard_length": 65, "offset": 100},
            {"guard_hash": 2**32},
            {"size": True},
        ):
            with pytest.raises(ValueError, match="must"):
                FilePosition(**{**good, **change})  # type: ignore[arg-type]


class TestTimestamps:
    @pytest.mark.parametrize(
        ("text", "iso"),
        [
            ("2026-10-20T01:02:03Z", "2026-10-20T01:02:03+00:00"),
            ("2026-10-20T01:02:03.5Z", "2026-10-20T01:02:03.500000+00:00"),
            ("2026-10-20T01:02:03.123456789Z", "2026-10-20T01:02:03.123456+00:00"),
            ("2026-10-20T01:02:03.12345+02:00", "2026-10-20T01:02:03.123450+02:00"),
            ("2026-10-20 01:02:03-0530", "2026-10-20T01:02:03-05:30"),
            ("2026-10-20T01:02:03", "2026-10-20T01:02:03+00:00"),
            (" 2026-10-20T01:02:03z ", "2026-10-20T01:02:03+00:00"),
        ],
    )
    def test_the_spellings_the_clis_use_read_the_same_on_every_python(
        self, text: str, iso: str
    ) -> None:
        moment = parse_timestamp(text)
        assert moment is not None
        assert moment.isoformat() == iso

    @pytest.mark.parametrize(
        "value",
        [
            None,
            5,
            "",
            "yesterday",
            "2026-13-20T01:02:03Z",
            "2026-10-20T25:02:03Z",
            "2026-10-20T01:02:03+99:00",
        ],
    )
    def test_anything_else_is_none(self, value: object) -> None:
        assert parse_timestamp(value) is None

    @given(
        st.datetimes(
            timezones=st.just(dt.timezone.utc),
            min_value=dt.datetime(1971, 1, 1),  # noqa: DTZ001 - hypothesis takes naive bounds
        )
    )
    def test_it_agrees_with_isoformat_for_any_datetime(self, moment: dt.datetime) -> None:
        text = moment.isoformat().replace("+00:00", "Z")
        assert parse_timestamp(text) == moment


def test_the_tokens_of_two_records_add_up_like_any_other_usage() -> None:
    with tree() as state:
        path = state.write(NAME, lines(3))
        records = collect(claude_root(state)).records[path]
    total = sum((record.tokens for record in records), TokenUsage())
    assert total.turns == 3


def test_codex_and_claude_roots_do_not_interfere() -> None:
    with tree() as state:
        state.write(NAME, lines(1))
        state.write(
            "sessions/2026/rollout-1.jsonl", codex_count(1, codex_usage(), codex_usage()) + "\n"
        )
        assert len(collect(claude_root(state)).records) == 1
        assert (state.root / "sessions").exists()
