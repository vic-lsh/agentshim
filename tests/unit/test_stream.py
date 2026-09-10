"""Line helpers shared by every provider parser."""

from __future__ import annotations

from agentshim import ToolTracker, parse_json_object


class TestParseJsonObject:
    def test_object_is_returned(self) -> None:
        assert parse_json_object('{"a": 1}\n') == {"a": 1}

    def test_blank_line_is_none(self) -> None:
        assert parse_json_object("   \n") is None

    def test_invalid_json_is_none(self) -> None:
        assert parse_json_object("starting claude...\n") is None

    def test_non_object_json_is_none(self) -> None:
        # A bare scalar or array on stdout is provider noise, not an event.
        assert parse_json_object("42\n") is None
        assert parse_json_object("[1, 2]\n") is None
        assert parse_json_object('"hello"\n') is None
        assert parse_json_object("null\n") is None


class TestToolTracker:
    def test_name_resolves_from_the_call(self) -> None:
        tracker = ToolTracker()
        tracker.start("t1", "Bash")
        assert tracker.name("t1") == "Bash"

    def test_unknown_id_falls_back(self) -> None:
        tracker = ToolTracker()
        assert tracker.name("nope") == "Tool"
        assert tracker.name(None) == "Tool"

    def test_duration_is_none_without_a_matching_call(self) -> None:
        tracker = ToolTracker()
        assert tracker.duration("t1") is None
        assert tracker.duration(None) is None

    def test_duration_is_non_negative_after_a_call(self) -> None:
        tracker = ToolTracker()
        tracker.start("t1", "Bash")
        duration = tracker.duration("t1")
        assert duration is not None
        assert duration >= 0
