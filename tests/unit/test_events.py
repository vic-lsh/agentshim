"""Event handlers and console rendering."""

from __future__ import annotations

import io

from agentshim import (
    AssistantText,
    CompositeEventHandler,
    ConsoleEventHandler,
    EventHandlerBase,
    Lifecycle,
    NullEventHandler,
    ProviderError,
    RawOutput,
    Reasoning,
    RunFinished,
    RunStarted,
    Stderr,
    ToolCall,
    ToolResult,
    compose_event_handlers,
)
from agentshim.testing import RecordingEventHandler


class TestCompose:
    def test_no_handlers_yields_a_null_handler(self) -> None:
        handler = compose_event_handlers(None, ())
        assert isinstance(handler, NullEventHandler)
        handler.on_event(AssistantText("x"))

    def test_single_handler_is_returned_unwrapped(self) -> None:
        recorder = RecordingEventHandler()
        assert compose_event_handlers(recorder, ()) is recorder

    def test_both_spellings_are_combined_in_order(self) -> None:
        first = RecordingEventHandler()
        second = RecordingEventHandler()
        handler = compose_event_handlers(first, [second])
        assert isinstance(handler, CompositeEventHandler)
        handler.on_event(AssistantText("hi"))
        assert first.events == [AssistantText("hi")]
        assert second.events == [AssistantText("hi")]

    def test_composite_fans_out_in_order(self) -> None:
        seen: list[str] = []

        class Marking(EventHandlerBase):
            def __init__(self, tag: str) -> None:
                self.tag = tag

            def on_event(self, event: object) -> None:
                seen.append(self.tag)

        CompositeEventHandler([Marking("a"), Marking("b")]).on_event(AssistantText("x"))
        assert seen == ["a", "b"]


class TestEventHandlerBase:
    def test_base_ignores_everything(self) -> None:
        handler = EventHandlerBase()
        handler.on_event(AssistantText("x"))
        handler.on_event(RunFinished(0))


class TestConsoleEventHandler:
    def _render(self, *events: object, **kwargs: object) -> str:
        stream = io.StringIO()
        handler = ConsoleEventHandler(stream, color=False, **kwargs)  # pyright: ignore[reportArgumentType]
        for event in events:
            handler.on_event(event)  # pyright: ignore[reportArgumentType]
        return stream.getvalue()

    def test_assistant_text_is_prefixed_per_line(self) -> None:
        out = self._render(AssistantText("Hello\n"), AssistantText("World"))
        assert "[agent] Hello" in out
        assert "[agent] World" in out

    def test_run_start_shows_argv_and_a_rule(self) -> None:
        out = self._render(RunStarted(("claude", "-p")))
        assert "claude -p" in out
        assert "=" * 80 in out

    def test_tool_call_and_result_render(self) -> None:
        out = self._render(
            ToolCall("t1", "Bash", {"cmd": "ls"}),
            ToolResult("t1", "Bash", "file.txt", "", None, 0.5),
        )
        assert "[Tool Use] Bash" in out
        assert "[Tool Result] file.txt" in out

    def test_empty_tool_result_says_it_succeeded(self) -> None:
        out = self._render(ToolResult("t1", "Bash", "", "", 0, None))
        assert "Bash ran successfully" in out

    def test_long_tool_args_are_truncated(self) -> None:
        out = self._render(ToolCall("t1", "Bash", {"cmd": "x" * 500}))
        assert "..." in out
        assert len(out) < 400

    def test_stderr_and_error_are_labelled(self) -> None:
        out = self._render(Stderr("boom\n"), ProviderError("bad schema"))
        assert "[stderr] boom" in out
        assert "[error] bad schema" in out

    def test_lifecycle_is_hidden_by_default(self) -> None:
        assert self._render(Lifecycle("turn_started", "1")) == ""
        assert "turn_started" in self._render(Lifecycle("turn_started", "1"), show_lifecycle=True)

    def test_raw_output_is_shown(self) -> None:
        assert "[agent] junk" in self._render(RawOutput("junk"))

    def test_reasoning_is_shown(self) -> None:
        assert "[agent] thinking" in self._render(Reasoning("thinking"))

    def test_color_can_be_enabled(self) -> None:
        stream = io.StringIO()
        ConsoleEventHandler(stream, color=True).on_event(ToolCall("t1", "Bash", None))
        assert "\033[34m" in stream.getvalue()


class TestRecordingEventHandler:
    def test_records_in_order_and_filters_by_type(self) -> None:
        recorder = RecordingEventHandler()
        recorder.on_event(AssistantText("a"))
        recorder.on_event(RunFinished(0))
        assert recorder.events == [AssistantText("a"), RunFinished(0)]
        assert recorder.of_type(RunFinished) == [RunFinished(0)]


def test_events_are_frozen_and_comparable() -> None:
    assert AssistantText("x") == AssistantText("x")
    assert RunStarted(("a",)) != RunStarted(("b",))
