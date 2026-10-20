"""``TurnResult.context_tokens``: how full the conversation is after a turn.

The figure is what the next resume starts from: the last model call's prompt
(uncached input, cache reads and cache writes) plus what that call wrote. It is
not the turn's summed usage, which counts the prompt once per model call. The
expected numbers below are read off the recorded frames by hand.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from agentshim import CliAgent, TokenUsage
from agentshim.providers.claude import ClaudeStreamParser
from agentshim.providers.codex import CodexStreamParser
from agentshim.providers.copilot import CopilotStreamParser
from agentshim.providers.gemini import GeminiStreamParser
from agentshim.providers.opencode import OpencodeStreamParser
from agentshim.testing import FakeExecutor, FakeRun, scripted_turn
from hypothesis import given
from hypothesis import strategies as st

if TYPE_CHECKING:
    from agentshim.core.events import AgentEvent
    from agentshim.core.provider import ParsedTurn

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
PARSERS = {
    "claude": ClaudeStreamParser,
    "codex": CodexStreamParser,
    "copilot": CopilotStreamParser,
    "gemini": GeminiStreamParser,
    "opencode": OpencodeStreamParser,
}
_ENV = {"PATH": "/usr/bin:/bin", "HOME": "/home/tester"}


def _lines(provider: str, name: str) -> list[str]:
    return (FIXTURES / provider / name).read_text().splitlines(keepends=True)


def replay(provider: str, name: str) -> ParsedTurn:
    events: list[AgentEvent] = []
    parser = PARSERS[provider](events.append)
    for line in _lines(provider, name):
        parser.feed_stdout(line)
    return parser.finish()


@pytest.mark.parametrize(
    ("provider", "name", "expected"),
    [
        # Two model calls: a Read, then the answer. The last call read 21,996
        # cached tokens, wrote 238 to the cache, sent 8 uncached and wrote 3.
        pytest.param(
            "claude", "haiku45_two_model_calls.jsonl", 8 + 21_996 + 238 + 3, id="claude-two-calls"
        ),
        # A Task subagent ran in between; its own (larger) context is not this
        # conversation's. The main thread's last call is the one counted.
        pytest.param(
            "claude", "haiku_task_subagent.jsonl", 5 + 22_154 + 4_375 + 3, id="claude-subagent"
        ),
        pytest.param(
            "claude", "skill_invoked_haiku.jsonl", 9 + 22_726 + 273 + 3, id="claude-skill"
        ),
        # opencode's last step: input, cache reads, cache writes and output.
        # Reasoning is not carried into the next step's prompt.
        pytest.param("opencode", "step_finish.jsonl", 2_140 + 15_529 + 0 + 156, id="opencode"),
    ],
)
def test_a_recorded_turn_reports_its_last_model_calls_context(
    provider: str, name: str, expected: int
) -> None:
    assert replay(provider, name).context_tokens == expected


@pytest.mark.parametrize(
    ("provider", "name"),
    [
        # ``codex exec`` reports only the thread's running total.
        ("codex", "exec_gpt6_luna.jsonl"),
        # Gemini reports one sum over the turn.
        ("gemini", "result_stats.jsonl"),
        # Copilot does not print its per-request usage.
        ("copilot", "session_turn_1.jsonl"),
        # A refused prompt never reached a model: the only frame is synthetic.
        ("claude", "haiku45_prompt_too_long_api_400.jsonl"),
        ("claude", "haiku45_prompt_too_long_blocking_limit.jsonl"),
    ],
)
def test_a_turn_that_reports_no_single_model_call_has_no_context_reading(
    provider: str, name: str
) -> None:
    assert replay(provider, name).context_tokens is None


def test_the_reading_reaches_the_turn_result() -> None:
    agent = CliAgent(
        "claude",
        executor=FakeExecutor(FakeRun(stdout=_lines("claude", "haiku45_two_model_calls.jsonl"))),
        env=dict(_ENV),
    )
    result = agent.start_session().turn("read note.txt")
    assert result.context_tokens == 8 + 21_996 + 238 + 3


@pytest.mark.parametrize("provider", ["claude", "codex", "copilot", "gemini"])
def test_a_scripted_turn_without_per_call_usage_has_no_reading(provider: str) -> None:
    agent = CliAgent(
        provider, executor=FakeExecutor(scripted_turn(provider, text="ok")), env=dict(_ENV)
    )
    assert agent.start_session().turn("hi").context_tokens is None


def test_an_opencode_step_is_one_model_call() -> None:
    usage = TokenUsage(
        input_tokens=1_000,
        cache_read_input_tokens=600,
        cache_write_input_tokens=100,
        output_tokens=50,
        reasoning_output_tokens=20,
    )
    run = scripted_turn("opencode", text="ok", usage=usage)
    agent = CliAgent("opencode", executor=FakeExecutor(run), env=dict(_ENV))
    assert agent.start_session().turn("hi").context_tokens == 1_000 + 50 - 20


# -- a property over Claude streams --------------------------------------------

_counts = st.integers(min_value=0, max_value=2_000_000)


@st.composite
def _assistant_frames(draw: st.DrawFn) -> dict[str, Any]:
    usage = {
        "input_tokens": draw(_counts),
        "cache_read_input_tokens": draw(_counts),
        "cache_creation_input_tokens": draw(_counts),
        "output_tokens": draw(_counts),
    }
    frame: dict[str, Any] = {
        "type": "assistant",
        "message": {
            "id": f"msg_{draw(st.integers(0, 5))}",
            "model": draw(st.sampled_from(["claude-haiku-4-5", "<synthetic>"])),
            "role": "assistant",
            "content": [{"type": "text", "text": "x"}],
            "usage": usage,
        },
        "parent_tool_use_id": draw(st.sampled_from([None, "toolu_sub"])),
    }
    return frame


def _expected(frames: list[dict[str, Any]]) -> int | None:
    expected = None
    for frame in frames:
        message = frame["message"]
        if frame["parent_tool_use_id"] is not None or message["model"] == "<synthetic>":
            continue
        expected = sum(message["usage"].values())
    return expected


@given(st.lists(_assistant_frames(), max_size=12))
def test_claude_counts_the_last_main_thread_model_call(frames: list[dict[str, Any]]) -> None:
    """Subagent calls and the CLI's synthetic error messages never set the reading."""
    events: list[AgentEvent] = []
    parser = ClaudeStreamParser(events.append)
    for frame in frames:
        parser.feed_stdout(json.dumps(frame) + "\n")
    result = {"type": "result", "subtype": "success", "is_error": False, "result": "x"}
    parser.feed_stdout(json.dumps(result) + "\n")
    assert parser.finish().context_tokens == _expected(frames)
