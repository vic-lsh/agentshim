"""Skill events from recorded provider streams.

Each fixture is real CLI stdout from one short turn in a workspace holding a
single test skill, ``zorp-format``, trimmed (thinking text, rate-limit and
token-estimate frames dropped, the ``skills`` list shortened, the workspace
path replaced by ``/work``). Recorded with Claude Code 2.1.288 (Haiku 4.5)
and Codex CLI 0.156.1 (gpt-6-luna). The ``skill_invoked`` turns asked for a
zorp record; the ``skill_not_invoked`` turns asked for 2 + 3.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from agentshim import (
    AssistantText,
    SkillInvoked,
    SkillsDiscovered,
    SkillSummary,
    SkillTracker,
    ToolCall,
    get_provider,
)
from agentshim.core.events import AgentEvent
from agentshim.providers.claude import ClaudeStreamParser
from agentshim.providers.codex import CodexStreamParser

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
PARSERS = {"claude": ClaudeStreamParser, "codex": CodexStreamParser}


def _replay(provider: str, name: str) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    parser = PARSERS[provider](events.append)
    for line in (FIXTURES / provider / name).read_text().splitlines(keepends=True):
        parser.feed_stdout(line)
    parser.finish()
    return events


def _summary(provider: str, events: list[AgentEvent]) -> SkillSummary:
    tracker = SkillTracker(get_provider(provider).profile)
    for event in events:
        tracker.on_event(event)
    return tracker.summary()


class TestClaude:
    def test_init_lists_the_offered_skills(self) -> None:
        events = _replay("claude", "skill_invoked_haiku.jsonl")
        discovered = [e for e in events if isinstance(e, SkillsDiscovered)]
        assert discovered == [
            SkillsDiscovered(
                ("orchestrate-agents", "zorp-format", "code-review", "anthropic-skills:pdf")
            )
        ]

    def test_the_skill_event_directly_follows_its_tool_call(self) -> None:
        events = _replay("claude", "skill_invoked_haiku.jsonl")
        index = next(i for i, e in enumerate(events) if isinstance(e, SkillInvoked))
        call = events[index - 1]
        assert isinstance(call, ToolCall)
        assert call.tool == "Skill"
        assert events[index] == SkillInvoked(name="zorp-format", tool_id=call.tool_id)
        answer = next(i for i, e in enumerate(events) if isinstance(e, AssistantText))
        assert index < answer

    def test_summary_of_an_invoking_turn(self) -> None:
        summary = _summary("claude", _replay("claude", "skill_invoked_haiku.jsonl"))
        assert summary.invoked == ("zorp-format",)
        assert summary.invocation_count == 1
        assert summary.discovered is not None
        assert "zorp-format" in summary.discovered

    def test_a_turn_that_needs_no_skill_reports_zero_not_unknown(self) -> None:
        summary = _summary("claude", _replay("claude", "skill_not_invoked_haiku.jsonl"))
        assert summary.invocations == ()
        assert summary.invocation_count == 0
        assert summary.discovered is not None


class TestCodex:
    def test_reading_skill_md_is_the_load(self) -> None:
        events = _replay("codex", "skill_invoked_gpt6_luna.jsonl")
        index = next(i for i, e in enumerate(events) if isinstance(e, SkillInvoked))
        call = events[index - 1]
        assert isinstance(call, ToolCall)
        assert events[index] == SkillInvoked(
            name="zorp-format",
            source_path="/work/.agents/skills/zorp-format/SKILL.md",
            tool_id=call.tool_id,
        )
        answer = next(i for i, e in enumerate(events) if isinstance(e, AssistantText))
        assert index < answer

    def test_discovery_is_unknown_and_zero_invocations_is_known(self) -> None:
        summary = _summary("codex", _replay("codex", "skill_not_invoked_gpt6_luna.jsonl"))
        assert summary.discovered is None
        assert summary.invocations == ()


@pytest.mark.parametrize("provider", ["copilot", "gemini", "opencode"])
def test_a_provider_without_a_skill_signal_reports_unknown(provider: str) -> None:
    summary = SkillTracker(get_provider(provider).profile).summary()
    assert summary == SkillSummary(discovered=None, invocations=None)
    assert summary.invoked is None
    assert summary.invocation_count is None
