"""Skill events reach ``TurnResult.skills`` and the caller's handler alike."""

from __future__ import annotations

import pytest
from agentshim import CliAgent, SkillInvoked, SkillsDiscovered, SkillTracker, get_provider
from agentshim.providers.claude.parser import skill_invocation
from agentshim.providers.codex.skills import skill_reads
from agentshim.testing import FakeExecutor, RecordingEventHandler, scripted_turn


def test_a_claude_turn_summarizes_the_skill_it_loaded() -> None:
    run = scripted_turn(
        "claude",
        text="done",
        session_id="s1",
        tool_calls=[("Skill", {"skill": "zorp-format"}, "Launching skill: zorp-format")],
    )
    recorder = RecordingEventHandler()
    agent = CliAgent("claude", executor=FakeExecutor(run), event_handler=recorder)
    result = agent.run("format apple")
    assert result.skills.invoked == ("zorp-format",)
    assert recorder.of_type(SkillInvoked) == list(result.skills.invocations or ())


def test_a_codex_turn_summarizes_a_skill_md_read() -> None:
    command = "/bin/bash -lc \"sed -n '1,240p' /w/.agents/skills/zorp-format/SKILL.md\""
    run = scripted_turn("codex", text="done", tool_calls=[("execute", {"command": command}, "")])
    result = CliAgent("codex", executor=FakeExecutor(run)).run("format apple")
    assert result.skills.invoked == ("zorp-format",)
    assert result.skills.discovered is None


def test_a_provider_with_no_skill_signal_reports_unknown_on_the_result() -> None:
    run = scripted_turn("gemini", text="done")
    result = CliAgent("gemini", executor=FakeExecutor(run)).run("hi")
    assert result.skills.invocations is None
    assert result.skills.invocation_count is None


def test_an_unknown_provider_still_reports_an_event_it_receives() -> None:
    tracker = SkillTracker(get_provider("gemini").profile)
    tracker.on_event(SkillInvoked("x"))
    assert tracker.summary().invoked == ("x",)


def test_discovery_merges_across_turns_in_first_seen_order() -> None:
    tracker = SkillTracker(get_provider("claude").profile)
    tracker.on_event(SkillsDiscovered(("a", "b")))
    tracker.on_event(SkillsDiscovered(("b", "c")))
    tracker.on_event(SkillInvoked("b"))
    tracker.on_event(SkillInvoked("b"))
    summary = tracker.summary()
    assert summary.discovered == ("a", "b", "c")
    assert summary.invoked == ("b",)
    assert summary.invocation_count == 2


@pytest.mark.parametrize(
    ("tool", "args", "expected"),
    [
        ("Skill", {"skill": "x"}, SkillInvoked("x", tool_id="t")),
        ("Skill", {}, None),
        (
            "Read",
            {"file_path": "/w/.claude/skills/x/SKILL.md"},
            SkillInvoked("x", source_path="/w/.claude/skills/x/SKILL.md", tool_id="t"),
        ),
        ("Read", {"file_path": "/w/.claude/skills/x/references/a.md"}, None),
        ("Bash", {"command": "cat /w/.claude/skills/x/SKILL.md"}, None),
    ],
)
def test_claude_skill_recognition(
    tool: str, args: dict[str, str], expected: SkillInvoked | None
) -> None:
    assert skill_invocation("t", tool, args) == expected


@pytest.mark.parametrize(
    ("command", "names"),
    [
        ("/bin/bash -lc \"sed -n '1,240p' /w/.agents/skills/a/SKILL.md\"", ["a"]),
        ("cat .agents/skills/a/SKILL.md .agents/skills/g/b/SKILL.md", ["a", "b"]),
        ("cat ~/.codex/skills/.system/installer/SKILL.md | head", ["installer"]),
        ("cat myskills/a/SKILL.md", []),
        ("ls skills/a/SKILL.mdx", []),
        ("cat .agents/skills/a/references/x.md", []),
        ("find . -name SKILL.md", []),
    ],
)
def test_codex_skill_md_reads(command: str, names: list[str]) -> None:
    assert [event.name for event in skill_reads(command, None)] == names


def test_a_scripted_claude_turn_offers_skills() -> None:
    run = scripted_turn("claude", text="ok", skills_offered=["a", "b"])
    result = CliAgent("claude", executor=FakeExecutor(run)).run("hi")
    assert result.skills.discovered == ("a", "b")
    assert result.skills.invocations == ()


@pytest.mark.parametrize("provider", ["codex", "copilot", "gemini", "opencode"])
def test_scripting_an_offered_list_fails_where_the_stream_has_none(provider: str) -> None:
    with pytest.raises(ValueError, match="offered skills"):
        scripted_turn(provider, text="ok", skills_offered=["a"])


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_a_scripted_skill_load_is_reported_in_order(provider: str) -> None:
    run = scripted_turn(provider, text="ok", skills_invoked=["a", "b"])
    result = CliAgent(provider, executor=FakeExecutor(run)).run("hi")
    assert [event.name for event in result.skills.invocations or ()] == ["a", "b"]


@pytest.mark.parametrize("provider", ["copilot", "gemini", "opencode"])
def test_scripting_a_skill_load_fails_where_the_stream_has_none(provider: str) -> None:
    with pytest.raises(ValueError, match="skill loads"):
        scripted_turn(provider, text="ok", skills_invoked=["a"])
