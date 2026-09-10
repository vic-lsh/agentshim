"""Copilot CLI argv construction."""

from __future__ import annotations

from agentshim import ArgvContext
from agentshim.providers.copilot import CopilotProvider

BASE_ARGV = [
    "--output-format",
    "json",
    "--stream",
    "off",
    "--allow-all-tools",
    "--allow-all-paths",
    "--allow-all-urls",
]


def _ctx(**overrides: object) -> ArgvContext:
    base: dict[str, object] = {
        "binary_path": "/usr/local/bin/copilot",
        "model": None,
        "env": {},
        "resume_session_id": None,
        "reasoning_effort": None,
        "schema_inline": None,
        "schema_path": None,
        "mcp_argv": (),
        "extra_args": (),
    }
    base.update(overrides)
    return ArgvContext(**base)  # pyright: ignore[reportArgumentType]


class TestBaseArgv:
    def test_required_flags(self) -> None:
        argv = CopilotProvider().build_argv(_ctx())
        assert argv[0] == "/usr/local/bin/copilot"
        assert argv[1:] == BASE_ARGV

    def test_model_is_appended_when_set(self) -> None:
        argv = CopilotProvider().build_argv(_ctx(model="gpt-5"))
        assert argv[argv.index("--model") + 1] == "gpt-5"

    def test_model_is_omitted_when_none(self) -> None:
        assert "--model" not in CopilotProvider().build_argv(_ctx())

    def test_the_prompt_never_reaches_argv(self) -> None:
        """The prompt goes on stdin, so no ``-p`` flag carries it."""
        argv = CopilotProvider().build_argv(_ctx(model="m", extra_args=("--foo",)))
        assert "-p" not in argv
        assert "--prompt" not in argv

    def test_extra_args_are_appended_verbatim_and_last(self) -> None:
        argv = CopilotProvider().build_argv(_ctx(extra_args=("--add-dir", "/srv")))
        assert argv[-2:] == ["--add-dir", "/srv"]


class TestResume:
    def test_resume_takes_the_session_id_as_a_separate_argument(self) -> None:
        argv = CopilotProvider().build_argv(_ctx(resume_session_id="abc-123"))
        assert argv[argv.index("--resume") + 1] == "abc-123"

    def test_no_resume_flag_on_a_fresh_turn(self) -> None:
        assert "--resume" not in CopilotProvider().build_argv(_ctx())


class TestUnsupportedCapabilities:
    def test_reasoning_effort_never_reaches_argv(self) -> None:
        """The profile declares no reasoning support, so the session rejects it first."""
        argv = CopilotProvider().build_argv(_ctx(reasoning_effort="high"))
        assert "--effort" not in argv
        assert "high" not in argv

    def test_a_schema_is_ignored_in_both_shapes(self) -> None:
        argv = CopilotProvider().build_argv(
            _ctx(schema_inline='{"type":"object"}', schema_path="schema.json")
        )
        assert argv[1:] == BASE_ARGV


class TestMcpArgv:
    def test_mcp_flags_are_appended_before_extra_args(self) -> None:
        argv = CopilotProvider().build_argv(
            _ctx(mcp_argv=("--additional-mcp-config", "{}"), extra_args=("--banner",))
        )
        assert argv[-3:] == ["--additional-mcp-config", "{}", "--banner"]

    def test_no_mcp_flags_without_servers(self) -> None:
        assert "--additional-mcp-config" not in CopilotProvider().build_argv(_ctx())
