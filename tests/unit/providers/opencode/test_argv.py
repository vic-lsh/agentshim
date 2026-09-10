"""opencode argv construction."""

from __future__ import annotations

from agentshim import ArgvContext
from agentshim.providers.opencode import OpencodeProvider


def _ctx(**overrides: object) -> ArgvContext:
    base: dict[str, object] = {
        "binary_path": "/usr/local/bin/opencode",
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
        argv = OpencodeProvider().build_argv(_ctx())
        assert argv == ["/usr/local/bin/opencode", "run", "--format", "json", "--thinking"]

    def test_run_is_the_first_word_after_the_binary(self) -> None:
        assert OpencodeProvider().build_argv(_ctx(model="m"))[1] == "run"

    def test_model_is_appended_when_set(self) -> None:
        argv = OpencodeProvider().build_argv(_ctx(model="anthropic/claude-sonnet-4-5"))
        assert argv[argv.index("--model") + 1] == "anthropic/claude-sonnet-4-5"

    def test_model_is_omitted_when_none(self) -> None:
        """No deployment-specific default: an unset model leaves the CLI's own."""
        assert "--model" not in OpencodeProvider().build_argv(_ctx())

    def test_the_prompt_never_reaches_argv(self) -> None:
        """0.5 put a quoted prompt in argv; it belongs on stdin only."""
        argv = OpencodeProvider().build_argv(_ctx(model="m", extra_args=("--share",)))
        assert all("deploy the app" not in arg for arg in argv)
        assert all('"' not in arg for arg in argv)

    def test_extra_args_are_appended_verbatim_and_last(self) -> None:
        argv = OpencodeProvider().build_argv(_ctx(extra_args=("--agent", "build")))
        assert argv[-2:] == ["--agent", "build"]

    def test_thinking_is_requested_so_reasoning_reaches_the_stream(self) -> None:
        assert "--thinking" in OpencodeProvider().build_argv(_ctx())


class TestResume:
    def test_the_session_id_is_passed_to_session(self) -> None:
        argv = OpencodeProvider().build_argv(_ctx(resume_session_id="ses_abc"))
        assert argv[1:4] == ["run", "--session", "ses_abc"]

    def test_no_session_flag_on_a_fresh_turn(self) -> None:
        assert "--session" not in OpencodeProvider().build_argv(_ctx())

    def test_resume_keeps_the_json_format(self) -> None:
        argv = OpencodeProvider().build_argv(_ctx(resume_session_id="ses_abc", model="m"))
        assert argv[argv.index("--format") + 1] == "json"


class TestUnsupportedOptions:
    def test_a_schema_is_never_put_in_argv(self) -> None:
        argv = OpencodeProvider().build_argv(
            _ctx(schema_inline='{"a":1}', schema_path="/schemas/s.json")
        )
        assert argv == ["/usr/local/bin/opencode", "run", "--format", "json", "--thinking"]

    def test_reasoning_effort_is_never_put_in_argv(self) -> None:
        argv = OpencodeProvider().build_argv(_ctx(reasoning_effort="high"))
        assert "--variant" not in argv
        assert "high" not in argv


class TestMcpArgv:
    def test_config_file_providers_contribute_no_flags(self) -> None:
        assert "--mcp" not in OpencodeProvider().build_argv(_ctx(mcp_argv=()))
