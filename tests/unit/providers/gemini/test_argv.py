"""Gemini CLI argv construction."""

from __future__ import annotations

from agentshim import ArgvContext
from agentshim.providers.gemini import GeminiProvider


def _ctx(**overrides: object) -> ArgvContext:
    base: dict[str, object] = {
        "binary_path": "/usr/local/bin/gemini",
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
        argv = GeminiProvider().build_argv(_ctx())
        assert argv == ["/usr/local/bin/gemini", "--yolo", "--output-format", "stream-json"]

    def test_model_is_appended_when_set(self) -> None:
        argv = GeminiProvider().build_argv(_ctx(model="gemini-2.5-pro"))
        assert argv[argv.index("--model") + 1] == "gemini-2.5-pro"

    def test_model_is_omitted_when_none(self) -> None:
        """There is no deployment-specific default model to fall back on."""
        assert "--model" not in GeminiProvider().build_argv(_ctx())

    def test_the_prompt_never_reaches_argv(self) -> None:
        """The prompt goes on stdin so ``pkill -f`` cannot match the CLI by it."""
        argv = GeminiProvider().build_argv(_ctx(model="m", extra_args=("--foo",)))
        assert all("deploy the app" not in arg for arg in argv)
        assert "--prompt" not in argv
        assert "-p" not in argv

    def test_extra_args_are_appended_verbatim_and_last(self) -> None:
        argv = GeminiProvider().build_argv(_ctx(extra_args=("--include-directories", "/src")))
        assert argv[-2:] == ["--include-directories", "/src"]


class TestResume:
    def test_the_session_id_is_passed_to_resume(self) -> None:
        argv = GeminiProvider().build_argv(_ctx(resume_session_id="abc-123"))
        assert argv[argv.index("--resume") + 1] == "abc-123"

    def test_no_resume_flag_on_a_fresh_turn(self) -> None:
        assert "--resume" not in GeminiProvider().build_argv(_ctx())

    def test_resume_keeps_the_stream_format(self) -> None:
        argv = GeminiProvider().build_argv(_ctx(resume_session_id="abc-123", model="m"))
        assert argv[argv.index("--output-format") + 1] == "stream-json"


class TestUnsupportedOptions:
    def test_a_schema_is_never_put_in_argv(self) -> None:
        argv = GeminiProvider().build_argv(
            _ctx(schema_inline='{"a":1}', schema_path="/schemas/s.json")
        )
        assert argv == ["/usr/local/bin/gemini", "--yolo", "--output-format", "stream-json"]

    def test_reasoning_effort_is_never_put_in_argv(self) -> None:
        assert GeminiProvider().build_argv(_ctx(reasoning_effort="high")) == [
            "/usr/local/bin/gemini",
            "--yolo",
            "--output-format",
            "stream-json",
        ]


class TestMcpArgv:
    def test_config_file_providers_contribute_no_flags(self) -> None:
        argv = GeminiProvider().build_argv(_ctx(mcp_argv=()))
        assert "--mcp-config" not in argv
