"""Claude Code argv construction."""

from __future__ import annotations

import json

from agentshim import ArgvContext
from agentshim.providers.claude import ClaudeProvider, SandboxConfig


def _ctx(**overrides: object) -> ArgvContext:
    base: dict[str, object] = {
        "binary_path": "/usr/local/bin/claude",
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
        argv = ClaudeProvider().build_argv(_ctx())
        assert argv[0] == "/usr/local/bin/claude"
        assert argv[1:6] == ["-p", "--dangerously-skip-permissions", "--output-format", "stream-json", "--verbose"]

    def test_model_is_appended_when_set(self) -> None:
        argv = ClaudeProvider().build_argv(_ctx(model="claude-opus-4"))
        assert argv[argv.index("--model") + 1] == "claude-opus-4"

    def test_model_is_omitted_when_none(self) -> None:
        assert "--model" not in ClaudeProvider().build_argv(_ctx())

    def test_the_prompt_never_reaches_argv(self) -> None:
        """The prompt goes on stdin so ``pkill -f`` cannot match the CLI by it."""
        argv = ClaudeProvider().build_argv(_ctx(model="m", extra_args=("--foo",)))
        assert all("deploy the app" not in arg for arg in argv)

    def test_extra_args_are_appended_verbatim_and_last(self) -> None:
        argv = ClaudeProvider().build_argv(_ctx(extra_args=("--append-system-prompt", "be brief")))
        assert argv[-2:] == ["--append-system-prompt", "be brief"]


class TestResume:
    def test_resume_precedes_print_mode(self) -> None:
        argv = ClaudeProvider().build_argv(_ctx(resume_session_id="abc-123"))
        assert argv[1:3] == ["--resume", "abc-123"]
        assert argv.index("--resume") < argv.index("-p")

    def test_no_resume_flag_on_a_fresh_turn(self) -> None:
        assert "--resume" not in ClaudeProvider().build_argv(_ctx())


class TestReasoningEffort:
    def test_effort_flag(self) -> None:
        argv = ClaudeProvider().build_argv(_ctx(reasoning_effort="high"))
        assert argv[argv.index("--effort") + 1] == "high"

    def test_effort_is_omitted_when_unset(self) -> None:
        assert "--effort" not in ClaudeProvider().build_argv(_ctx())


class TestOutputSchema:
    def test_schema_is_inlined(self) -> None:
        inline = '{"type":"object","properties":{"a":{"type":"integer"}}}'
        argv = ClaudeProvider().build_argv(_ctx(schema_inline=inline))
        assert argv[argv.index("--json-schema") + 1] == inline
        assert "stream-json" in argv

    def test_a_schema_path_is_ignored_by_an_inline_provider(self) -> None:
        argv = ClaudeProvider().build_argv(_ctx(schema_path="/tmp/schema.json"))
        assert "--json-schema" not in argv
        assert "/tmp/schema.json" not in argv

    def test_schema_is_omitted_when_unset(self) -> None:
        assert "--json-schema" not in ClaudeProvider().build_argv(_ctx())


class TestSandboxOption:
    def test_no_settings_flag_without_a_sandbox(self) -> None:
        assert "--settings" not in ClaudeProvider().build_argv(_ctx())

    def test_settings_flag_carries_the_sandbox_block(self) -> None:
        provider = ClaudeProvider(sandbox=True)
        argv = provider.build_argv(_ctx())
        settings = json.loads(argv[argv.index("--settings") + 1])
        assert settings["sandbox"]["enabled"] is True

    def test_sandbox_config_is_honoured(self) -> None:
        provider = ClaudeProvider(sandbox=SandboxConfig(allowed_domains=["github.com"]))
        argv = provider.build_argv(_ctx())
        settings = json.loads(argv[argv.index("--settings") + 1])
        assert settings["sandbox"]["network"] == {"allowedDomains": ["github.com"]}

    def test_sandbox_env_is_only_set_with_a_sandbox(self) -> None:
        assert ClaudeProvider().sandbox_env == {}
        assert ClaudeProvider(sandbox=True).sandbox_env == {"CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR": "1"}


class TestMcpArgv:
    def test_config_file_providers_contribute_no_flags(self) -> None:
        argv = ClaudeProvider().build_argv(_ctx(mcp_argv=()))
        assert "--mcp-config" not in argv
