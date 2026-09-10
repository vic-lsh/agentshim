"""Codex argv construction."""

from __future__ import annotations

from agentshim import ArgvContext
from agentshim.providers.codex import CodexProvider

_BASE_FLAGS = ["--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check", "--json"]


def _ctx(**overrides: object) -> ArgvContext:
    base: dict[str, object] = {
        "binary_path": "/usr/local/bin/codex",
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
        argv = CodexProvider().build_argv(_ctx())
        assert argv[0] == "/usr/local/bin/codex"
        assert argv[1] == "exec"
        assert argv[2:5] == _BASE_FLAGS

    def test_a_fresh_turn_has_no_resume_subcommand_or_sentinel(self) -> None:
        """Plain ``codex exec`` reads stdin by default, so it needs no ``-``."""
        argv = CodexProvider().build_argv(_ctx())
        assert "resume" not in argv
        assert "-" not in argv

    def test_model_is_appended_when_set(self) -> None:
        argv = CodexProvider().build_argv(_ctx(model="gpt-5"))
        assert argv[argv.index("--model") + 1] == "gpt-5"

    def test_model_is_omitted_when_none(self) -> None:
        assert "--model" not in CodexProvider().build_argv(_ctx())

    def test_extra_args_are_appended_verbatim_and_last(self) -> None:
        argv = CodexProvider().build_argv(_ctx(extra_args=("--include-plan-tool",)))
        assert argv[-1] == "--include-plan-tool"


class TestResume:
    def test_the_stdin_sentinel_follows_the_thread_id(self) -> None:
        """Without ``-``, ``codex exec resume`` ignores stdin and hangs."""
        argv = CodexProvider().build_argv(_ctx(resume_session_id="thread-123"))
        assert argv[:5] == ["/usr/local/bin/codex", "exec", "resume", "thread-123", "-"]

    def test_the_base_flags_follow_the_positionals(self) -> None:
        argv = CodexProvider().build_argv(_ctx(resume_session_id="thread-123"))
        assert argv[5:8] == _BASE_FLAGS

    def test_resumed_turns_still_carry_model_and_schema(self) -> None:
        argv = CodexProvider().build_argv(
            _ctx(resume_session_id="thread-123", model="gpt-5", schema_path="/w/s.json")
        )
        assert argv[argv.index("--model") + 1] == "gpt-5"
        assert argv[argv.index("--output-schema") + 1] == "/w/s.json"


class TestShellPathPolicy:
    def test_the_launcher_path_reaches_spawned_commands(self) -> None:
        argv = CodexProvider().build_argv(_ctx(env={"PATH": "/opt/go/bin"}))
        assert argv[argv.index("--config") + 1] == 'shell_environment_policy.set.PATH="/opt/go/bin"'

    def test_a_path_with_a_quote_is_toml_escaped(self) -> None:
        argv = CodexProvider().build_argv(_ctx(env={"PATH": '/opt/go/bin:/path/with"quote'}))
        assert argv[argv.index("--config") + 1] == (
            'shell_environment_policy.set.PATH="/opt/go/bin:/path/with\\"quote"'
        )

    def test_no_policy_flag_without_a_path(self) -> None:
        argv = CodexProvider().build_argv(_ctx(env={}))
        assert not any("shell_environment_policy" in arg for arg in argv)


class TestReasoningEffort:
    def test_effort_is_a_toml_config_override(self) -> None:
        argv = CodexProvider().build_argv(_ctx(reasoning_effort="xhigh"))
        assert 'model_reasoning_effort="xhigh"' in argv
        assert argv[argv.index('model_reasoning_effort="xhigh"') - 1] == "--config"

    def test_effort_is_omitted_when_unset(self) -> None:
        assert not any(
            "model_reasoning_effort" in arg for arg in CodexProvider().build_argv(_ctx())
        )


class TestOutputSchema:
    def test_the_schema_is_passed_as_a_path(self) -> None:
        argv = CodexProvider().build_argv(_ctx(schema_path="/w/.cache/schemas/judge.json"))
        assert argv[argv.index("--output-schema") + 1] == "/w/.cache/schemas/judge.json"

    def test_an_inline_schema_is_ignored_by_a_file_path_provider(self) -> None:
        argv = CodexProvider().build_argv(_ctx(schema_inline='{"type":"object"}'))
        assert "--output-schema" not in argv
        assert '{"type":"object"}' not in argv

    def test_schema_is_omitted_when_unset(self) -> None:
        assert "--output-schema" not in CodexProvider().build_argv(_ctx())


class TestMcpArgv:
    def test_mcp_flags_precede_the_schema_and_extra_args(self) -> None:
        argv = CodexProvider().build_argv(
            _ctx(
                mcp_argv=("--config", 'mcp_servers.x.command="python"'),
                schema_path="/w/s.json",
                extra_args=("--foo",),
            )
        )
        assert argv.index('mcp_servers.x.command="python"') < argv.index("--output-schema")
        assert argv[-1] == "--foo"
