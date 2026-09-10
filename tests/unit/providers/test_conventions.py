"""Conventions every provider package must honour, checked across all of them.

The five provider packages were written in parallel, so the per-provider
suites cannot catch a rule that one of them silently drops. Every test here
is parametrized over ``provider_names()``: adding a provider without meeting
the convention fails here rather than in a consumer's integration.
"""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, Any

import pytest
from agentshim import (
    ArgvContext,
    CliAgent,
    McpMechanism,
    OutputSchemaStyle,
    RawOutput,
    SchemaDialect,
    StdioMcpServer,
    TokenUsage,
    UsageReport,
    get_provider,
    provider_names,
)
from agentshim.providers import get_scripted_lines
from agentshim.testing import FakeExecutor, RecordingEventHandler, scripted_turn

if TYPE_CHECKING:
    from pathlib import Path

    from agentshim.core.events import AgentEvent

PROVIDERS = provider_names()
_ENV = {"PATH": "/usr/bin"}
_PROMPT = "deploy the app right now"
_SESSION_ID = "session-under-test"

#: The flag each provider's argv must carry when a turn resumes.
RESUME_FLAGS = {
    "claude": "--resume",
    "codex": "resume",
    "copilot": "--resume",
    "gemini": "--resume",
    "opencode": "--session",
}

#: Fields a profile may legitimately leave empty.
_MAY_BE_EMPTY = frozenset({"darwin_state_dirs", "auth_env_vars"})

#: Fields a profile may legitimately leave unset: not every provider's CLI
#: documents a state-relocation variable, and only CONFIG_FILE providers
#: write an MCP config file.
_MAY_BE_NONE = frozenset({"schema_dialect", "state_root_env", "mcp_config_file"})


def _argv(name: str, *, resume_session_id: str | None = None) -> list[str]:
    provider = get_provider(name)
    return provider.build_argv(
        ArgvContext(
            binary_path=f"/usr/local/bin/{provider.profile.binary}",
            model=None,
            env=dict(_ENV),
            resume_session_id=resume_session_id,
            reasoning_effort=None,
            schema_inline=None,
            schema_path=None,
            mcp_argv=(),
            extra_args=(),
        )
    )


def _usage() -> TokenUsage:
    return TokenUsage(input_tokens=150, output_tokens=40, cached_input_tokens=50, turns=1)


def _replay(name: str, lines: list[str]) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    parser = get_provider(name).new_parser(events.append, expect_structured=False)
    for line in lines:
        parser.feed_stdout(line)
    parser.finish()
    return events


@pytest.mark.parametrize("name", PROVIDERS)
class TestProfile:
    def test_every_field_is_populated(self, name: str) -> None:
        profile = get_provider(name).profile
        for field in dataclasses.fields(profile):
            value: Any = getattr(profile, field.name)
            if field.name in _MAY_BE_NONE:
                continue
            assert value is not None, field.name
            if field.name not in _MAY_BE_EMPTY:
                assert value not in ("", ()), field.name

    def test_the_name_matches_the_registry_key(self, name: str) -> None:
        assert get_provider(name).profile.name == name

    def test_a_schema_dialect_is_declared_exactly_when_a_schema_is_supported(
        self, name: str
    ) -> None:
        profile = get_provider(name).profile
        if profile.output_schema is OutputSchemaStyle.NONE:
            assert profile.schema_dialect is None
        else:
            assert isinstance(profile.schema_dialect, SchemaDialect)


@pytest.mark.parametrize("name", PROVIDERS)
class TestEnvAndAuthConventions:
    """Invariants for the 0.6.1 env/auth-file fields, pinned across providers."""

    def test_auth_files_lie_inside_state_dirs(self, name: str) -> None:
        profile = get_provider(name).profile
        for auth_file in profile.auth_files:
            assert any(
                auth_file == state_dir or auth_file.startswith(f"{state_dir}/")
                for state_dir in profile.state_dirs
            ), auth_file

    def test_mcp_config_file_is_set_exactly_for_config_file_providers(self, name: str) -> None:
        profile = get_provider(name).profile
        if profile.mcp is McpMechanism.CONFIG_FILE:
            assert profile.mcp_config_file is not None
        else:
            assert profile.mcp_config_file is None

    def test_mcp_config_file_matches_where_the_installation_really_writes(
        self, name: str, tmp_path: Path
    ) -> None:
        profile = get_provider(name).profile
        if profile.mcp is not McpMechanism.CONFIG_FILE:
            pytest.skip("only CONFIG_FILE providers write a workspace MCP config")
        provider = get_provider(name)
        server = StdioMcpServer(name="probe", command="probe-cmd")
        assert profile.mcp_config_file is not None
        installation = provider.install_mcp(tmp_path, [server])
        try:
            assert (tmp_path / profile.mcp_config_file).is_file()
        finally:
            installation.restore()

    def test_state_root_env_is_an_uppercase_variable_name_when_set(self, name: str) -> None:
        profile = get_provider(name).profile
        if profile.state_root_env is not None:
            assert profile.state_root_env == profile.state_root_env.upper()

    def test_container_env_keys_are_uppercase(self, name: str) -> None:
        profile = get_provider(name).profile
        for key in profile.container_env:
            assert key == key.upper(), key


@pytest.mark.parametrize("name", PROVIDERS)
class TestArgv:
    def test_the_prompt_never_appears_in_argv(self, name: str) -> None:
        executor = FakeExecutor(scripted_turn(name, text="ok"))
        CliAgent(name, executor=executor, env=dict(_ENV)).run(_PROMPT)
        request = executor.requests[0]
        assert request.stdin == _PROMPT
        assert all(_PROMPT not in arg for arg in request.argv)

    def test_a_resumed_argv_carries_the_resume_flag_and_the_session_id(self, name: str) -> None:
        assert get_provider(name).profile.supports_resume
        argv = _argv(name, resume_session_id=_SESSION_ID)
        flag = RESUME_FLAGS[name]
        assert flag in argv
        assert argv[argv.index(flag) + 1] == _SESSION_ID

    def test_a_fresh_argv_carries_no_resume_flag(self, name: str) -> None:
        assert _SESSION_ID not in _argv(name)


@pytest.mark.parametrize("name", PROVIDERS)
class TestScriptedRoundTrip:
    def test_text_session_id_and_usage_survive_a_whole_turn(self, name: str) -> None:
        usage = _usage()
        executor = FakeExecutor(
            scripted_turn(name, text="pong", session_id=_SESSION_ID, usage=usage)
        )

        result = CliAgent(name, executor=executor, env=dict(_ENV)).run("ping")

        assert result.text == "pong"
        assert result.session_id == _SESSION_ID
        assert result.usage.tokens == usage
        assert result.usage.provider == name
        assert result.exit_code == 0

    def test_the_usage_invariant_holds(self, name: str) -> None:
        executor = FakeExecutor(scripted_turn(name, text="pong", usage=_usage()))
        tokens = CliAgent(name, executor=executor, env=dict(_ENV)).run("ping").usage.tokens
        assert tokens.cached_input_tokens <= tokens.input_tokens

    def test_a_usage_report_is_emitted_when_the_stream_carries_usage(self, name: str) -> None:
        recorder = RecordingEventHandler()
        executor = FakeExecutor(scripted_turn(name, text="pong", usage=_usage()))

        CliAgent(name, executor=executor, env=dict(_ENV), event_handler=recorder).run("ping")

        reports = [event for event in recorder.events if isinstance(event, UsageReport)]
        assert reports
        assert reports[-1].usage.tokens.cached_input_tokens <= reports[-1].usage.tokens.input_tokens

    def test_every_scripted_line_is_one_json_object(self, name: str) -> None:
        run = scripted_turn(name, text="hi", session_id=_SESSION_ID, usage=_usage())
        for line in run.stdout:
            assert isinstance(json.loads(line), dict), line

    def test_every_scripted_line_is_newline_terminated(self, name: str) -> None:
        """Line-delimited JSON: one object per line, and none spanning two."""
        run = scripted_turn(name, text="hi", session_id=_SESSION_ID, usage=_usage())
        for line in run.stdout:
            assert line.endswith("\n"), line
            assert line.count("\n") == 1, line


@pytest.mark.parametrize("name", PROVIDERS)
class TestParserRobustness:
    @pytest.mark.parametrize("line", ["42\n", '["a", "b"]\n', "not json at all\n", '"text"\n'])
    def test_a_non_object_line_becomes_raw_output(self, name: str, line: str) -> None:
        events = _replay(name, [line])
        assert [event for event in events if isinstance(event, RawOutput)] == [
            RawOutput(line.rstrip("\n"))
        ]

    def test_a_blank_line_produces_nothing(self, name: str) -> None:
        assert _replay(name, ["\n", "\n"]) == []


@pytest.mark.parametrize("name", PROVIDERS)
class TestScriptedLinesContract:
    def test_structured_output_is_supported_or_rejected_never_ignored(self, name: str) -> None:
        provider = get_provider(name)
        scripted_lines = get_scripted_lines(name)
        payload = {"answer": 4}
        if provider.profile.output_schema is OutputSchemaStyle.NONE:
            with pytest.raises(ValueError, match="no native output schema"):
                scripted_lines(text="x", structured_output=payload)
            return
        lines = scripted_lines(text="", structured_output=payload)
        events: list[AgentEvent] = []
        parser = provider.new_parser(events.append, expect_structured=True)
        for line in lines:
            parser.feed_stdout(line)
        assert parser.finish().structured_output == payload
