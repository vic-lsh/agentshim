"""``CliAgent`` and ``AgentSession`` semantics."""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from agentshim import (
    AgentEvent,
    AgentEventHandler,
    AgentSession,
    AssistantText,
    CliAgent,
    CliCheckError,
    CliExitError,
    CliNotFoundError,
    CliTimeoutError,
    CommandRequest,
    CommandResult,
    CommandStreamSink,
    HostCommandExecutor,
    McpMechanism,
    OutputSchema,
    OutputSchemaStyle,
    ProviderCapabilityError,
    RunFinished,
    RunStarted,
    SchemaDialectError,
    SessionResumeError,
    SessionStarted,
    StdioMcpServer,
    TokenUsage,
    TurnRequest,
    UsageReport,
)
from agentshim.providers.claude import ClaudeProvider
from agentshim.testing import FakeExecutor, FakeRun, RecordingEventHandler, scripted_turn

_ENV = {"PATH": "/usr/bin:/bin", "HOME": "/home/tester"}

_MAPPING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"metrics": {"type": "object", "additionalProperties": {"type": "number"}}},
    "required": ["metrics"],
    "additionalProperties": False,
}


def _agent(
    executor: FakeExecutor,
    *,
    event_handler: AgentEventHandler | None = None,
    event_handlers: Sequence[AgentEventHandler] = (),
    check_timeout: float = 15.0,
    log: Callable[[str], None] | None = None,
) -> CliAgent:
    return CliAgent(
        "claude",
        executor=executor,
        env=dict(_ENV),
        event_handler=event_handler,
        event_handlers=event_handlers,
        check_timeout=check_timeout,
        log=log,
    )


class TestConstruction:
    def test_binary_lookup_and_health_check_run_once(self) -> None:
        executor = FakeExecutor(FakeRun(), binaries={"claude": "/opt/claude"})
        agent = _agent(executor)
        assert agent.binary_path == "/opt/claude"
        assert executor.checked == ["/opt/claude"]

    def test_a_missing_binary_fails_at_construction(self) -> None:
        executor = FakeExecutor(FakeRun(), binaries={"other": "/opt/other"})
        with pytest.raises(CliNotFoundError):
            _agent(executor)

    def test_a_broken_binary_fails_at_construction(self) -> None:
        class Broken(FakeExecutor):
            def check_binary(self, path: str, env: Mapping[str, str], *, timeout: float) -> None:
                del env, timeout  # names fixed by the CommandExecutor protocol
                raise CliCheckError(path, "not working")

        with pytest.raises(CliCheckError):
            _agent(Broken(FakeRun()))

    def test_a_provider_instance_can_be_passed_directly(self) -> None:
        agent = CliAgent(
            ClaudeProvider(sandbox=True), executor=FakeExecutor(FakeRun()), env=dict(_ENV)
        )
        assert agent.profile.name == "claude"

    def test_the_supplied_env_is_used_verbatim(self) -> None:
        executor = FakeExecutor(scripted_turn("claude", text="hi"))
        agent = _agent(executor)
        agent.run("go")
        assert dict(executor.requests[0].env) == _ENV

    def test_the_log_callback_receives_a_readiness_line(self) -> None:
        messages: list[str] = []
        _agent(FakeExecutor(FakeRun()), log=messages.append)
        assert any("Claude Code" in message for message in messages)

    def test_check_timeout_is_forwarded(self) -> None:
        seen: list[float] = []

        class Recording(FakeExecutor):
            def check_binary(self, path: str, env: Mapping[str, str], *, timeout: float) -> None:
                del path, env  # names fixed by the CommandExecutor protocol
                seen.append(timeout)

        _agent(Recording(FakeRun()), check_timeout=42.0)
        assert seen == [42.0]


class TestOneShotRun:
    def test_run_returns_a_turn_result(self) -> None:
        executor = FakeExecutor(scripted_turn("claude", text="pong", session_id="s1"))
        result = _agent(executor).run("ping")
        assert result.text == "pong"
        assert result.session_id == "s1"
        assert result.resumed is False
        assert result.exit_code == 0
        assert result.duration_ms >= 0

    def test_run_does_not_retain_conversation_state(self) -> None:
        executor = FakeExecutor(scripted_turn("claude", text="ok", session_id="s1"))
        agent = _agent(executor)
        agent.run("first")
        agent.run("second")
        assert all("--resume" not in list(request.argv) for request in executor.requests)

    def test_run_forwards_cwd_and_timeout(self) -> None:
        executor = FakeExecutor(scripted_turn("claude", text="ok"))
        _agent(executor).run("hi", cwd="/workspace", timeout=12.5)
        assert executor.requests[0].cwd == "/workspace"
        assert executor.requests[0].timeout == 12.5


class TestPromptDelivery:
    def test_the_prompt_goes_on_stdin_only(self) -> None:
        executor = FakeExecutor(scripted_turn("claude", text="ok"))
        _agent(executor).run("deploy the app")
        request = executor.requests[0]
        assert request.stdin == "deploy the app"
        assert all("deploy the app" not in arg for arg in request.argv)


class TestSessionDefaults:
    def test_session_defaults_apply_to_every_turn(self) -> None:
        executor = FakeExecutor(scripted_turn("claude", text="ok"))
        session = _agent(executor).start_session(cwd="/sess", timeout=60)
        session.turn("hi")
        assert executor.requests[0].cwd == "/sess"
        assert executor.requests[0].timeout == 60

    def test_a_request_overrides_the_session_defaults(self) -> None:
        executor = FakeExecutor(scripted_turn("claude", text="ok"))
        session = _agent(executor).start_session(cwd="/sess", timeout=60)
        session.turn(TurnRequest(prompt="hi", cwd="/per-call", timeout=5))
        assert executor.requests[0].cwd == "/per-call"
        assert executor.requests[0].timeout == 5

    def test_timeout_none_means_no_limit(self) -> None:
        executor = FakeExecutor(scripted_turn("claude", text="ok"))
        _agent(executor).start_session().turn("hi")
        assert executor.requests[0].timeout is None

    def test_a_request_env_overlays_the_agent_env(self) -> None:
        executor = FakeExecutor(scripted_turn("claude", text="ok"))
        _agent(executor).start_session().turn(
            TurnRequest(prompt="hi", env={"EXTRA": "1", "HOME": "/other"})
        )
        env = dict(executor.requests[0].env)
        assert env["EXTRA"] == "1"
        assert env["HOME"] == "/other"
        assert env["PATH"] == _ENV["PATH"]

    def test_extra_args_reach_argv(self) -> None:
        executor = FakeExecutor(scripted_turn("claude", text="ok"))
        _agent(executor).start_session().turn(TurnRequest(prompt="hi", extra_args=("--foo", "bar")))
        assert list(executor.requests[0].argv)[-2:] == ["--foo", "bar"]


class TestResume:
    def test_the_second_turn_resumes_the_first(self) -> None:
        executor = FakeExecutor([scripted_turn("claude", text="one", session_id="s1")] * 2)
        session = _agent(executor).start_session()

        first = session.turn("hello")
        assert first.resumed is False
        assert session.session_id == "s1"

        second = session.turn("follow up")
        assert second.resumed is True
        argv = list(executor.requests[1].argv)
        assert argv[argv.index("--resume") + 1] == "s1"

    def test_a_preset_session_id_makes_the_first_turn_a_resume(self) -> None:
        executor = FakeExecutor(scripted_turn("claude", text="ok", session_id="s1"))
        result = _agent(executor).start_session(session_id="s1").turn("hi")
        assert result.resumed is True
        assert "--resume" in list(executor.requests[0].argv)

    def test_adopt_sets_the_conversation(self) -> None:
        session = _agent(FakeExecutor(scripted_turn("claude", text="ok"))).start_session()
        assert session.adopt("s9") is True
        assert session.session_id == "s9"

    def test_adopt_is_refused_while_a_turn_is_live(self) -> None:
        adopted: list[bool] = []

        class Reentrant(FakeExecutor):
            def run(self, request: CommandRequest, sink: CommandStreamSink) -> CommandResult:
                adopted.append(session.adopt("other"))
                return super().run(request, sink)

        executor = Reentrant(scripted_turn("claude", text="ok", session_id="s1"))
        session = _agent(executor).start_session()
        session.turn("hi")
        assert adopted == [False]
        assert session.session_id == "s1"

    def test_adopt_is_refused_when_the_provider_cannot_resume(self) -> None:
        provider = ClaudeProvider()
        provider.profile = replace(provider.profile, supports_resume=False)  # pyright: ignore[reportAttributeAccessIssue]
        agent = CliAgent(provider, executor=FakeExecutor(FakeRun()), env=dict(_ENV))
        session = agent.start_session()
        assert session.adopt("s1") is False
        assert session.session_id is None

    def test_forget_starts_a_fresh_conversation(self) -> None:
        executor = FakeExecutor([scripted_turn("claude", text="ok", session_id="s1")] * 2)
        session = _agent(executor).start_session()
        session.turn("hi")
        session.forget()
        assert session.session_id is None
        session.turn("again")
        assert "--resume" not in list(executor.requests[1].argv)

    def test_a_failing_resumed_turn_raises_session_resume_failed(self) -> None:
        executor = FakeExecutor(FakeRun(returncode=1, stderr=["no conversation found\n"]))
        session = _agent(executor).start_session(session_id="gone")
        with pytest.raises(SessionResumeError) as excinfo:
            session.turn("hi")
        assert excinfo.value.session_id == "gone"

    def test_a_failing_fresh_turn_raises_a_plain_exit_error(self) -> None:
        executor = FakeExecutor(FakeRun(returncode=1, stderr=["boom\n"]))
        with pytest.raises(CliExitError) as excinfo:
            _agent(executor).start_session().turn("hi")
        assert not isinstance(excinfo.value, SessionResumeError)
        assert excinfo.value.returncode == 1
        assert excinfo.value.stderr == "boom\n"


class TestLastResult:
    def test_last_result_tracks_the_latest_turn(self) -> None:
        executor = FakeExecutor(
            [scripted_turn("claude", text="one"), scripted_turn("claude", text="two")]
        )
        session = _agent(executor).start_session()
        assert session.last_result is None
        session.turn("a")
        assert session.last_result is not None
        assert session.last_result.text == "one"
        session.turn("b")
        assert session.last_result.text == "two"

    def test_a_failed_turn_leaves_the_previous_result(self) -> None:
        executor = FakeExecutor([scripted_turn("claude", text="one"), FakeRun(returncode=1)])
        session = _agent(executor).start_session()
        session.turn("a")
        with pytest.raises(CliExitError):
            session.turn("b")
        assert session.last_result is not None
        assert session.last_result.text == "one"


class TestEvents:
    def test_the_event_stream_brackets_the_run(self) -> None:
        recorder = RecordingEventHandler()
        executor = FakeExecutor(scripted_turn("claude", text="pong", session_id="s1"))
        _agent(executor, event_handler=recorder).run("ping")

        kinds = [type(event).__name__ for event in recorder.events]
        assert kinds[0] == "RunStarted"
        assert kinds[-1] == "RunFinished"
        assert "SessionStarted" in kinds
        assert "AssistantText" in kinds
        assert "UsageReport" in kinds

    def test_run_started_carries_the_real_argv(self) -> None:
        recorder = RecordingEventHandler()
        executor = FakeExecutor(scripted_turn("claude", text="ok"))
        _agent(executor, event_handler=recorder).run("ping")
        started = recorder.events[0]
        assert isinstance(started, RunStarted)
        assert list(started.argv) == list(executor.requests[0].argv)

    def test_run_finished_carries_the_exit_code(self) -> None:
        recorder = RecordingEventHandler()
        executor = FakeExecutor(FakeRun(returncode=3))
        with pytest.raises(CliExitError):
            _agent(executor, event_handler=recorder).run("ping")
        assert RunFinished(3) in recorder.events

    def test_both_handler_spellings_receive_events(self) -> None:
        first = RecordingEventHandler()
        second = RecordingEventHandler()
        executor = FakeExecutor(scripted_turn("claude", text="ok"))
        _agent(executor, event_handler=first, event_handlers=[second]).run("ping")
        assert first.events == second.events
        assert first.events

    def test_handlers_run_on_the_calling_thread(self) -> None:
        threads: list[int] = []

        class ThreadRecorder:
            def on_event(self, event: AgentEvent) -> None:
                del event  # name fixed by the AgentEventHandler protocol
                threads.append(threading.get_ident())

        binary = shutil.which("cat")
        if binary is None:
            pytest.skip("cat is not available")
        lines = scripted_turn("claude", text="pong", session_id="s1").stdout

        agent = CliAgent(
            "claude",
            executor=_CatExecutor(binary),
            env=os.environ.copy(),
            event_handler=ThreadRecorder(),
        )
        agent.run("".join(lines))

        assert threads
        assert set(threads) == {threading.get_ident()}


class _CatExecutor(HostCommandExecutor):
    """Runs ``cat`` so the parser is fed by real reader threads."""

    def __init__(self, binary: str) -> None:
        self._binary = binary

    def find_binary(self, name: str, env: Mapping[str, str]) -> str:
        del name, env  # names fixed by the CommandExecutor protocol
        return self._binary

    def check_binary(self, path: str, env: Mapping[str, str], *, timeout: float) -> None:
        del path, env, timeout  # names fixed by the CommandExecutor protocol

    def run(self, request: CommandRequest, sink: CommandStreamSink) -> CommandResult:
        return super().run(replace(request, argv=[self._binary], timeout=30), sink)


class TestStructuredOutput:
    def test_structured_output_reaches_the_turn_result(self, tmp_path: Path) -> None:
        payload = {"metrics": {"latency_ms": 12.5}}
        executor = FakeExecutor(scripted_turn("claude", text="done", structured_output=payload))
        result = (
            _agent(executor)
            .start_session()
            .turn(
                TurnRequest(
                    prompt="go",
                    output_schema=OutputSchema(schema=_MAPPING_SCHEMA, host_dir=tmp_path),
                )
            )
        )
        assert result.structured_output == payload
        assert result.text == "done"

    def test_the_schema_is_inlined_into_argv(self, tmp_path: Path) -> None:
        executor = FakeExecutor(
            scripted_turn("claude", text="done", structured_output={"metrics": {}})
        )
        _agent(executor).start_session().turn(
            TurnRequest(
                prompt="go", output_schema=OutputSchema(schema=_MAPPING_SCHEMA, host_dir=tmp_path)
            )
        )
        argv = list(executor.requests[0].argv)
        inline = argv[argv.index("--json-schema") + 1]
        assert " " not in inline
        assert json.loads(inline) == _MAPPING_SCHEMA

    def test_an_inline_provider_writes_no_schema_file(self, tmp_path: Path) -> None:
        executor = FakeExecutor(scripted_turn("claude", text="done"))
        _agent(executor).start_session().turn(
            TurnRequest(
                prompt="go", output_schema=OutputSchema(schema=_MAPPING_SCHEMA, host_dir=tmp_path)
            )
        )
        assert list(tmp_path.iterdir()) == []

    def test_structured_output_is_none_without_a_schema(self) -> None:
        executor = FakeExecutor(scripted_turn("claude", text="done"))
        assert _agent(executor).run("go").structured_output is None

    def test_a_schema_the_dialect_rejects_fails_before_the_process_starts(
        self, tmp_path: Path
    ) -> None:
        executor = FakeExecutor(scripted_turn("claude", text="done"))
        schema = {"type": "object", "properties": {"a": {"type": "string"}}, "allOf": []}
        with pytest.raises(SchemaDialectError) as excinfo:
            _agent(executor).start_session().turn(
                TurnRequest(
                    prompt="go", output_schema=OutputSchema(schema=schema, host_dir=tmp_path)
                )
            )
        assert any("allOf" in problem for problem in excinfo.value.problems)
        assert executor.requests == []


class TestCapabilityChecks:
    def _provider_without(self, **overrides: object) -> ClaudeProvider:
        provider = ClaudeProvider()
        provider.profile = replace(provider.profile, **overrides)  # pyright: ignore[reportAttributeAccessIssue]
        return provider

    def test_reasoning_effort_is_checked(self) -> None:
        provider = self._provider_without(supports_reasoning_effort=False)
        executor = FakeExecutor(scripted_turn("claude", text="ok"))
        agent = CliAgent(provider, executor=executor, env=dict(_ENV))
        with pytest.raises(ProviderCapabilityError, match="reasoning effort"):
            agent.start_session().turn(TurnRequest(prompt="hi", reasoning_effort="high"))
        assert executor.requests == []
        assert OutputSchemaStyle.NONE is not None

    def test_output_schema_is_checked(self, tmp_path: Path) -> None:
        provider = self._provider_without(output_schema=OutputSchemaStyle.NONE)
        agent = CliAgent(provider, executor=FakeExecutor(FakeRun()), env=dict(_ENV))
        with pytest.raises(ProviderCapabilityError, match="output schema"):
            agent.start_session().turn(
                TurnRequest(
                    prompt="hi",
                    output_schema=OutputSchema(schema=_MAPPING_SCHEMA, host_dir=tmp_path),
                )
            )

    def test_mcp_support_is_checked(self) -> None:
        provider = self._provider_without(mcp=McpMechanism.NONE)
        agent = CliAgent(provider, executor=FakeExecutor(FakeRun()), env=dict(_ENV))
        with pytest.raises(ProviderCapabilityError, match="MCP"):
            agent.start_session().turn(
                TurnRequest(prompt="hi", mcp_servers=[StdioMcpServer(name="a", command="python")])
            )

    def test_reasoning_effort_reaches_argv_when_supported(self) -> None:
        executor = FakeExecutor(scripted_turn("claude", text="ok"))
        _agent(executor).start_session().turn(TurnRequest(prompt="hi", reasoning_effort="high"))
        argv = list(executor.requests[0].argv)
        assert argv[argv.index("--effort") + 1] == "high"


class TestMcpLifecycle:
    def test_servers_are_installed_before_and_restored_after_the_turn(self, tmp_path: Path) -> None:
        seen: list[bool] = []

        class Watching(FakeExecutor):
            def run(self, request: CommandRequest, sink: CommandStreamSink) -> CommandResult:
                seen.append((tmp_path / ".mcp.json").exists())
                return super().run(request, sink)

        executor = Watching(scripted_turn("claude", text="ok"))
        _agent(executor).start_session(cwd=str(tmp_path)).turn(
            TurnRequest(prompt="hi", mcp_servers=[StdioMcpServer(name="board", command="python")])
        )

        assert seen == [True]
        assert not (tmp_path / ".mcp.json").exists()

    def test_the_config_is_restored_even_when_the_turn_fails(self, tmp_path: Path) -> None:
        executor = FakeExecutor(FakeRun(returncode=1))
        with pytest.raises(CliExitError):
            _agent(executor).start_session(cwd=str(tmp_path)).turn(
                TurnRequest(
                    prompt="hi", mcp_servers=[StdioMcpServer(name="board", command="python")]
                )
            )
        assert not (tmp_path / ".mcp.json").exists()

    def test_no_config_is_written_when_no_servers_are_requested(self, tmp_path: Path) -> None:
        executor = FakeExecutor(scripted_turn("claude", text="ok"))
        _agent(executor).start_session(cwd=str(tmp_path)).turn("hi")
        assert list(tmp_path.iterdir()) == []

    def test_mcp_workspace_overrides_cwd_for_the_config_file(self, tmp_path: Path) -> None:
        # A container-executed turn has no host cwd: the CLI runs at the
        # container path while the config file belongs on the bind-mounted
        # host workspace.
        seen: list[bool] = []

        class Watching(FakeExecutor):
            def run(self, request: CommandRequest, sink: CommandStreamSink) -> CommandResult:
                seen.append((tmp_path / ".mcp.json").exists())
                assert request.cwd is None
                return super().run(request, sink)

        executor = Watching(scripted_turn("claude", text="ok"))
        _agent(executor).start_session().turn(
            TurnRequest(
                prompt="hi",
                mcp_servers=[StdioMcpServer(name="board", command="python")],
                mcp_workspace=tmp_path,
            )
        )

        assert seen == [True]
        assert not (tmp_path / ".mcp.json").exists()


class TestTimeoutAndCancel:
    def test_a_timeout_propagates_as_cli_timeout_error(self) -> None:
        executor = FakeExecutor(FakeRun(timeout=True))
        with pytest.raises(CliTimeoutError):
            _agent(executor).start_session(timeout=1.0).turn("hi")

    def test_a_real_slow_subprocess_times_out(self, tmp_path: Path) -> None:
        agent = _real_agent(tmp_path, "sleep 30")
        with pytest.raises(CliTimeoutError) as excinfo:
            agent.start_session(timeout=0.5).turn("hi")
        assert excinfo.value.timeout == 0.5

    def test_cancel_terminates_a_real_subprocess(self, tmp_path: Path) -> None:
        agent = _real_agent(tmp_path, "sleep 30")
        session = agent.start_session(timeout=20)
        started = time.monotonic()

        canceller = threading.Thread(target=_cancel_soon, args=(session,))
        canceller.start()
        try:
            with pytest.raises(CliExitError):
                session.turn("hi")
        finally:
            canceller.join()

        assert time.monotonic() - started < 15

    def test_cancel_before_a_turn_is_a_no_op(self) -> None:
        _agent(FakeExecutor(FakeRun())).start_session().cancel()


def _cancel_soon(session: AgentSession) -> None:
    time.sleep(0.4)
    session.cancel(grace_s=1.0)


def _real_agent(tmp_path: Path, body: str) -> CliAgent:
    script = tmp_path / "claude"
    script.write_text(f"#!/bin/sh\nif [ \"$1\" = '--help' ]; then exit 0; fi\n{body}\n")
    script.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}{os.pathsep}{env['PATH']}"
    return CliAgent("claude", executor=HostCommandExecutor(), env=env)


class TestRealSubprocessTurn:
    def test_a_scripted_stream_from_a_real_process_parses(self, tmp_path: Path) -> None:
        lines = scripted_turn("claude", text="pong", session_id="s1").stdout
        payload = "".join(lines).replace("'", "'\"'\"'")
        agent = _real_agent(tmp_path, f"printf '%s' '{payload}'")
        recorder = RecordingEventHandler()
        agent.event_handler = recorder

        result = agent.start_session(timeout=30).turn("ping")

        assert result.text == "pong"
        assert result.session_id == "s1"
        assert result.exit_code == 0
        kinds = {type(event).__name__ for event in recorder.events}
        assert {"SessionStarted", "AssistantText", "UsageReport"} <= kinds


def test_a_string_request_is_accepted_everywhere() -> None:
    executor = FakeExecutor([scripted_turn("claude", text="ok")] * 2)
    agent = _agent(executor)
    assert agent.run("prompt").text == "ok"
    assert agent.start_session().turn("prompt").text == "ok"


def test_usage_survives_onto_the_turn_result() -> None:
    usage = TokenUsage(input_tokens=7050, output_tokens=120, cached_input_tokens=50, turns=9)
    executor = FakeExecutor(scripted_turn("claude", text="done", usage=usage))
    result = _agent(executor).run("go")
    assert result.usage.tokens.turns == 9
    assert result.usage.tokens.output_tokens == 120
    assert result.usage.tokens.input_tokens == 7050
    assert result.usage.tokens.cached_input_tokens == 50
    assert result.usage.provider == "claude"


def test_events_and_result_agree_on_usage() -> None:
    recorder = RecordingEventHandler()
    executor = FakeExecutor(
        scripted_turn("claude", text="done", usage=TokenUsage(input_tokens=10, turns=1))
    )
    result = _agent(executor, event_handler=recorder).run("go")
    reports = [event for event in recorder.events if isinstance(event, UsageReport)]
    assert reports[-1].usage == result.usage


def test_assistant_text_events_match_the_result_text() -> None:
    recorder = RecordingEventHandler()
    executor = FakeExecutor(scripted_turn("claude", text="hello", session_id="s1"))
    result = _agent(executor, event_handler=recorder).run("go")
    assert AssistantText("hello") in recorder.events
    assert SessionStarted("s1") in recorder.events
    assert result.text == "hello"


def test_python_is_not_required_on_the_path_for_argv() -> None:
    """argv[0] is always the resolved binary path, never a bare name."""
    executor = FakeExecutor(
        scripted_turn("claude", text="ok"), binaries={"claude": "/opt/bin/claude"}
    )
    agent = _agent(executor)
    agent.run("go")
    assert next(iter(executor.requests[0].argv)) == "/opt/bin/claude"
    assert sys.executable  # sanity: the test runner has an interpreter
