"""The shipped test doubles."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from agentshim import (
    AgentEventHandler,
    AssistantText,
    CliAgent,
    CliNotFoundError,
    CliTimeoutError,
    CommandRequest,
    NullSink,
    OutputSchema,
    SessionResumeError,
    SessionStarted,
    StdioMcpServer,
    TokenUsage,
    ToolCall,
    ToolResult,
    TurnRequest,
    UsageReport,
)
from agentshim.testing import (
    FakeCommandHandle,
    FakeExecutor,
    FakeRun,
    RecordingEventHandler,
    installed_mcp_servers,
    scripted_resume_failure,
    scripted_turn,
)

_ENV = {"PATH": "/usr/bin"}


def _agent(executor: FakeExecutor, *, event_handler: AgentEventHandler | None = None) -> CliAgent:
    return CliAgent("claude", executor=executor, env=dict(_ENV), event_handler=event_handler)


class TestFakeExecutor:
    def test_a_single_run_can_be_passed_directly(self) -> None:
        executor = FakeExecutor(FakeRun(stdout=["line\n"]))
        result = executor.run(CommandRequest(["x"], None, None, {}, None), NullSink())
        assert result.stdout == "line\n"

    def test_runs_are_consumed_in_order(self) -> None:
        executor = FakeExecutor([FakeRun(stdout=["a\n"]), FakeRun(stdout=["b\n"])])
        first = executor.run(CommandRequest(["x"], None, None, {}, None), NullSink())
        second = executor.run(CommandRequest(["x"], None, None, {}, None), NullSink())
        assert (first.stdout, second.stdout) == ("a\n", "b\n")

    def test_the_last_run_repeats_once_exhausted(self) -> None:
        executor = FakeExecutor([FakeRun(stdout=["only\n"])])
        for _ in range(3):
            assert (
                executor.run(CommandRequest(["x"], None, None, {}, None), NullSink()).stdout
                == "only\n"
            )

    def test_a_callable_picks_a_run_per_request(self) -> None:
        def choose(request: CommandRequest) -> FakeRun:
            return FakeRun(stdout=[f"{len(request.argv)}\n"])

        executor = FakeExecutor(choose)
        assert (
            executor.run(CommandRequest(["a", "b"], None, None, {}, None), NullSink()).stdout
            == "2\n"
        )

    def test_requests_are_recorded(self) -> None:
        executor = FakeExecutor(FakeRun())
        request = CommandRequest(["claude"], "hi", "/w", {}, 5)
        executor.run(request, NullSink())
        assert executor.requests == [request]

    def test_handles_are_recorded_and_track_stop_calls(self) -> None:
        executor = FakeExecutor(FakeRun())
        executor.run(CommandRequest(["x"], None, None, {}, None), NullSink())
        handle = executor.handles[0]
        assert isinstance(handle, FakeCommandHandle)
        assert (handle.terminated, handle.killed) == (False, False)
        handle.terminate()
        handle.kill()
        assert (handle.terminated, handle.killed) == (True, True)

    def test_the_handle_is_published_before_the_run_callable_is_consulted(self) -> None:
        # A blocking scripted run must be cancellable through the handle the
        # sink already holds, matching the real executor's ordering.
        executor: FakeExecutor | None = None

        def choose(request: CommandRequest) -> FakeRun:
            del request
            assert executor is not None
            assert executor.handles, "handle must exist before the run is chosen"
            return FakeRun()

        executor = FakeExecutor(choose)
        executor.run(CommandRequest(["x"], None, None, {}, None), NullSink())
        assert len(executor.handles) == 1

    def test_a_timeout_run_raises_and_kills(self) -> None:
        executor = FakeExecutor(FakeRun(timeout=True))
        with pytest.raises(CliTimeoutError):
            executor.run(CommandRequest(["x"], None, None, {}, 3), NullSink())
        assert executor.handles[0].killed is True

    def test_stderr_is_streamed_too(self) -> None:
        seen: list[str] = []

        class Sink(NullSink):
            def stderr(self, line: str) -> None:
                seen.append(line)

        FakeExecutor(FakeRun(stderr=["oops\n"])).run(
            CommandRequest(["x"], None, None, {}, None), Sink()
        )
        assert seen == ["oops\n"]

    def test_binary_lookup_defaults_to_a_plausible_path(self) -> None:
        assert FakeExecutor(FakeRun()).find_binary("claude", {}) == "/usr/local/bin/claude"

    def test_declared_binaries_are_enforced(self) -> None:
        executor = FakeExecutor(FakeRun(), binaries={"claude": "/opt/claude"})
        assert executor.find_binary("claude", {}) == "/opt/claude"
        with pytest.raises(CliNotFoundError):
            executor.find_binary("codex", {})


class TestScriptedTurn:
    def test_it_round_trips_through_the_real_parser(self) -> None:
        recorder = RecordingEventHandler()
        executor = FakeExecutor(scripted_turn("claude", text="pong", session_id="s1"))

        result = _agent(executor, event_handler=recorder).run("ping")

        assert result.text == "pong"
        assert result.session_id == "s1"
        kinds = {type(event).__name__ for event in recorder.events}
        assert {"SessionStarted", "AssistantText", "UsageReport"} <= kinds
        assert SessionStarted("s1") in recorder.events
        assert AssistantText("pong") in recorder.events

    def test_every_scripted_line_is_a_json_object(self) -> None:
        run = scripted_turn(
            "claude", text="hi", session_id="s1", tool_calls=[("Bash", {"cmd": "ls"}, "out")]
        )
        for line in run.stdout:
            assert line.endswith("\n")
            assert isinstance(json.loads(line), dict)

    def test_tool_calls_become_paired_events(self) -> None:
        recorder = RecordingEventHandler()
        executor = FakeExecutor(
            scripted_turn("claude", text="done", tool_calls=[("Bash", {"cmd": "ls"}, "file.txt")])
        )
        _agent(executor, event_handler=recorder).run("go")

        calls = [event for event in recorder.events if isinstance(event, ToolCall)]
        results = [event for event in recorder.events if isinstance(event, ToolResult)]
        assert [call.tool for call in calls] == ["Bash"]
        assert calls[0].args == {"cmd": "ls"}
        assert [result.tool for result in results] == ["Bash"]
        assert results[0].stdout == "file.txt"

    def test_usage_round_trips_exactly(self) -> None:
        usage = TokenUsage(
            input_tokens=150,
            output_tokens=40,
            cached_input_tokens=50,
            cache_write_input_tokens=30,
            turns=4,
        )
        executor = FakeExecutor(scripted_turn("claude", text="done", usage=usage))
        result = _agent(executor).run("go")
        assert result.usage.tokens == usage

    def test_structured_output_round_trips(self, tmp_path: Path) -> None:
        payload = {"a": 3, "b": [1, 2]}
        executor = FakeExecutor(scripted_turn("claude", text="done", structured_output=payload))
        request = TurnRequest(
            prompt="go",
            output_schema=OutputSchema(schema={"type": "object"}, host_dir=tmp_path),
        )
        assert _agent(executor).run(request).structured_output == payload

    def test_an_unrequested_structured_payload_is_ignored(self) -> None:
        # Only a turn that asked for a schema gets a payload; the prose
        # answer stays the result otherwise.
        executor = FakeExecutor(scripted_turn("claude", text="done", structured_output={"a": 3}))
        result = _agent(executor).run("go")
        assert result.structured_output is None
        assert result.text == "done"

    def test_a_nonzero_return_code_is_scripted(self) -> None:
        run = scripted_turn("claude", text="done", returncode=2)
        assert run.returncode == 2

    def test_usage_reports_are_emitted(self) -> None:
        recorder = RecordingEventHandler()
        executor = FakeExecutor(
            scripted_turn("claude", text="x", usage=TokenUsage(input_tokens=5, turns=1))
        )
        _agent(executor, event_handler=recorder).run("go")
        assert any(isinstance(event, UsageReport) for event in recorder.events)

    def test_an_unknown_provider_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no scripted stream"):
            scripted_turn("nope", text="x")


class TestScriptedResumeFailure:
    def test_it_raises_session_resume_error_on_a_resumed_turn(self) -> None:
        executor = FakeExecutor(scripted_resume_failure("claude", session_id="old-session"))
        session = _agent(executor).start_session()
        assert session.adopt("old-session")

        with pytest.raises(SessionResumeError):
            session.turn("go")

    def test_the_scripted_message_names_the_session(self) -> None:
        run = scripted_resume_failure("codex", session_id="thread-xyz")
        assert any("thread-xyz" in line for line in run.stderr)

    def test_a_nonzero_exit_code_is_scripted(self) -> None:
        run = scripted_resume_failure("claude")
        assert run.returncode != 0

    def test_an_unknown_provider_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no resume-failure stream"):
            scripted_resume_failure("nope")


class TestInstalledMcpServers:
    """Documents and pins the ``FakeExecutor`` ``run``-callback usage pattern.

    A config-file provider's config only exists on disk for the lifetime of
    the turn: the library restores (and, when nothing else asked for one,
    removes) it in the session's ``finally``, which runs before ``run()``
    returns. So the helper has to be called from inside the ``run`` callback,
    not after.
    """

    def test_a_config_file_provider_is_only_readable_during_the_turn(self, tmp_path: Path) -> None:
        stdio = StdioMcpServer(name="tool", command="npx", args=["-y", "thing"], env={"K": "v"})
        during: dict[str, object] = {}

        def run(request: CommandRequest) -> FakeRun:
            during.update(installed_mcp_servers("claude", request, tmp_path))
            return scripted_turn("claude", text="ok")

        executor = FakeExecutor(run)
        CliAgent("claude", executor=executor, env=_ENV).run(
            TurnRequest(prompt="go", cwd=str(tmp_path), mcp_servers=[stdio])
        )

        assert during["tool"] == {"command": "npx", "args": ["-y", "thing"], "env": {"K": "v"}}
        # The turn is over now, and claude's install is restored: nothing
        # left to read.
        after = installed_mcp_servers("claude", executor.requests[-1], tmp_path)
        assert after == {}

    def test_codex_flags_can_still_be_read_after_the_turn(self, tmp_path: Path) -> None:
        # A CLI-flags provider has no config file to restore: the servers
        # live in the one argv the turn ran, which ``FakeExecutor`` keeps on
        # ``requests`` for the life of the test either way.
        stdio = StdioMcpServer(name="tool", command="npx", args=["-y", "thing"])
        executor = FakeExecutor(scripted_turn("codex", text="ok"))

        CliAgent("codex", executor=executor, env=_ENV).run(
            TurnRequest(prompt="go", cwd=str(tmp_path), mcp_servers=[stdio])
        )

        servers = installed_mcp_servers("codex", executor.requests[-1], tmp_path)
        assert servers["tool"]["command"] == "npx"
