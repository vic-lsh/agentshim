"""``OneShotTransport``: a conversation is the same turns as driving ``AgentSession``.

The headline property is equivalence: scripted executor runs fed through
``CliAgent``/``AgentSession`` and through the transport (and through a full
``Agent`` session) must produce the same results, events and commands, for
every provider.
"""

from __future__ import annotations

import threading
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from agentshim import (
    Agent,
    ApprovalPolicy,
    CliAgent,
    CliExitError,
    CommandRequest,
    CommandResult,
    CommandStreamSink,
    Continuity,
    ConversationSpec,
    FailureKind,
    NativePermissions,
    OneShotTransport,
    OutputSchema,
    ProviderCapabilityError,
    SessionResumeError,
    SessionStateError,
    SkillScope,
    StdioMcpServer,
    TurnInterrupted,
    TurnRequest,
    TurnResult,
    provider_names,
)
from agentshim.testing import (
    FakeCommandHandle,
    FakeConfinement,
    FakeExecutor,
    FakeRun,
    RecordingEventHandler,
    scripted_failure,
    scripted_resume_failure,
    scripted_turn,
)
from agentshim.testing.contracts import (
    CheckpointStoreContract,
    ConversationContract,
    TransportContract,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from agentshim import AgentEvent, CheckpointStore, Conversation, Transport

_ENV = {"PATH": "/usr/bin:/bin", "HOME": "/home/tester"}
_PROVIDERS = provider_names()


def _ignore(event: AgentEvent) -> None:
    """Event callback that drops everything."""


def _spec(**changes: object) -> ConversationSpec:
    base = ConversationSpec(
        cwd="/work",
        model=None,
        permissions=NativePermissions.bypass(),
        approvals=ApprovalPolicy.DENY,
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def _same(result: TurnResult) -> TurnResult:
    """A result without its measured wall time, which differs between runs."""
    return replace(result, duration_ms=0)


def _legacy_turns(
    provider: str, runs: list[FakeRun], prompts: list[str], *, resume_id: str | None = None
) -> tuple[list[TurnResult], list[object], FakeExecutor]:
    executor = FakeExecutor(list(runs))
    handler = RecordingEventHandler()
    agent = CliAgent(provider, executor=executor, env=dict(_ENV), event_handler=handler)
    session = agent.start_session(cwd="/work", session_id=resume_id)
    results = [session.turn(TurnRequest(prompt=prompt)) for prompt in prompts]
    return results, list(handler.events), executor


def _transport_turns(
    provider: str, runs: list[FakeRun], prompts: list[str], *, resume_id: str | None = None
) -> tuple[list[TurnResult], list[object], FakeExecutor]:
    executor = FakeExecutor(list(runs))
    handler = RecordingEventHandler()
    transport = OneShotTransport(provider, executor=executor, env=dict(_ENV))
    conversation = transport.open(_spec(resume_id=resume_id))
    results = [
        conversation.turn(TurnRequest(prompt=prompt), handler.on_event) for prompt in prompts
    ]
    return results, list(handler.events), executor


def _raised(
    turns: Callable[..., object], provider: str, runs: list[FakeRun], resume_id: str | None = None
) -> CliExitError:
    """The error one turn raises when driven through *turns*."""
    with pytest.raises(CliExitError) as info:
        turns(provider, runs, ["x"], resume_id=resume_id)
    return info.value


@pytest.mark.parametrize("provider", _PROVIDERS)
class TestEquivalenceWithAgentSession:
    def test_fresh_then_resumed_turns_match(self, provider: str) -> None:
        runs = [
            scripted_turn(provider, text="one", session_id="s-1"),
            scripted_turn(provider, text="two", session_id="s-1"),
        ]
        prompts = ["first", "second"]
        legacy, legacy_events, legacy_exec = _legacy_turns(provider, runs, prompts)
        new, new_events, new_exec = _transport_turns(provider, runs, prompts)
        assert [_same(r) for r in new] == [_same(r) for r in legacy]
        assert new_events == legacy_events
        assert new_exec.requests == legacy_exec.requests

    def test_an_adopted_conversation_resumes_the_same_way(self, provider: str) -> None:
        runs = [scripted_turn(provider, text="back", session_id="s-9")]
        legacy, legacy_events, legacy_exec = _legacy_turns(provider, runs, ["x"], resume_id="s-9")
        new, new_events, new_exec = _transport_turns(provider, runs, ["x"], resume_id="s-9")
        assert [_same(r) for r in new] == [_same(r) for r in legacy]
        assert new_events == legacy_events
        assert new_exec.requests == legacy_exec.requests
        assert new[0].resumed is True

    def test_a_refused_resume_raises_the_same_error(self, provider: str) -> None:
        runs = [scripted_resume_failure(provider, session_id="s-1")]
        errors = [
            _raised(turns, provider, runs, resume_id="s-1")
            for turns in (_legacy_turns, _transport_turns)
        ]
        assert type(errors[0]) is type(errors[1])
        assert str(errors[0]) == str(errors[1])
        assert errors[0].kind == errors[1].kind  # type: ignore[attr-defined]

    def test_a_session_over_the_transport_matches_the_legacy_turns(self, provider: str) -> None:
        runs = [
            scripted_turn(provider, text="one", session_id="s-1"),
            scripted_turn(provider, text="two", session_id="s-1"),
        ]
        legacy, legacy_events, legacy_exec = _legacy_turns(provider, runs, ["a", "b"])
        executor = FakeExecutor(list(runs))
        handler = RecordingEventHandler()
        agent = Agent(
            provider,
            executor=executor,
            env=dict(_ENV),
            permissions=NativePermissions.bypass(),
            approvals=ApprovalPolicy.DENY,
            event_handlers=(handler,),
        )
        session = agent.session("/work")
        turns = [session.run(session.prepare_turn(TurnRequest(prompt=p))) for p in ("a", "b")]
        assert [_same(t.result) for t in turns] == [_same(r) for r in legacy]
        expected = [
            Continuity.CONTINUED,
            Continuity.RESET if agent.profile.renewal else Continuity.CONTINUED,
        ]
        assert [t.continuity for t in turns] == expected
        assert list(handler.events) == legacy_events
        assert executor.requests == legacy_exec.requests


@pytest.mark.parametrize("provider", ["claude", "codex"])
@pytest.mark.parametrize("kind", list(FailureKind))
def test_a_classified_failure_keeps_its_kind(provider: str, kind: FailureKind) -> None:
    if provider == "codex" and kind is FailureKind.SCHEMA:
        pytest.skip("codex decoding is constrained to the schema; it never fails this way")
    run = scripted_failure(provider, kind)
    raised = [_raised(turns, provider, [run]) for turns in (_legacy_turns, _transport_turns)]
    assert raised[0].kind is raised[1].kind is kind
    assert str(raised[0]) == str(raised[1])
    assert raised[0].detail == raised[1].detail


@pytest.mark.parametrize("provider", [p for p in _PROVIDERS if p != "claude"])
def test_spec_mcp_servers_reach_the_command_when_the_request_has_none(provider: str) -> None:
    server = StdioMcpServer(name="srv", command="python", args=("-m", "x"))
    profile = OneShotTransport(provider, executor=FakeExecutor(FakeRun()), env=_ENV).profile
    if profile.mcp.value != "cli_flags":
        pytest.skip("only flag-based MCP installs are visible in argv without a workspace")
    runs = [scripted_turn(provider, text="t", session_id="s")]
    via_request = FakeExecutor(list(runs))
    CliAgent(provider, executor=via_request, env=dict(_ENV)).start_session(cwd="/w").turn(
        TurnRequest(prompt="x", mcp_servers=(server,))
    )
    via_spec = FakeExecutor(list(runs))
    OneShotTransport(provider, executor=via_spec, env=dict(_ENV)).open(
        _spec(cwd="/w", mcp_servers=(server,))
    ).turn(TurnRequest(prompt="x"), _ignore)
    assert via_spec.requests == via_request.requests
    assert any("srv" in part for part in via_spec.requests[0].argv)


def test_a_request_with_its_own_mcp_servers_overrides_the_spec() -> None:
    spec_server = StdioMcpServer(name="fromspec", command="python")
    own = StdioMcpServer(name="fromrequest", command="python")
    executor = FakeExecutor(scripted_turn("codex", text="t", session_id="s"))
    transport = OneShotTransport("codex", executor=executor, env=dict(_ENV))
    transport.open(_spec(cwd="/w", mcp_servers=(spec_server,))).turn(
        TurnRequest(prompt="x", mcp_servers=(own,)), _ignore
    )
    argv = " ".join(executor.requests[0].argv)
    assert "fromrequest" in argv
    assert "fromspec" not in argv


def test_spec_reasoning_effort_applies_when_the_request_has_none() -> None:
    runs = [scripted_turn("codex", text="t", session_id="s")]
    explicit = FakeExecutor(list(runs))
    CliAgent("codex", executor=explicit, env=dict(_ENV)).start_session(cwd="/w").turn(
        TurnRequest(prompt="x", reasoning_effort="high")
    )
    via_spec = FakeExecutor(list(runs))
    OneShotTransport("codex", executor=via_spec, env=dict(_ENV)).open(
        _spec(cwd="/w", reasoning_effort="high")
    ).turn(TurnRequest(prompt="x"), _ignore)
    assert via_spec.requests == explicit.requests


def test_the_scopes_and_model_of_the_spec_reach_the_command() -> None:
    runs = [scripted_turn("claude", text="t", session_id="s")]
    legacy = FakeExecutor(list(runs))
    CliAgent("claude", model="m1", executor=legacy, env=dict(_ENV)).start_session(
        cwd="/w", skill_scope=SkillScope.PROJECT
    ).turn(TurnRequest(prompt="x"))
    new = FakeExecutor(list(runs))
    OneShotTransport("claude", executor=new, env=dict(_ENV)).open(
        _spec(cwd="/w", model="m1", skill_scope=SkillScope.PROJECT)
    ).turn(TurnRequest(prompt="x"), _ignore)
    assert new.requests == legacy.requests


def test_the_binary_is_checked_once_however_many_conversations_open() -> None:
    executor = FakeExecutor(FakeRun())
    transport = OneShotTransport("claude", executor=executor, env=dict(_ENV))
    transport.open(_spec())
    transport.open(_spec())
    assert len(executor.checked) == 1


def test_a_permission_mode_the_profile_lacks_is_rejected() -> None:
    transport = OneShotTransport("claude", executor=FakeExecutor(FakeRun()), env=dict(_ENV))
    with pytest.raises(ProviderCapabilityError, match="read_only"):
        transport.open(_spec(permissions=NativePermissions.read_only()))


def test_reasoning_effort_the_provider_lacks_is_rejected_at_open() -> None:
    name = next(
        (
            n
            for n in _PROVIDERS
            if not OneShotTransport(
                n, executor=FakeExecutor(FakeRun()), env=_ENV
            ).profile.supports_reasoning_effort
        ),
        None,
    )
    if name is None:
        pytest.skip("every shipped provider supports reasoning effort")
    transport = OneShotTransport(name, executor=FakeExecutor(FakeRun()), env=dict(_ENV))
    with pytest.raises(ProviderCapabilityError, match="reasoning"):
        transport.open(_spec(reasoning_effort="high"))


def test_a_closed_conversation_refuses_turns() -> None:
    executor = FakeExecutor(scripted_turn("claude", text="t", session_id="s"))
    conversation = OneShotTransport("claude", executor=executor, env=dict(_ENV)).open(_spec())
    conversation.close()
    with pytest.raises(SessionStateError):
        conversation.turn(TurnRequest(prompt="x"), _ignore)


def test_events_stop_flowing_to_a_turn_once_it_ends() -> None:
    executor = FakeExecutor(
        [
            scripted_turn("claude", text="one", session_id="s"),
            scripted_turn("claude", text="two", session_id="s"),
        ]
    )
    conversation = OneShotTransport("claude", executor=executor, env=dict(_ENV)).open(_spec())
    first, second = RecordingEventHandler(), RecordingEventHandler()
    conversation.turn(TurnRequest(prompt="a"), first.on_event)
    count = len(first.events)
    conversation.turn(TurnRequest(prompt="b"), second.on_event)
    assert len(first.events) == count
    assert second.events


class _BlockingExecutor:
    """An executor whose command runs until it is terminated, then exits like a killed process."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self._stopped = threading.Event()
        self._inner = FakeExecutor(FakeRun())

    def find_binary(self, name: str, env: Mapping[str, str]) -> str:
        return self._inner.find_binary(name, env)

    def check_binary(self, path: str, env: Mapping[str, str], *, timeout: float) -> None:
        self._inner.check_binary(path, env, timeout=timeout)

    def run(self, request: CommandRequest, sink: CommandStreamSink) -> CommandResult:
        del request
        stopped = self._stopped

        class Handle(FakeCommandHandle):
            def terminate(self) -> None:
                super().terminate()
                stopped.set()

            def kill(self) -> None:
                super().kill()
                stopped.set()

        sink.started(Handle())
        self.started.set()
        stopped.wait()
        return CommandResult(returncode=-15, stdout="", stderr="")


def test_interrupting_a_running_turn_returns_an_interrupted_result_and_keeps_the_conversation() -> (
    None
):
    executor = _BlockingExecutor()
    transport = OneShotTransport("claude", executor=executor, env=dict(_ENV))  # type: ignore[arg-type]
    conversation = transport.open(_spec(resume_id="s-1"))
    events = RecordingEventHandler()
    outcome: list[TurnResult] = []
    worker = threading.Thread(
        target=lambda: outcome.append(
            conversation.turn(TurnRequest(prompt="long"), events.on_event)
        )
    )
    worker.start()
    executor.started.wait()
    conversation.interrupt()
    worker.join(30)
    assert not worker.is_alive()
    assert outcome[0].interrupted is True
    assert events.of_type(TurnInterrupted)
    assert conversation.conversation_id == "s-1"


def test_a_refused_resume_is_not_mistaken_for_an_earlier_interrupt() -> None:
    executor = FakeExecutor(scripted_resume_failure("claude", session_id="s-1"))
    conversation = OneShotTransport("claude", executor=executor, env=dict(_ENV)).open(
        _spec(resume_id="s-1")
    )
    conversation.interrupt()
    with pytest.raises(SessionResumeError):
        conversation.turn(TurnRequest(prompt="x"), _ignore)


# -- contract suites ------------------------------------------------------------


def _one_shot() -> OneShotTransport:
    executor = FakeExecutor(scripted_turn("claude", text="ok", session_id="s-1"))
    return OneShotTransport("claude", executor=executor, env=dict(_ENV))


class TestOneShotTransportContract(TransportContract):
    def make_transport(self) -> Transport:
        return _one_shot()


class TestOneShotConversationContract(ConversationContract):
    def make_conversation(self) -> Conversation:
        return _one_shot().open(self.spec())

    @staticmethod
    def spec() -> ConversationSpec:
        return _spec()


class TestInMemoryCheckpointStoreContract(CheckpointStoreContract):
    def make_store(self) -> CheckpointStore:
        from agentshim import InMemoryCheckpointStore  # noqa: PLC0415

        return InMemoryCheckpointStore()


def test_a_confined_agent_runs_inside_the_confinement_and_maps_paths() -> None:
    confinement = FakeConfinement(path_map={"/host": "/ctr"}, env={"IN_CONFINEMENT": "1"})
    executor = FakeExecutor(scripted_turn("codex", text="t", session_id="s"))
    agent = Agent(
        "codex",
        executor=executor,
        confinement=confinement,
        permissions=NativePermissions.bypass(),
        approvals=ApprovalPolicy.DENY,
    )
    server = StdioMcpServer(name="srv", command="/host/bin/srv", args=("--flag", "/host/data"))
    session = agent.session("/host/ws", mcp_servers=(server,))
    session.run(session.prepare_turn(TurnRequest(prompt="x")))
    request = executor.requests[-1]
    assert request.argv[0] == "fake-confine"
    assert confinement.wraps[-1][1] == "/ctr/ws"
    assert dict(request.env) == {"IN_CONFINEMENT": "1"}
    joined = " ".join(request.argv)
    assert "/ctr/bin/srv" in joined
    assert "/ctr/data" in joined
    assert "/host" not in joined
    assert "--flag" in joined


def test_a_confined_agent_names_a_schema_directory_as_the_agent_sees_it(tmp_path: Path) -> None:
    confinement = FakeConfinement(path_map={str(tmp_path): "/ctr/schemas"})
    executor = FakeExecutor(scripted_turn("codex", text="{}", session_id="s"))
    agent = Agent(
        "codex",
        executor=executor,
        confinement=confinement,
        permissions=NativePermissions.bypass(),
        approvals=ApprovalPolicy.DENY,
    )
    schema = OutputSchema(
        schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        host_dir=tmp_path,
    )
    session = agent.session("/host/ws")
    session.run(session.prepare_turn(TurnRequest(prompt="x", output_schema=schema)))
    joined = " ".join(executor.requests[-1].argv)
    assert "/ctr/schemas/" in joined
    assert str(tmp_path) not in joined


def test_confinement_and_env_cannot_both_be_given() -> None:
    with pytest.raises(ValueError, match="confinement"):
        Agent(
            "claude",
            executor=FakeExecutor(FakeRun()),
            confinement=FakeConfinement(),
            env={"A": "b"},
            permissions=NativePermissions.bypass(),
            approvals=ApprovalPolicy.DENY,
        )
