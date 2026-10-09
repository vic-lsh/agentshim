"""Contract suites for agentshim's interfaces.

Each class is a base with abstract factory methods and ``test_*`` methods. A
consumer subclasses it as ``Test<Impl>``, implements the factories, and pytest
collects the inherited tests against its implementation::

    class TestMyProcess(ProcessContract):
        def make_echo(self) -> Process: ...
        def make_silent(self) -> Process: ...

This module must not import pytest, so it is safe to import anywhere.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from agentshim.core.checkpoints import Checkpoint
from agentshim.core.clock import StopSignal
from agentshim.core.conversation import ConversationSpec, SteerableConversation
from agentshim.core.errors import (
    AgentShimError,
    NoRunningTurnError,
    ProcessClosedError,
    ProviderCapabilityError,
)
from agentshim.core.events import SteerConsumed, SteerDelivered
from agentshim.core.permissions import ApprovalPolicy, NativeMode, NativePermissions
from agentshim.core.turn import TurnRequest
from agentshim.execution.process import ProcessExited, StderrLine, StdoutLine

if TYPE_CHECKING:
    from collections.abc import Mapping

    from agentshim.core.checkpoints import CheckpointStore
    from agentshim.core.clock import Clock
    from agentshim.core.conversation import Conversation, Transport
    from agentshim.core.events import AgentEvent
    from agentshim.core.turn import TurnResult
    from agentshim.execution.confinement import Confinement
    from agentshim.execution.process import Process, ProcessOutput

#: Generous bound for output a live process is about to produce. A passing run
#: never waits this long; it only bounds how long a broken one can hang.
READ_TIMEOUT_S = 30.0
#: Short wait for a process that is known to be silent.
SILENCE_PROBE_S = 0.05


def _ignore(event: AgentEvent) -> None:
    """Event callback that drops everything."""


class ProcessContract:
    """Behavior every ``Process`` has.

    ``make_echo`` returns a started process that echoes each stdin line to
    stdout (newline kept) and exits 0 when stdin closes. ``make_silent``
    returns a started process that never writes and never exits by itself.
    Both must be fresh per call. Processes made here are stopped in
    ``teardown_method``.
    """

    def make_echo(self) -> Process:
        """Return a started echo process."""
        raise NotImplementedError

    def make_silent(self) -> Process:
        """Return a started process that never writes."""
        raise NotImplementedError

    def _made(self) -> list[Process]:
        if not hasattr(self, "_started"):
            self._started: list[Process] = []
        return self._started

    def _echo(self) -> Process:
        process = self.make_echo()
        self._made().append(process)
        return process

    def _silent(self) -> Process:
        process = self.make_silent()
        self._made().append(process)
        return process

    def teardown_method(self) -> None:
        """Stop every process the test started."""
        for process in self._made():
            process.kill()
        self._made().clear()

    @staticmethod
    def _drain(process: Process) -> list[ProcessOutput]:
        """Read until ``ProcessExited`` (inclusive)."""
        items: list[ProcessOutput] = []
        while True:
            item = process.next_output(READ_TIMEOUT_S)
            assert item is not None, f"no output within {READ_TIMEOUT_S}s; got {items}"
            items.append(item)
            if isinstance(item, ProcessExited):
                return items

    def test_a_write_is_read_back(self) -> None:
        process = self._echo()
        process.write("hello\n")
        assert process.next_output(READ_TIMEOUT_S) == StdoutLine("hello\n")

    def test_lines_come_back_in_order_then_the_exit(self) -> None:
        process = self._echo()
        process.write("a\n")
        process.write("b\n")
        process.close_stdin()
        assert self._drain(process) == [StdoutLine("a\n"), StdoutLine("b\n"), ProcessExited(0)]

    def test_closing_stdin_ends_the_process(self) -> None:
        process = self._echo()
        process.close_stdin()
        process.close_stdin()
        assert self._drain(process)[-1] == ProcessExited(0)
        assert process.wait(READ_TIMEOUT_S) == 0

    def test_kill_ends_the_process_with_a_failure_code(self) -> None:
        process = self._silent()
        process.kill()
        process.kill()
        exited = self._drain(process)[-1]
        assert isinstance(exited, ProcessExited)
        assert exited.returncode != 0
        assert process.wait(READ_TIMEOUT_S) == exited.returncode

    def test_terminate_ends_the_process(self) -> None:
        process = self._silent()
        process.terminate()
        process.terminate()
        exited = self._drain(process)[-1]
        assert isinstance(exited, ProcessExited)
        assert exited.returncode != 0

    def test_output_after_exit_keeps_reporting_the_exit(self) -> None:
        process = self._echo()
        process.close_stdin()
        exited = self._drain(process)[-1]
        for _ in range(3):
            assert process.next_output(0.0) == exited

    def test_silence_times_out_with_none(self) -> None:
        process = self._silent()
        assert process.next_output(SILENCE_PROBE_S) is None
        assert process.wait(0.0) is None

    def test_writing_after_close_raises(self) -> None:
        process = self._echo()
        process.close_stdin()
        try:
            process.write("late\n")
        except ProcessClosedError:
            return
        msg = "write after close_stdin did not raise ProcessClosedError"
        raise AssertionError(msg)

    def test_writing_after_exit_raises(self) -> None:
        process = self._silent()
        process.kill()
        self._drain(process)
        try:
            process.write("late\n")
        except ProcessClosedError:
            return
        msg = "write after exit did not raise ProcessClosedError"
        raise AssertionError(msg)

    def test_only_known_output_kinds_are_produced(self) -> None:
        process = self._echo()
        process.write("x\n")
        process.close_stdin()
        for item in self._drain(process):
            assert isinstance(item, (StdoutLine, StderrLine, ProcessExited))


class ConfinementContract:
    """Behavior every ``Confinement`` has."""

    def make_confinement(self) -> Confinement:
        """Return a fresh confinement."""
        raise NotImplementedError

    def host_paths(self) -> list[str]:
        """Host paths to probe ``agent_path`` with; override to cover mapped ones."""
        return ["/", "/work", "/work/a/b", "/srv/x//y/./z"]

    def test_wrap_keeps_the_argv_as_its_tail(self) -> None:
        confinement = self.make_confinement()
        argv = ["prog", "--flag", "a value with spaces", "", "-e", "K=V"]
        for cwd in (None, "/work", "/work/a/b"):
            wrapped = confinement.wrap(argv, cwd)
            assert wrapped[len(wrapped) - len(argv) :] == argv
            assert all(isinstance(part, str) for part in wrapped)

    def test_wrap_does_not_mutate_its_input(self) -> None:
        confinement = self.make_confinement()
        argv = ["prog", "x"]
        confinement.wrap(argv, "/work")
        assert argv == ["prog", "x"]

    def test_agent_path_is_stable_on_its_own_output(self) -> None:
        confinement = self.make_confinement()
        for path in self.host_paths():
            once = confinement.agent_path(path)
            assert confinement.agent_path(once) == once

    def test_agent_path_accepts_path_objects(self) -> None:
        confinement = self.make_confinement()
        for path in self.host_paths():
            assert confinement.agent_path(Path(path)) == confinement.agent_path(path)

    def test_env_is_a_string_mapping(self) -> None:
        env: Mapping[str, str] = self.make_confinement().env
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in env.items())

    def test_reap_is_idempotent(self) -> None:
        confinement = self.make_confinement()
        confinement.reap()
        confinement.reap()


class ClockContract:
    """Behavior every ``Clock`` has, that does not depend on real waiting."""

    def make_clock(self) -> Clock:
        """Return a fresh clock."""
        raise NotImplementedError

    def test_monotonic_never_goes_backwards(self) -> None:
        clock = self.make_clock()
        readings = [clock.monotonic() for _ in range(5)]
        assert readings == sorted(readings)

    def test_a_wait_on_a_set_stop_returns_true_at_once(self) -> None:
        clock = self.make_clock()
        stop = StopSignal()
        stop.set()
        assert clock.wait(1000.0, stop) is True

    def test_a_zero_wait_completes_and_reports_not_stopped(self) -> None:
        clock = self.make_clock()
        assert clock.wait(0.0) is False
        assert clock.wait(0.0, StopSignal()) is False

    def test_time_does_not_go_backwards_across_a_wait(self) -> None:
        clock = self.make_clock()
        before = clock.monotonic()
        clock.wait(0.0)
        assert clock.monotonic() >= before

    def test_waiting_does_not_set_the_stop(self) -> None:
        clock = self.make_clock()
        stop = StopSignal()
        clock.wait(0.0, stop)
        assert not stop.is_set()


class ConversationContract:
    """Behavior every ``Conversation`` has.

    ``make_conversation`` returns a fresh, open conversation whose every turn
    succeeds. Conversations made here are closed in ``teardown_method``.
    """

    def make_conversation(self) -> Conversation:
        """Return a fresh conversation that answers every turn successfully."""
        raise NotImplementedError

    def _made(self) -> list[Conversation]:
        if not hasattr(self, "_conversations"):
            self._conversations: list[Conversation] = []
        return self._conversations

    def _conversation(self) -> Conversation:
        conversation = self.make_conversation()
        self._made().append(conversation)
        return conversation

    def teardown_method(self) -> None:
        """Close every conversation the test opened."""
        for conversation in self._made():
            conversation.close()
        self._made().clear()

    def test_a_turn_returns_a_result_and_names_the_conversation(self) -> None:
        conversation = self._conversation()
        result = conversation.turn(TurnRequest(prompt="hello"), _ignore)
        assert isinstance(result.text, str)
        assert result.interrupted is False
        if result.session_id is not None:
            assert conversation.conversation_id == result.session_id

    def test_events_are_emitted_on_the_calling_thread(self) -> None:
        import threading  # noqa: PLC0415 - only this check needs it

        conversation = self._conversation()
        seen: list[tuple[int, AgentEvent]] = []
        conversation.turn(
            TurnRequest(prompt="hello"),
            lambda event: seen.append((threading.get_ident(), event)),
        )
        assert all(thread == threading.get_ident() for thread, _ in seen)

    def test_turns_follow_one_another_in_the_same_conversation(self) -> None:
        conversation = self._conversation()
        first = conversation.turn(TurnRequest(prompt="one"), _ignore)
        second = conversation.turn(TurnRequest(prompt="two"), _ignore)
        assert second.interrupted is False
        if first.session_id is not None:
            assert conversation.conversation_id is not None

    def test_interrupting_an_idle_conversation_does_nothing(self) -> None:
        conversation = self._conversation()
        conversation.interrupt()
        result = conversation.turn(TurnRequest(prompt="hello"), _ignore)
        assert result.interrupted is False

    def test_close_is_idempotent_and_interrupt_after_close_is_harmless(self) -> None:
        conversation = self._conversation()
        conversation.close()
        conversation.close()
        conversation.interrupt()

    def test_a_turn_after_close_raises_an_agentshim_error(self) -> None:
        conversation = self._conversation()
        conversation.close()
        try:
            conversation.turn(TurnRequest(prompt="late"), _ignore)
        except AgentShimError:
            return
        msg = "a turn on a closed conversation did not raise AgentShimError"
        raise AssertionError(msg)


class SteerableConversationContract(ConversationContract):
    """Behavior every ``SteerableConversation`` has, on top of a conversation's.

    The checks steer from inside the event handler, which runs on the turn's
    thread while the turn is running, at the first event the conversation will
    take a steer at. So a turn of the conversation ``make_conversation``
    returns must stay steerable for at least one event, and must accept the
    message into the turn (a fake provider that finishes at once would not).
    """

    def make_steerable_conversation(self) -> Conversation:
        """A fresh conversation whose first turn stays steerable until it is steered.

        Defaults to ``make_conversation``; override it when the turn a steer
        needs (one that waits for the message) would hang the plain checks.
        """
        return self.make_conversation()

    def _steerable(self) -> Conversation:
        conversation = self.make_steerable_conversation()
        self._made().append(conversation)
        return conversation

    def _steered_turn(
        self, conversation: Conversation, text: str
    ) -> tuple[TurnResult, list[AgentEvent]]:
        assert isinstance(conversation, SteerableConversation)
        seen: list[AgentEvent] = []
        sent: list[str] = []

        def on_event(event: AgentEvent) -> None:
            seen.append(event)
            if sent:
                return
            try:
                conversation.steer(text)
            except NoRunningTurnError:
                return  # a provider may not take a message until its turn has started
            sent.append(text)

        result = conversation.turn(TurnRequest(prompt="hello"), on_event)
        assert sent, "the turn emitted no event to steer from"
        return result, seen

    def test_steering_an_idle_conversation_raises(self) -> None:
        conversation = self._steerable()
        assert isinstance(conversation, SteerableConversation)
        try:
            conversation.steer("too early")
        except NoRunningTurnError:
            return
        msg = "steer with no turn running did not raise NoRunningTurnError"
        raise AssertionError(msg)

    def test_a_steer_during_a_turn_leaves_one_ordinary_result(self) -> None:
        result, _ = self._steered_turn(self._steerable(), "change course")
        assert result.interrupted is False

    def test_steer_events_name_the_text_and_delivery_comes_first(self) -> None:
        _, seen = self._steered_turn(self._steerable(), "change course")
        steer_events = [e for e in seen if isinstance(e, (SteerDelivered, SteerConsumed))]
        assert steer_events, "a steer that was accepted reported nothing"
        assert all(e.text == "change course" for e in steer_events)
        assert isinstance(steer_events[0], SteerDelivered)

    def test_a_steer_does_not_leak_into_the_next_turn(self) -> None:
        conversation = self._steerable()
        self._steered_turn(conversation, "change course")
        later: list[AgentEvent] = []
        result = conversation.turn(TurnRequest(prompt="next"), later.append)
        assert result.interrupted is False
        assert not [e for e in later if isinstance(e, (SteerDelivered, SteerConsumed))]

    def test_steering_after_the_turn_ended_raises(self) -> None:
        conversation = self._steerable()
        self._steered_turn(conversation, "change course")
        assert isinstance(conversation, SteerableConversation)
        try:
            conversation.steer("late")
        except NoRunningTurnError:
            return
        msg = "steer after the turn ended did not raise NoRunningTurnError"
        raise AssertionError(msg)


class TransportContract:
    """Behavior every ``Transport`` has.

    ``make_transport`` returns a transport whose conversations answer every
    turn successfully; ``make_spec`` a spec it accepts (BYPASS permissions).
    """

    def make_transport(self) -> Transport:
        """Return a fresh transport."""
        raise NotImplementedError

    def make_spec(self) -> ConversationSpec:
        """Return a spec the transport accepts."""
        return ConversationSpec(
            cwd="/work",
            model=None,
            permissions=NativePermissions.bypass(),
            approvals=ApprovalPolicy.DENY,
        )

    def test_open_returns_a_conversation_that_runs_turns(self) -> None:
        transport = self.make_transport()
        conversation = transport.open(self.make_spec())
        try:
            result = conversation.turn(TurnRequest(prompt="hello"), _ignore)
            assert isinstance(result.text, str)
        finally:
            conversation.close()

    def test_every_open_makes_an_independent_conversation(self) -> None:
        transport = self.make_transport()
        first = transport.open(self.make_spec())
        second = transport.open(self.make_spec())
        try:
            assert first is not second
            first.close()
            result = second.turn(TurnRequest(prompt="hello"), _ignore)
            assert isinstance(result.text, str)
        finally:
            first.close()
            second.close()

    def test_the_profile_declares_bypass(self) -> None:
        profile = self.make_transport().profile
        assert NativeMode.BYPASS in profile.native_permission_modes

    def test_a_mode_outside_the_profile_is_rejected(self) -> None:
        transport = self.make_transport()
        modes = transport.profile.native_permission_modes
        spec = self.make_spec()
        for permissions in (NativePermissions.read_only(), NativePermissions.workspace_write()):
            if permissions.mode in modes:
                continue
            try:
                transport.open(replace(spec, permissions=permissions))
            except ProviderCapabilityError:
                continue
            msg = f"open accepted unsupported mode {permissions.mode.value}"
            raise AssertionError(msg)


class CheckpointStoreContract:
    """Behavior every ``CheckpointStore`` has."""

    def make_store(self) -> CheckpointStore:
        """Return a fresh, empty store."""
        raise NotImplementedError

    def test_a_missing_key_loads_none(self) -> None:
        assert self.make_store().load("absent") is None

    def test_a_saved_checkpoint_loads_back(self) -> None:
        store = self.make_store()
        store.save("k", Checkpoint("conv-1"))
        assert store.load("k") == Checkpoint("conv-1")

    def test_a_save_replaces_the_previous_one(self) -> None:
        store = self.make_store()
        store.save("k", Checkpoint("conv-1"))
        store.save("k", Checkpoint("conv-2"))
        assert store.load("k") == Checkpoint("conv-2")

    def test_keys_are_independent(self) -> None:
        store = self.make_store()
        store.save("a", Checkpoint("conv-a"))
        store.save("b", Checkpoint("conv-b"))
        store.clear("a")
        assert store.load("a") is None
        assert store.load("b") == Checkpoint("conv-b")

    def test_clearing_an_empty_key_is_not_an_error(self) -> None:
        store = self.make_store()
        store.clear("never-saved")
        store.clear("never-saved")
