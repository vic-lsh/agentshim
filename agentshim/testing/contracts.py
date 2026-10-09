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

from pathlib import Path
from typing import TYPE_CHECKING

from agentshim.core.clock import StopSignal
from agentshim.core.errors import ProcessClosedError
from agentshim.execution.process import ProcessExited, StderrLine, StdoutLine

if TYPE_CHECKING:
    from collections.abc import Mapping

    from agentshim.core.clock import Clock
    from agentshim.execution.confinement import Confinement
    from agentshim.execution.process import Process, ProcessOutput

#: Generous bound for output a live process is about to produce. A passing run
#: never waits this long; it only bounds how long a broken one can hang.
READ_TIMEOUT_S = 30.0
#: Short wait for a process that is known to be silent.
SILENCE_PROBE_S = 0.05


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
