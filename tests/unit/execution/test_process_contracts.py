"""The ``Process`` contract, run against the real host shell and the fake."""

from __future__ import annotations

import os
import sys

from agentshim import (
    HostCommandExecutor,
    Process,
    ProcessExited,
    SpawnRequest,
    StderrLine,
    StdoutLine,
)
from agentshim.testing import EchoPeer, FakeExecutor, FakeProcess, SilentPeer
from agentshim.testing.contracts import READ_TIMEOUT_S, ProcessContract

_ECHO = "import sys\nfor line in sys.stdin:\n    sys.stdout.write(line)\n    sys.stdout.flush()\n"
_SILENT = "import time\ntime.sleep(600)\n"


def _spawn(code: str) -> Process:
    request = SpawnRequest(argv=[sys.executable, "-u", "-c", code], cwd=None, env=dict(os.environ))
    return HostCommandExecutor().spawn(request)


class TestHostProcess(ProcessContract):
    def make_echo(self) -> Process:
        return _spawn(_ECHO)

    def make_silent(self) -> Process:
        return _spawn(_SILENT)

    def test_stderr_and_stdout_are_distinguished_and_exit_is_last(self) -> None:
        process = _spawn(
            "import sys\nsys.stderr.write('e\\n')\nsys.stderr.flush()\nsys.stdout.write('o\\n')\n"
        )
        self._made().append(process)
        items = self._drain(process)
        assert items[-1] == ProcessExited(0)
        assert sorted(items[:-1], key=repr) == [StderrLine("e\n"), StdoutLine("o\n")]

    def test_a_final_line_without_a_newline_is_delivered(self) -> None:
        process = _spawn("import sys\nsys.stdout.write('tail')\n")
        self._made().append(process)
        assert self._drain(process) == [StdoutLine("tail"), ProcessExited(0)]

    def test_the_exit_code_is_reported(self) -> None:
        process = _spawn("raise SystemExit(7)")
        self._made().append(process)
        assert self._drain(process) == [ProcessExited(7)]
        assert process.wait(READ_TIMEOUT_S) == 7

    def test_cwd_and_env_reach_the_child(self) -> None:
        request = SpawnRequest(
            argv=[sys.executable, "-c", "import os\nprint(os.getcwd(), os.environ['AS_PROBE'])"],
            cwd=os.path.realpath(os.sep),
            env={**os.environ, "AS_PROBE": "yes"},
        )
        process = HostCommandExecutor().spawn(request)
        self._made().append(process)
        first = self._drain(process)[0]
        assert first == StdoutLine(f"{os.path.realpath(os.sep)} yes\n")

    def test_kill_takes_down_the_whole_process_group(self) -> None:
        # The parent starts a grandchild that inherits stdout; the stream only
        # reaches EOF (and ProcessExited only arrives) once the group is dead.
        code = (
            "import subprocess, sys, time\n"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
            "time.sleep(600)\n"
        )
        process = _spawn(code)
        self._made().append(process)
        process.kill()
        assert isinstance(self._drain(process)[-1], ProcessExited)


class TestFakeProcess(ProcessContract):
    def make_echo(self) -> Process:
        return FakeProcess(EchoPeer())

    def make_silent(self) -> Process:
        return FakeProcess(SilentPeer())


class TestFakeExecutorSpawn(ProcessContract):
    def make_echo(self) -> Process:
        return FakeExecutor([], peers=lambda _request: EchoPeer()).spawn(
            SpawnRequest(argv=["cat"], cwd=None, env={})
        )

    def make_silent(self) -> Process:
        return FakeExecutor([], peers=lambda _request: SilentPeer()).spawn(
            SpawnRequest(argv=["sleep"], cwd=None, env={})
        )


def test_a_zero_timeout_polls_and_returns_output_that_is_already_queued() -> None:
    process = _spawn("import sys\nsys.stdout.write('ready\\n')\n")
    try:
        # Once the child has exited, the line is in (or about to reach) the
        # queue; a poll must hand it over instead of reporting a timeout.
        assert process.wait(READ_TIMEOUT_S) == 0
        item = None
        for _ in range(2_000_000):
            item = process.next_output(0)
            if item is not None:
                break
        assert item == StdoutLine("ready\n")
    finally:
        process.kill()
        process.wait(READ_TIMEOUT_S)
