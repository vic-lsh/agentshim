"""``HostCommandExecutor`` against real subprocesses."""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from agentshim import (
    CliCheckError,
    CliNotFoundError,
    CliTimeoutError,
    CommandHandle,
    CommandRequest,
    HostCommandExecutor,
)


class RecordingSink:
    def __init__(self) -> None:
        self.handles: list[CommandHandle] = []
        self.stdout_lines: list[str] = []
        self.stderr_lines: list[str] = []
        self.threads: set[int] = set()

    def started(self, handle: CommandHandle) -> None:
        self.handles.append(handle)
        self.threads.add(threading.get_ident())

    def stdout(self, line: str) -> None:
        self.stdout_lines.append(line)
        self.threads.add(threading.get_ident())

    def stderr(self, line: str) -> None:
        self.stderr_lines.append(line)
        self.threads.add(threading.get_ident())


def _request(
    code: str, *, stdin: str | None = "", timeout: float | None = 30, cwd: str | None = None
) -> CommandRequest:
    return CommandRequest(
        argv=[sys.executable, "-c", code],
        stdin=stdin,
        cwd=cwd,
        env=os.environ.copy(),
        timeout=timeout,
    )


class TestRun:
    def test_streams_stdout_stderr_stdin_cwd_and_env(self, tmp_path: Path) -> None:
        sink = RecordingSink()
        env = os.environ.copy()
        env["AGENTSHIM_EXECUTOR_TEST"] = "env-value"
        code = (
            "import os, sys\n"
            "print('cwd=' + os.path.basename(os.getcwd()))\n"
            "print('env=' + os.environ['AGENTSHIM_EXECUTOR_TEST'])\n"
            "print('stdin=' + sys.stdin.read().strip())\n"
            "print('stderr-line', file=sys.stderr)\n"
        )
        request = CommandRequest(
            argv=[sys.executable, "-c", code],
            stdin="prompt text\n",
            cwd=str(tmp_path),
            env=env,
            timeout=30,
        )

        result = HostCommandExecutor().run(request, sink)

        assert result.returncode == 0
        assert len(sink.handles) == 1
        assert sink.stdout_lines == [
            f"cwd={tmp_path.name}\n",
            "env=env-value\n",
            "stdin=prompt text\n",
        ]
        assert sink.stderr_lines == ["stderr-line\n"]
        assert result.stdout == "".join(sink.stdout_lines)
        assert result.stderr == "".join(sink.stderr_lines)

    def test_callbacks_run_on_the_calling_thread(self) -> None:
        sink = RecordingSink()
        code = "import sys\nprint('a')\nprint('b', file=sys.stderr)\n"
        HostCommandExecutor().run(_request(code), sink)
        assert sink.threads == {threading.get_ident()}

    def test_nonzero_exit_is_reported_not_raised(self) -> None:
        result = HostCommandExecutor().run(_request("import sys; sys.exit(7)"), RecordingSink())
        assert result.returncode == 7

    def test_a_one_mib_prompt_does_not_deadlock(self) -> None:
        """A prompt larger than the pipe buffer needs a stdin writer thread.

        ``cat`` echoes it back while we are still writing, so a single-threaded
        write-then-read would block on a full stdout pipe forever.
        """
        binary = shutil.which("cat")
        if binary is None:
            pytest.skip("cat is not available")
        prompt = ("x" * 1023 + "\n") * 1024
        assert len(prompt) >= 1024 * 1024
        sink = RecordingSink()

        result = HostCommandExecutor().run(
            CommandRequest(
                argv=[binary], stdin=prompt, cwd=None, env=os.environ.copy(), timeout=60
            ),
            sink,
        )

        assert result.returncode == 0
        assert result.stdout == prompt

    def test_concurrent_runs_do_not_interleave(self) -> None:
        line_count = 100

        def run_one(index: int) -> int:
            sink = RecordingSink()
            code = (
                "import sys\n"
                "data = sys.stdin.read().strip()\n"
                f"for n in range({line_count}): print(f'out-{{n}}')\n"
                f"for n in range({line_count}): print(f'err-{{n}}', file=sys.stderr)\n"
                "print('stdin=' + data)\n"
            )
            result = HostCommandExecutor().run(_request(code, stdin=f"prompt-{index}\n"), sink)
            assert result.returncode == 0
            assert len(sink.stdout_lines) == line_count + 1
            assert len(sink.stderr_lines) == line_count
            assert sink.stdout_lines[-1] == f"stdin=prompt-{index}\n"
            return index

        with ThreadPoolExecutor(max_workers=6) as pool:
            assert sorted(pool.map(run_one, range(12))) == list(range(12))

    def test_no_timeout_means_no_limit(self) -> None:
        result = HostCommandExecutor().run(
            _request("import time; time.sleep(0.3); print('done')", timeout=None),
            RecordingSink(),
        )
        assert result.returncode == 0
        assert result.stdout == "done\n"


class TestTimeout:
    def test_timeout_raises_cli_timeout_error_and_kills_the_process(self) -> None:
        sink = RecordingSink()
        request = _request("import time; time.sleep(30)", timeout=0.3)

        with pytest.raises(CliTimeoutError) as excinfo:
            HostCommandExecutor().run(request, sink)

        assert excinfo.value.timeout == 0.3
        assert excinfo.value.argv[0] == sys.executable
        assert len(sink.handles) == 1
        handle = sink.handles[0]
        assert handle.process.poll() is not None  # pyright: ignore[reportAttributeAccessIssue]

    def test_timeout_fires_even_while_output_keeps_arriving(self) -> None:
        code = "import time, sys\nwhile True:\n    print('tick'); sys.stdout.flush(); time.sleep(0.01)\n"
        with pytest.raises(CliTimeoutError):
            HostCommandExecutor().run(_request(code, timeout=0.4), RecordingSink())

    def test_timeout_kills_the_whole_process_group(self) -> None:
        code = (
            "import subprocess, sys, time\n"
            f"child = subprocess.Popen([{sys.executable!r}, '-c', 'import time; time.sleep(30)'])\n"
            "print(child.pid, flush=True)\n"
            "time.sleep(30)\n"
        )
        sink = RecordingSink()
        with pytest.raises(CliTimeoutError):
            HostCommandExecutor().run(_request(code, timeout=1.0), sink)

        child_pid = int(sink.stdout_lines[0].strip())
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except (ProcessLookupError, PermissionError):
                return
            time.sleep(0.05)
        pytest.fail("the grandchild survived the process-group kill")


class TestCancel:
    def test_terminate_from_another_thread_stops_a_real_process(self) -> None:
        sink = RecordingSink()
        code = "import time, sys\nprint('ready', flush=True)\ntime.sleep(30)\n"
        executor = HostCommandExecutor()

        def stop() -> None:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and not sink.handles:
                time.sleep(0.01)
            sink.handles[0].kill()

        stopper = threading.Thread(target=stop)
        stopper.start()
        try:
            result = executor.run(_request(code, timeout=20), sink)
        finally:
            stopper.join()

        assert result.returncode != 0


class TestSinkFailures:
    def test_a_sink_exception_propagates_after_the_process_is_killed(self) -> None:
        class Exploding(RecordingSink):
            def stdout(self, line: str) -> None:
                super().stdout(line)
                message = "sink failed"
                raise RuntimeError(message)

        sink = Exploding()
        code = "import time, sys\nprint('one', flush=True)\ntime.sleep(30)\n"
        with pytest.raises(RuntimeError, match="sink failed"):
            HostCommandExecutor().run(_request(code, timeout=20), sink)
        assert sink.handles[0].process.poll() is not None  # pyright: ignore[reportAttributeAccessIssue]


class TestBinaryLookup:
    def test_find_binary_prefers_the_env_path(self, tmp_path: Path) -> None:
        fake = tmp_path / "claude"
        fake.write_text("#!/bin/sh\n")
        fake.chmod(0o755)
        assert HostCommandExecutor().find_binary("claude", {"PATH": str(tmp_path)}) == str(fake)

    def test_missing_binary_raises_cli_not_found(self) -> None:
        with pytest.raises(CliNotFoundError) as excinfo:
            HostCommandExecutor().find_binary("agentshim-does-not-exist", {"PATH": "/nonexistent"})
        assert excinfo.value.binary == "agentshim-does-not-exist"

    def test_check_binary_accepts_a_working_binary(self, tmp_path: Path) -> None:
        script = tmp_path / "ok"
        script.write_text("#!/bin/sh\necho usage\n")
        script.chmod(0o755)
        HostCommandExecutor().check_binary(str(script), os.environ.copy(), timeout=10)

    def test_check_binary_rejects_a_failing_binary(self, tmp_path: Path) -> None:
        script = tmp_path / "broken"
        script.write_text("#!/bin/sh\necho nope >&2\nexit 3\n")
        script.chmod(0o755)
        with pytest.raises(CliCheckError, match="exited with code 3"):
            HostCommandExecutor().check_binary(str(script), os.environ.copy(), timeout=10)

    def test_check_binary_times_out(self, tmp_path: Path) -> None:
        script = tmp_path / "hangs"
        script.write_text("#!/bin/sh\nsleep 30\n")
        script.chmod(0o755)
        with pytest.raises(CliCheckError, match="did not respond"):
            HostCommandExecutor().check_binary(str(script), os.environ.copy(), timeout=0.5)

    def test_check_binary_does_not_inherit_the_callers_stdin(self, tmp_path: Path) -> None:
        """The probe must never read a TTY: a stopped child deadlocks the parent."""
        script = tmp_path / "reads-stdin"
        script.write_text("#!/bin/sh\ncat > /dev/null\n")
        script.chmod(0o755)
        HostCommandExecutor().check_binary(str(script), os.environ.copy(), timeout=10)
