from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from agentshim.cli_agent import CLICodingAgent
from agentshim.executor import (
    CommandHandle,
    CommandRequest,
    CommandResult,
    CommandStreamSink,
    HostCommandExecutor,
)


class FakeCommandHandle:
    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass


class RecordingSink:
    def __init__(self) -> None:
        self.handles: list[CommandHandle] = []
        self.stdout_lines: list[str] = []
        self.stderr_lines: list[str] = []

    def started(self, handle: CommandHandle) -> None:
        self.handles.append(handle)

    def stdout(self, line: str) -> None:
        self.stdout_lines.append(line)

    def stderr(self, line: str) -> None:
        self.stderr_lines.append(line)


class RecordingExecutor:
    def __init__(self, *, started_handle: CommandHandle | None = None):
        self.find_binary_calls: list[tuple[str, dict[str, str]]] = []
        self.check_binary_calls: list[tuple[str, dict[str, str], int]] = []
        self.run_calls: list[CommandRequest] = []
        self.started_handle = started_handle

    def find_binary(self, binary_name: str, env: dict[str, str]) -> str:
        self.find_binary_calls.append((binary_name, env))
        return f"/executor/bin/{binary_name}"

    def check_binary(
        self,
        binary_path: str,
        env: dict[str, str],
        *,
        timeout: int,
    ) -> None:
        self.check_binary_calls.append((binary_path, env, timeout))

    def run(
        self,
        request: CommandRequest,
        sink: CommandStreamSink,
    ) -> CommandResult:
        self.run_calls.append(request)
        if self.started_handle is not None:
            sink.started(self.started_handle)
        sink.stdout("executor output\n")
        sink.stderr("executor stderr\n")
        return CommandResult(returncode=0, stdout="executor output\n", stderr="executor stderr\n")


class DummyCodingAgent(CLICodingAgent):
    def __init__(self, *, executor: RecordingExecutor):
        super().__init__("dummy", executor=executor)

    def _get_command(self, prompt: str, resume_session_id: str | None = None) -> list[str]:
        return [self.binary_path, "run"]


def test_cli_agent_uses_injected_executor_for_lookup_check_and_run(monkeypatch):
    monkeypatch.setattr(
        "agentshim.cli_agent.get_interactive_env",
        lambda: {"PATH": "/custom/bin", "HOME": "/tmp"},
    )
    executor = RecordingExecutor()

    agent = DummyCodingAgent(executor=executor)
    result = agent.generate("hello", cwd="/workspace", timeout=123, silent=True)

    assert result == "executor output"
    assert executor.find_binary_calls == [
        ("dummy", {"PATH": "/custom/bin", "HOME": "/tmp"})
    ]
    assert executor.check_binary_calls == [
        ("/executor/bin/dummy", {"PATH": "/custom/bin", "HOME": "/tmp"}, 15)
    ]
    assert executor.run_calls == [
        CommandRequest(
            argv=["/executor/bin/dummy", "run"],
            stdin="hello",
            cwd="/workspace",
            env={"PATH": "/custom/bin", "HOME": "/tmp"},
            timeout=123,
        )
    ]


def test_cli_agent_forwards_started_command_handle(monkeypatch):
    monkeypatch.setattr(
        "agentshim.cli_agent.get_interactive_env",
        lambda: {"PATH": "/custom/bin"},
    )
    handle = FakeCommandHandle()
    executor = RecordingExecutor(started_handle=handle)
    observed: list[CommandHandle] = []

    agent = DummyCodingAgent(executor=executor)
    agent.generate("hello", silent=True, on_process_started=observed.append)

    assert observed == [handle]


def test_cli_agent_ignores_started_callback_errors(monkeypatch):
    monkeypatch.setattr(
        "agentshim.cli_agent.get_interactive_env",
        lambda: {"PATH": "/custom/bin"},
    )
    executor = RecordingExecutor(started_handle=FakeCommandHandle())

    def raise_on_started(handle: CommandHandle) -> None:
        raise RuntimeError("callback failed")

    agent = DummyCodingAgent(executor=executor)

    assert agent.generate("hello", silent=True, on_process_started=raise_on_started) == "executor output"


def test_cli_agent_surfaces_injected_executor_failures(monkeypatch):
    class FailingExecutor(RecordingExecutor):
        def run(self, *args: Any, **kwargs: Any) -> CommandResult:
            return CommandResult(returncode=7, stdout="", stderr="boom")

    monkeypatch.setattr(
        "agentshim.cli_agent.get_interactive_env",
        lambda: {"PATH": "/custom/bin"},
    )
    agent = DummyCodingAgent(executor=FailingExecutor())

    with pytest.raises(RuntimeError, match="dummy exited with code 7: boom"):
        agent.generate("hello", silent=True)


def test_host_executor_runs_real_command_and_streams_output(tmp_path):
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

    result = HostCommandExecutor().run(
        CommandRequest(
            argv=[sys.executable, "-c", code],
            stdin="prompt text\n",
            cwd=str(tmp_path),
            env=env,
            timeout=5,
        ),
        sink,
    )

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


def test_host_executor_handles_concurrent_real_streams():
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

        result = HostCommandExecutor().run(
            CommandRequest(
                argv=[sys.executable, "-c", code],
                stdin=f"prompt-{index}\n",
                cwd=None,
                env=os.environ.copy(),
                timeout=10,
            ),
            sink,
        )

        assert result.returncode == 0
        assert len(sink.handles) == 1
        assert len(sink.stdout_lines) == line_count + 1
        assert len(sink.stderr_lines) == line_count
        assert sink.stdout_lines[-1] == f"stdin=prompt-{index}\n"
        assert result.stdout == "".join(sink.stdout_lines)
        assert result.stderr == "".join(sink.stderr_lines)
        return index

    with ThreadPoolExecutor(max_workers=6) as executor:
        assert sorted(executor.map(run_one, range(18))) == list(range(18))


def test_host_executor_timeout_kills_real_process():
    sink = RecordingSink()
    request = CommandRequest(
        argv=[sys.executable, "-c", "import time; time.sleep(30)"],
        stdin="",
        cwd=None,
        env=os.environ.copy(),
        timeout=0.2,
    )

    with pytest.raises(subprocess.TimeoutExpired):
        HostCommandExecutor().run(request, sink)

    assert len(sink.handles) == 1
    assert sink.handles[0].process.poll() is not None
