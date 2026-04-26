from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import io
    from collections.abc import Callable


class CommandHandle(Protocol):
    """Executor-neutral handle for a started command.

    Executors pass a handle to :meth:`CommandStreamSink.started` immediately
    after the command has been launched. Callers can keep this handle to stop
    the command from outside the normal timeout path.
    """

    def terminate(self) -> None:
        """Ask the running command to terminate."""
        ...

    def kill(self) -> None:
        """Forcefully stop the running command."""
        ...


@dataclass(frozen=True)
class CommandRequest:
    """Inputs needed to run a provider CLI command.

    Attributes:
        argv: Complete command argument vector. ``argv[0]`` is the provider
            binary path returned by :meth:`CommandExecutor.find_binary`.
        stdin: Prompt text to write to the command's standard input.
        cwd: Working directory for the command, or ``None`` to inherit the
            executor's default.
        env: Environment variables to use for the command.
        timeout: Maximum command runtime in seconds, or ``None`` for no
            timeout. Executors should raise ``subprocess.TimeoutExpired`` or
            an equivalent timeout error when this limit is exceeded.
    """

    argv: list[str]
    stdin: str
    cwd: str | None
    env: dict[str, str]
    timeout: float | None


@dataclass(frozen=True)
class CommandResult:
    """Completed CLI command output.

    ``stdout`` and ``stderr`` should contain the same text that was streamed
    through :class:`CommandStreamSink`. ``returncode`` follows
    ``subprocess.Popen.returncode`` conventions.
    """

    returncode: int | None
    stdout: str
    stderr: str


class CommandStreamSink(Protocol):
    """Receives command lifecycle and line-oriented output events.

    Provider parsers consume these callbacks incrementally, so executors
    should call ``stdout`` and ``stderr`` as output becomes available. Lines
    should include their trailing newline when the underlying stream provided
    one, matching ``TextIO.readline`` behavior.
    """

    def started(self, handle: CommandHandle) -> None:
        """Called once, immediately after the command has started."""
        ...

    def stdout(self, line: str) -> None:
        """Called for each stdout line."""
        ...

    def stderr(self, line: str) -> None:
        """Called for each stderr line."""
        ...


@dataclass
class CallbackCommandStreamSink:
    """Command stream sink backed by callables.

    This is a convenience adapter for callers that already have simple
    callback functions. Exceptions raised by ``on_started`` are ignored to
    preserve the historical best-effort process-start callback behavior.
    """

    on_stdout: Callable[[str], None]
    on_stderr: Callable[[str], None]
    on_started: Callable[[CommandHandle], None] | None = None

    def started(self, handle: CommandHandle) -> None:
        if self.on_started is not None:
            try:
                self.on_started(handle)
            except Exception:
                pass

    def stdout(self, line: str) -> None:
        self.on_stdout(line)

    def stderr(self, line: str) -> None:
        self.on_stderr(line)


class CommandExecutor(Protocol):
    """Controls binary lookup, validation, and streaming command execution.

    Implement this protocol to run provider CLIs somewhere other than the
    local host process, such as inside an existing container or over a remote
    shell. agentshim still owns provider-specific command construction,
    stdout/stderr parsing, session state, usage accounting, and event
    handling.
    """

    def find_binary(self, binary_name: str, env: dict[str, str]) -> str:
        """Return the executable path/name to put in provider commands.

        The returned value becomes ``CommandRequest.argv[0]``. Host executors
        usually resolve an absolute path; container or remote executors may
        return the original binary name if lookup happens in the target
        runtime.
        """
        ...

    def check_binary(
        self,
        binary_path: str,
        env: dict[str, str],
        *,
        timeout: int,
    ) -> None:
        """Validate the executable before the first command run.

        Raise ``RuntimeError`` if the executable is unavailable or broken.
        Executors that cannot cheaply validate the target runtime may make
        this a no-op.
        """
        ...

    def run(
        self,
        request: CommandRequest,
        sink: CommandStreamSink,
    ) -> CommandResult:
        """Run ``request`` and stream stdout/stderr into ``sink``.

        Implementations should:

        - call ``sink.started(handle)`` once the command is running
        - write ``request.stdin`` to the command's stdin and close it
        - call ``sink.stdout(line)`` and ``sink.stderr(line)`` for each line
        - honor ``request.cwd``, ``request.env``, and ``request.timeout``
        - return a ``CommandResult`` after the command exits
        """
        ...


@dataclass
class ProcessCommandHandle:
    """Command handle for a local ``subprocess.Popen`` process."""

    process: Any

    @property
    def pid(self) -> int:
        return int(self.process.pid)

    def terminate(self) -> None:
        self.process.terminate()

    def kill(self) -> None:
        self.process.kill()


class HostCommandExecutor:
    """Default executor that runs commands on the host via ``subprocess``.

    The host executor uses line-buffered text pipes, streams stdout and stderr
    from separate reader threads, starts commands in a new process group, and
    kills the process group when a timeout or forced cleanup occurs.
    """

    def __init__(self, *, shutil_module: Any = shutil, subprocess_module: Any = subprocess):
        self._shutil = shutil_module
        self._subprocess = subprocess_module

    def find_binary(self, binary_name: str, env: dict[str, str]) -> str:
        binary_path = self._shutil.which(binary_name, path=env.get("PATH"))
        if not binary_path:
            binary_path = self._shutil.which(binary_name)
        if not binary_path:
            raise RuntimeError(
                f"{binary_name} binary not found in PATH. Please ensure {binary_name} is installed and available."
            )
        return binary_path

    def check_binary(
        self,
        binary_path: str,
        env: dict[str, str],
        *,
        timeout: int,
    ) -> None:
        try:
            result = self._subprocess.run(
                [binary_path, "--help"],
                capture_output=True,
                text=True,
                check=False,
                env=env,
                stdin=self._subprocess.DEVNULL,
                timeout=timeout,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"CLI tool at '{binary_path}' is not working correctly. "
                    f"'{binary_path} --help' exited with code {result.returncode}. "
                    f"Stderr: {result.stderr}"
                )
        except self._subprocess.TimeoutExpired as e:
            raise RuntimeError(f"CLI tool at '{binary_path}' did not respond to '--help' within {timeout}s.") from e
        except FileNotFoundError as e:
            raise RuntimeError(
                f"CLI tool not found at '{binary_path}'. Please ensure it is installed and in your PATH."
            ) from e

    def run(
        self,
        request: CommandRequest,
        sink: CommandStreamSink,
    ) -> CommandResult:
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        def read_stdout(pipe: io.TextIOWrapper) -> None:
            for line in iter(pipe.readline, ""):
                if not line:
                    break
                stdout_lines.append(line)
                sink.stdout(line)
            pipe.close()

        def read_stderr(pipe: io.TextIOWrapper) -> None:
            for line in iter(pipe.readline, ""):
                if not line:
                    break
                stderr_lines.append(line)
                sink.stderr(line)
            pipe.close()

        process = self._subprocess.Popen(
            request.argv,
            stdin=self._subprocess.PIPE,
            stdout=self._subprocess.PIPE,
            stderr=self._subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=request.cwd,
            env=request.env,
            start_new_session=True,
        )

        sink.started(ProcessCommandHandle(process))

        stdout_thread = threading.Thread(target=read_stdout, args=(process.stdout,))
        stderr_thread = threading.Thread(target=read_stderr, args=(process.stderr,))

        stdout_thread.daemon = True
        stderr_thread.daemon = True

        stdout_thread.start()
        stderr_thread.start()

        try:
            if process.stdin:
                process.stdin.write(request.stdin)
                process.stdin.close()
        except BrokenPipeError:
            pass

        try:
            process.wait(timeout=request.timeout)
        except self._subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise self._subprocess.TimeoutExpired(request.argv, request.timeout) from None
        finally:
            if process.poll() is None:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    process.wait()
                except (ProcessLookupError, OSError):
                    pass

        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)

        return CommandResult(
            returncode=process.returncode,
            stdout="".join(stdout_lines),
            stderr="".join(stderr_lines),
        )
