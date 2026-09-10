"""Run provider CLIs on the local host with ``subprocess``."""

from __future__ import annotations

import contextlib
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
from typing import TYPE_CHECKING

from agentshim.core.errors import (
    CliCheckError,
    CliExitError,
    CliNotFoundError,
    CliTimeoutError,
)

from .executor import CommandRequest, CommandResult, NullSink

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import IO

    from .executor import CommandStreamSink

_JOIN_TIMEOUT_S = 5.0
_EOF = None


class _TimedOutError(Exception):
    """Internal signal that the wall-clock budget is spent."""


class ProcessCommandHandle:
    """Handle for a local process started in its own session."""

    def __init__(self, process: subprocess.Popen[str]) -> None:
        """Wrap a started process so a caller can stop it from outside."""
        self.process = process

    @property
    def pid(self) -> int:
        """PID of the process, which is also its process-group id."""
        return self.process.pid

    def terminate(self) -> None:
        """Send SIGTERM to the process group."""
        _signal_group(self.process, signal.SIGTERM)

    def kill(self) -> None:
        """Send SIGKILL to the process group."""
        _signal_group(self.process, signal.SIGKILL)


def _signal_group(process: subprocess.Popen[str], sig: int) -> None:
    """Signal the whole process group so the CLI's children die with it."""
    try:
        os.killpg(os.getpgid(process.pid), sig)
    except (ProcessLookupError, PermissionError, OSError):
        with contextlib.suppress(ProcessLookupError, ValueError, OSError):
            process.send_signal(sig)


class HostCommandExecutor:
    """Default executor: one process per command, in its own session.

    stdin is written from a helper thread so a prompt larger than the pipe
    buffer cannot deadlock against a child that is still printing. stdout and
    stderr are read on helper threads into one queue that the calling thread
    drains, so every sink callback is serialized on the caller's thread.
    """

    def find_binary(self, name: str, env: Mapping[str, str]) -> str:
        """Look *name* up on the turn's PATH, falling back to this process's.

        Raises ``CliNotFoundError`` when neither lookup finds the binary.
        """
        path = shutil.which(name, path=env.get("PATH")) or shutil.which(name)
        if not path:
            raise CliNotFoundError(name)
        return path

    def check_binary(self, path: str, env: Mapping[str, str], *, timeout: float) -> None:
        """Run ``<path> --help`` and report any failure as ``CliCheckError``."""
        request = CommandRequest(
            argv=[path, "--help"], stdin=None, cwd=None, env=env, timeout=timeout
        )
        try:
            result = self.run(request, NullSink())
        except CliTimeoutError as exc:
            raise CliCheckError(
                path, f"{path} did not respond to '--help' within {timeout}s"
            ) from exc
        except FileNotFoundError as exc:
            raise CliCheckError(path, f"CLI tool not found at {path!r}") from exc
        except OSError as exc:
            raise CliCheckError(path, f"cannot execute {path!r}: {exc}") from exc
        if result.returncode != 0:
            raise CliCheckError(
                path,
                f"'{path} --help' exited with code {result.returncode}. Stderr: {result.stderr.strip()}",
            )

    def run(self, request: CommandRequest, sink: CommandStreamSink) -> CommandResult:
        """Run the command, streaming into *sink* until both streams close.

        The process is killed on every exit path, including a sink that raised
        or a timeout, so no CLI outlives the call.
        """
        argv = list(request.argv)
        # Running caller-supplied argv is this executor's whole purpose, and the
        # list form never reaches a shell.
        process = subprocess.Popen(  # noqa: S603
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # A provider CLI may print a byte that is not valid UTF-8 (a file
            # excerpt, a mis-encoded tool result). Without these the decoder
            # raises mid-stream and the rest of the turn's output is lost.
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=request.cwd,
            env=dict(request.env),
            start_new_session=True,
        )
        lines: queue.Queue[tuple[str, str | None]] = queue.Queue()
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        workers: list[threading.Thread] = []
        failures: list[tuple[str, BaseException]] = []

        try:
            workers.append(_spawn(_write_stdin, process.stdin, request.stdin))
            workers.append(_spawn(_pump, process.stdout, "out", lines, failures))
            workers.append(_spawn(_pump, process.stderr, "err", lines, failures))
            sink.started(ProcessCommandHandle(process))
            deadline = None if request.timeout is None else time.monotonic() + request.timeout

            open_streams = 2
            while open_streams:
                kind, line = _next(lines, deadline)
                if line is None:
                    open_streams -= 1
                    continue
                if kind == "out":
                    stdout_lines.append(line)
                    sink.stdout(line)
                else:
                    stderr_lines.append(line)
                    sink.stderr(line)

            _wait(process, deadline)
        except _TimedOutError:
            _kill(process)
            _join(workers)
            timeout = request.timeout if request.timeout is not None else 0.0
            raise CliTimeoutError(argv, timeout) from None
        except BaseException:
            # A sink raised, or the caller was interrupted: do not leave the
            # CLI running behind the exception.
            _kill(process)
            raise
        finally:
            _kill(process)
            _join(workers)
            _close(process)

        returncode = process.returncode if process.returncode is not None else -1
        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_lines)
        if failures:
            raise _reader_failure(argv, returncode, stdout, stderr, failures[0])
        return CommandResult(returncode=returncode, stdout=stdout, stderr=stderr)


def _reader_failure(
    argv: list[str],
    returncode: int,
    stdout: str,
    stderr: str,
    failure: tuple[str, BaseException],
) -> CliExitError:
    """Turn a reader-thread crash into an error the caller cannot mistake for success.

    A stream that stopped being read is not a stream that ended: presenting it
    as a clean EOF would hand back a truncated (often empty) transcript with a
    zero exit code. The bytes that were read before the failure are kept, so a
    caller can still see how far the turn got.
    """
    kind, exc = failure
    stream_name = "stdout" if kind == "out" else "stderr"
    binary = argv[0] if argv else "cli"
    return CliExitError(
        argv,
        returncode,
        stdout,
        stderr,
        message=f"could not read {stream_name} from {binary}: {exc!r}",
    )


def _spawn(target: object, *args: object) -> threading.Thread:
    thread = threading.Thread(target=target, args=args, daemon=True)  # pyright: ignore[reportArgumentType]
    thread.start()
    return thread


def _write_stdin(stream: IO[str] | None, data: str | None) -> None:
    if stream is None:
        return
    try:
        if data:
            stream.write(data)
        stream.flush()
    except (BrokenPipeError, ValueError, OSError):
        pass
    finally:
        with contextlib.suppress(BrokenPipeError, ValueError, OSError):
            stream.close()


def _pump(
    stream: IO[str] | None,
    kind: str,
    sink: queue.Queue[tuple[str, str | None]],
    failures: list[tuple[str, BaseException]],
) -> None:
    if stream is None:
        sink.put((kind, _EOF))
        return
    try:
        for line in iter(stream.readline, ""):
            sink.put((kind, line))
    except (ValueError, OSError) as exc:
        failures.append((kind, exc))
    finally:
        # Closing the read end makes the child's next write fail instead of
        # blocking forever on a full pipe that nobody is draining.
        with contextlib.suppress(BrokenPipeError, ValueError, OSError):
            stream.close()
        sink.put((kind, _EOF))


def _next(
    lines: queue.Queue[tuple[str, str | None]],
    deadline: float | None,
) -> tuple[str, str | None]:
    if deadline is None:
        return lines.get()
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _TimedOutError
    try:
        return lines.get(timeout=remaining)
    except queue.Empty:
        raise _TimedOutError from None


def _wait(process: subprocess.Popen[str], deadline: float | None) -> None:
    if deadline is None:
        process.wait()
        return
    remaining = max(deadline - time.monotonic(), 0.0)
    try:
        process.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        raise _TimedOutError from None


def _kill(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    _signal_group(process, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=_JOIN_TIMEOUT_S)


def _join(workers: list[threading.Thread]) -> None:
    for worker in workers:
        worker.join(timeout=_JOIN_TIMEOUT_S)


def _close(process: subprocess.Popen[str]) -> None:
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is None:
            continue
        with contextlib.suppress(BrokenPipeError, ValueError, OSError):
            stream.close()
