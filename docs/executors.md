# Executors

Provider classes accept an optional `executor=`. The executor controls binary
lookup, CLI validation, and streaming process execution while `agentshim` keeps
owning provider command construction, stdout/stderr parsing, session state, and
event emission.

This is useful when a caller needs to run the CLI somewhere other than the
current host process, for example through an existing container, remote shell,
or custom sandbox.

```python
from agentshim import (
    CodexCodingAgent,
    CommandHandle,
    CommandRequest,
    CommandResult,
    CommandStreamSink,
)


class MyCommandHandle:
    def terminate(self) -> None:
        ...

    def kill(self) -> None:
        ...


class MyExecutor:
    def find_binary(self, binary_name: str, env: dict[str, str]) -> str:
        # This value becomes request.argv[0]. Container or remote executors can
        # return the binary name if lookup happens in the target runtime.
        return binary_name

    def check_binary(self, binary_path: str, env: dict[str, str], *, timeout: int) -> None:
        # Raise RuntimeError if the target CLI is unavailable. No-op is fine
        # when validation is not cheap or is handled by the runtime.
        return None

    def run(self, request: CommandRequest, sink: CommandStreamSink) -> CommandResult:
        handle = MyCommandHandle()
        sink.started(handle)

        stdout = ""
        stderr = ""

        # Run request.argv in your target runtime, with request.stdin,
        # request.cwd, request.env, and request.timeout. Stream each complete
        # line as it arrives, preserving trailing newlines when present.
        line = "streamed output\n"
        stdout += line
        sink.stdout(line)

        return CommandResult(returncode=0, stdout=stdout, stderr=stderr)


agent = CodexCodingAgent(executor=MyExecutor())
```

The default `HostCommandExecutor` preserves the normal local `subprocess`
behavior. `CodingAgent(provider=..., executor=...)` forwards the same executor
to the selected provider.

Executor contract:

- `CommandRequest.argv` is the complete provider CLI command. `argv[0]` is the
  value returned by `find_binary`.
- `CommandRequest.stdin` is the prompt text to write to the command's standard
  input, then stdin should be closed.
- `CommandRequest.cwd`, `env`, and `timeout` should be honored by the executor.
- Call `sink.started(handle)` once after the command starts. The handle only
  needs `terminate()` and `kill()`.
- Call `sink.stdout(line)` and `sink.stderr(line)` as output is produced. Lines
  should include trailing newlines when the underlying stream provided them.
- Return `CommandResult(returncode, stdout, stderr)` after the command exits.
  The returned text should match what was streamed through the sink.

If you were using the pre-0.5 executor preview, replace
`run_streaming(cmd, ..., on_stdout, on_stderr, on_process_started)` with
`run(request, sink)`. The parser/event APIs remain internal; custom executors
only provide a stable command runtime.
