# Executors

A `CommandExecutor` is the whole of agentshim's process transport. Implement
it to run a provider CLI somewhere other than the local host; agentshim keeps
argv construction, parsing, session state, usage accounting, and events.

```python
class CommandExecutor(Protocol):
    def find_binary(self, name: str, env: Mapping[str, str]) -> str: ...
    def check_binary(self, path: str, env: Mapping[str, str], *, timeout: float) -> None: ...
    def run(self, request: CommandRequest, sink: CommandStreamSink) -> CommandResult: ...
```

`HostCommandExecutor` is the default. It starts the process in its own
session, writes stdin from a helper thread so a prompt larger than the pipe
buffer cannot deadlock, and drains its reader threads on the calling thread so
every sink callback is serialized there. On timeout it kills the process group
and raises `CliTimeoutError`.

## Wrapping argv

`TransformingExecutor` rewrites every request before an inner executor sees
it, which is how you put the CLI in a sandbox or a container without agentshim
knowing about either. The health check goes through `run`, so the transform
applies to it too.

```python
from dataclasses import replace
from agentshim import CliAgent, CommandRequest, HostCommandExecutor, TransformingExecutor

def in_container(request: CommandRequest) -> CommandRequest:
    return replace(request, argv=["docker", "exec", "-i", "workspace", *request.argv])

agent = CliAgent(
    "claude",
    executor=TransformingExecutor(HostCommandExecutor(), in_container),
)
```

Pass `find_binary=` when lookup should happen in the target runtime rather
than on the host:

```python
TransformingExecutor(inner, in_container, find_binary=lambda name, env: f"/usr/local/bin/{name}")
```

## Writing one from scratch

An executor must call `sink.started(handle)` once the command is running,
write `request.stdin` and close it, call `sink.stdout(line)` and
`sink.stderr(line)` for each line (newlines kept), honour `cwd`, `env` and
`timeout`, and return a `CommandResult`. Raise `CliNotFoundError` from
`find_binary`, `CliCheckError` from `check_binary`, and `CliTimeoutError` from
`run`.

Sink callbacks must all happen on the thread that called `run`: that is what
makes the event-handler thread contract hold.
