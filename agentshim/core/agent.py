"""The agent and its resumable session."""

from __future__ import annotations

import posixpath
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from ..execution.executor import CommandRequest
from ..execution.host import HostCommandExecutor
from .env import interactive_env
from .errors import CliExitError, ProviderCapabilityError, SchemaDialectError
from .events import RunFinished, RunStarted, compose_event_handlers
from .profile import McpMechanism, OutputSchemaStyle, SchemaDialect
from .provider import ArgvContext
from .schema import compact_json, dialect_problems, materialize
from .turn import TurnRequest, TurnResult, coerce_request

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from ..execution.executor import CommandExecutor, CommandHandle
    from .events import AgentEvent, AgentEventHandler
    from .profile import ProviderProfile
    from .provider import Provider, StreamParser
    from .turn import OutputSchema


def _resolve_provider(provider: str | Provider) -> Provider:
    """Resolve a provider name to a provider instance.

    The import is function-local on purpose: ``core`` must not depend on
    ``providers`` at module level, but the documented ``CliAgent("claude")``
    spelling has to keep working.
    """
    if isinstance(provider, str):
        from ..providers import get_provider

        return get_provider(provider)
    return provider


class CliAgent:
    """A configured provider CLI: binary, model, environment, handlers.

    Binary lookup and the health check run once, here, so a broken install
    fails at construction instead of halfway through the first turn.
    """

    def __init__(
        self,
        provider: str | Provider,
        *,
        model: str | None = None,
        executor: CommandExecutor | None = None,
        env: Mapping[str, str] | None = None,
        event_handler: AgentEventHandler | None = None,
        event_handlers: Sequence[AgentEventHandler] = (),
        check_timeout: float = 15.0,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.provider: Provider = _resolve_provider(provider)
        self.profile: ProviderProfile = self.provider.profile
        self.model = model
        self.executor: CommandExecutor = executor if executor is not None else HostCommandExecutor()
        self.env: dict[str, str] = dict(env) if env is not None else interactive_env()
        self.event_handler = compose_event_handlers(event_handler, event_handlers)
        self.log: Callable[[str], None] = log if log is not None else _discard

        self.binary_path = self.executor.find_binary(self.profile.binary, self.env)
        self.executor.check_binary(self.binary_path, self.env, timeout=check_timeout)
        self.log(f"{self.profile.display_name} ready at {self.binary_path}")

    def start_session(
        self,
        *,
        cwd: str | None = None,
        timeout: float | None = None,
        session_id: str | None = None,
    ) -> AgentSession:
        """Open a conversation whose turns resume one another."""
        return AgentSession(self, cwd=cwd, timeout=timeout, session_id=session_id)

    def run(
        self,
        request: TurnRequest | str,
        *,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> TurnResult:
        """Run one turn in a throwaway session."""
        return self.start_session(cwd=cwd, timeout=timeout).turn(request)


def _discard(message: str) -> None:
    """Default log sink: drop the message."""


class AgentSession:
    """A provider conversation. One turn at a time; ``cancel`` is thread-safe."""

    def __init__(
        self,
        agent: CliAgent,
        *,
        cwd: str | None = None,
        timeout: float | None = None,
        session_id: str | None = None,
    ) -> None:
        self._agent = agent
        self._cwd = cwd
        self._timeout = timeout
        self.session_id: str | None = session_id
        self.last_result: TurnResult | None = None
        self._lock = threading.Lock()
        self._handle: CommandHandle | None = None
        self._idle = threading.Event()
        self._idle.set()

    @property
    def profile(self) -> ProviderProfile:
        return self._agent.profile

    def adopt(self, session_id: str) -> bool:
        """Continue an existing provider conversation on the next turn.

        Returns ``False`` when the provider cannot resume, or when a turn is
        in flight and switching conversations would race it.
        """
        if not self.profile.supports_resume:
            return False
        with self._lock:
            if not self._idle.is_set():
                return False
            self.session_id = session_id
        return True

    def forget(self) -> None:
        """Start the next turn as a fresh conversation."""
        self.session_id = None

    def cancel(self, grace_s: float = 5.0) -> None:
        """Stop the running turn: terminate, then kill after ``grace_s``."""
        with self._lock:
            handle = self._handle
        if handle is None:
            return
        handle.terminate()
        if not self._idle.wait(grace_s):
            handle.kill()

    def turn(self, request: TurnRequest | str) -> TurnResult:
        """Run one prompt and return everything it produced."""
        req = coerce_request(request)
        agent = self._agent
        provider = agent.provider
        cwd = req.cwd if req.cwd is not None else self._cwd
        timeout = req.timeout if req.timeout is not None else self._timeout
        env = dict(agent.env)
        if req.env:
            env.update(req.env)

        self._check_capabilities(req)
        schema_inline, schema_path = self._resolve_schema(req.output_schema)

        workspace = Path(cwd) if cwd is not None else None
        installation = provider.install_mcp(workspace, req.mcp_servers)
        resumed = self.session_id is not None
        handler = agent.event_handler

        def emit(event: AgentEvent) -> None:
            handler.on_event(event)

        self._idle.clear()
        try:
            argv = provider.build_argv(
                ArgvContext(
                    binary_path=agent.binary_path,
                    model=agent.model,
                    env=env,
                    resume_session_id=self.session_id,
                    reasoning_effort=req.reasoning_effort,
                    schema_inline=schema_inline,
                    schema_path=schema_path,
                    mcp_argv=installation.argv,
                    extra_args=req.extra_args,
                )
            )
            parser = provider.new_parser(emit, expect_structured=req.output_schema is not None)
            emit(RunStarted(tuple(argv)))
            started = time.monotonic()
            result = agent.executor.run(
                CommandRequest(argv=argv, stdin=req.prompt, cwd=cwd, env=env, timeout=timeout),
                _ParserSink(parser, self._set_handle),
            )
            duration_ms = int((time.monotonic() - started) * 1000)
            emit(RunFinished(result.returncode))
            parsed = parser.finish()

            if result.returncode != 0:
                raise provider.classify_exit(
                    CliExitError(argv, result.returncode, result.stdout, result.stderr),
                    resumed=resumed,
                )

            if parsed.session_id:
                self.session_id = parsed.session_id
            turn_result = TurnResult(
                text=parsed.text,
                structured_output=parsed.structured_output,
                session_id=self.session_id,
                resumed=resumed,
                usage=parsed.usage,
                cost_usd=parsed.cost_usd,
                duration_ms=duration_ms,
                exit_code=result.returncode,
            )
            self.last_result = turn_result
            return turn_result
        finally:
            installation.restore()
            self._set_handle(None)
            self._idle.set()

    def _set_handle(self, handle: CommandHandle | None) -> None:
        with self._lock:
            self._handle = handle

    def _check_capabilities(self, req: TurnRequest) -> None:
        profile = self.profile
        if req.reasoning_effort is not None and not profile.supports_reasoning_effort:
            raise ProviderCapabilityError(f"{profile.name} does not support reasoning effort")
        if req.output_schema is not None and profile.output_schema is OutputSchemaStyle.NONE:
            raise ProviderCapabilityError(f"{profile.name} does not support a native output schema")
        if req.mcp_servers and profile.mcp is McpMechanism.NONE:
            raise ProviderCapabilityError(f"{profile.name} does not support MCP servers")

    def _resolve_schema(self, schema: OutputSchema | None) -> tuple[str | None, str | None]:
        if schema is None:
            return None, None
        profile = self.profile
        dialect = profile.schema_dialect if profile.schema_dialect is not None else SchemaDialect.STRICT
        problems = dialect_problems(schema.schema, dialect)
        if problems:
            raise SchemaDialectError(problems)
        if profile.output_schema is OutputSchemaStyle.INLINE_JSON:
            return compact_json(schema.schema), None
        written = materialize(schema.schema, schema.host_dir)
        base = schema.cli_dir if schema.cli_dir is not None else str(schema.host_dir)
        return None, posixpath.join(base, written.name)


class _ParserSink:
    """Feeds a ``StreamParser`` and publishes the process handle.

    Every callback arrives on the thread that called ``turn()``, which is
    what makes the event-handler thread contract hold.
    """

    def __init__(
        self,
        parser: StreamParser,
        on_started: Callable[[CommandHandle | None], None],
    ) -> None:
        self._parser = parser
        self._on_started = on_started

    def started(self, handle: CommandHandle) -> None:
        self._on_started(handle)

    def stdout(self, line: str) -> None:
        self._parser.feed_stdout(line)

    def stderr(self, line: str) -> None:
        self._parser.feed_stderr(line)
