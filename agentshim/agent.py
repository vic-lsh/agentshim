"""The agent and its resumable session.

Composition layer: this is where a provider name becomes a provider and a
turn becomes a subprocess. It sits above ``providers`` and ``execution`` so
neither of them has to know that agents exist.
"""

from __future__ import annotations

import copy
import posixpath
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from agentshim.core.env import interactive_env
from agentshim.core.errors import (
    AgentShimError,
    CliExitError,
    CliTimeoutError,
    ProviderCapabilityError,
    SchemaDialectError,
    SessionResumeError,
)
from agentshim.core.events import RunFinished, RunStarted, compose_event_handlers
from agentshim.core.profile import (
    ConfigScope,
    McpMechanism,
    McpScope,
    OutputSchemaStyle,
    SchemaDialect,
    SkillScope,
)
from agentshim.core.provider import ArgvContext, ContextualStreamParser, ParserContext
from agentshim.core.schema import compact_json, dialect_problems, materialize
from agentshim.core.skills import SkillTracker
from agentshim.core.turn import TurnRequest, TurnResult, coerce_request
from agentshim.execution.executor import CommandRequest
from agentshim.execution.host import HostCommandExecutor
from agentshim.providers import get_provider

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from agentshim.core.events import AgentEvent, AgentEventHandler
    from agentshim.core.profile import ProviderProfile
    from agentshim.core.provider import McpInstallation, ParsedTurn, Provider, StreamParser
    from agentshim.core.turn import OutputSchema
    from agentshim.core.usage import ProviderUsage
    from agentshim.execution.executor import CommandExecutor, CommandHandle, CommandResult


def _resolve_provider(provider: str | Provider) -> Provider:
    """Resolve a provider name to a provider instance.

    This module sits above ``providers`` precisely so the documented
    ``CliAgent("claude")`` spelling can be a plain module-level import.
    """
    if isinstance(provider, str):
        return get_provider(provider)
    return provider


class CliAgent:
    """A configured provider CLI: binary, model, environment, handlers.

    Binary lookup and the health check run once, here, so a broken install
    fails at construction instead of halfway through the first turn.
    """

    # Each argument is an independent documented option of the public
    # constructor; folding them into a config object would break the API.
    def __init__(  # noqa: PLR0913
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
        """Configure a provider CLI and prove it is usable before any turn runs.

        Construction does real work: the binary is resolved on PATH and the
        provider's health check is run, so a broken install is reported here
        instead of halfway through the first turn.

        Args:
            provider: A registered provider name, or a ``Provider`` instance
                for a provider that is not in the registry.
            model: Provider-specific model id, or ``None`` for the CLI default.
            executor: Where commands run; defaults to the local host.
            env: Replaces the environment entirely rather than extending it.
                The default is ``interactive_env()``, not ``os.environ``,
                because provider CLIs are usually installed by a shell rc file.
            event_handler: Merged with ``event_handlers``; both spellings exist
                and either may be omitted.
            event_handlers: Merged with ``event_handler``, and fanned out to in
                the order given.
            check_timeout: Seconds allowed for the one-off health check. It is
                not a budget for a turn.
            log: Sink for agentshim's own progress lines; dropped by default.
        """
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

    def derive(self, *, model: str | None, event_handler: AgentEventHandler) -> CliAgent:
        """Return a copy that uses *model* and reports events to *event_handler*.

        The binary lookup and health check are not repeated: the copy shares
        the executor, environment and resolved binary of this agent. That is
        what lets one checked install serve many conversations, each with its
        own event routing.
        """
        clone = copy.copy(self)
        clone.model = model
        clone.event_handler = event_handler
        return clone

    def start_session(  # noqa: PLR0913 - one keyword per independent session option
        self,
        *,
        cwd: str | None = None,
        timeout: float | None = None,
        session_id: str | None = None,
        previous_usage: ProviderUsage | None = None,
        skill_scope: SkillScope = SkillScope.ALL,
        mcp_scope: McpScope = McpScope.ALL,
        config_scope: ConfigScope = ConfigScope.ALL,
    ) -> AgentSession:
        """Open a conversation whose turns resume one another.

        ``skill_scope``, ``mcp_scope`` and ``config_scope`` are fixed for the
        conversation (see ``AgentSession``). ``previous_usage`` seeds the
        baseline when adopting a cumulative-usage provider conversation after
        a restart; pass its most recent ``TurnResult.usage``.
        """
        return AgentSession(
            self,
            cwd=cwd,
            timeout=timeout,
            session_id=session_id,
            previous_usage=previous_usage,
            skill_scope=skill_scope,
            mcp_scope=mcp_scope,
            config_scope=config_scope,
        )

    def run(  # noqa: PLR0913 - one keyword per independent session option
        self,
        request: TurnRequest | str,
        *,
        cwd: str | None = None,
        timeout: float | None = None,
        skill_scope: SkillScope = SkillScope.ALL,
        mcp_scope: McpScope = McpScope.ALL,
        config_scope: ConfigScope = ConfigScope.ALL,
    ) -> TurnResult:
        """Run one turn in a throwaway session."""
        session = self.start_session(
            cwd=cwd,
            timeout=timeout,
            skill_scope=skill_scope,
            mcp_scope=mcp_scope,
            config_scope=config_scope,
        )
        return session.turn(request)


def _discard(message: str) -> None:
    """Default log sink: drop the message."""


class AgentSession:
    """A provider conversation. One turn at a time; ``cancel`` is thread-safe."""

    def __init__(  # noqa: PLR0913 - one keyword per independent session option
        self,
        agent: CliAgent,
        *,
        cwd: str | None = None,
        timeout: float | None = None,
        session_id: str | None = None,
        previous_usage: ProviderUsage | None = None,
        skill_scope: SkillScope = SkillScope.ALL,
        mcp_scope: McpScope = McpScope.ALL,
        config_scope: ConfigScope = ConfigScope.ALL,
    ) -> None:
        """Bind a conversation to one agent, with per-turn defaults.

        The session owns the provider's conversation id, which is what makes
        successive turns resume one another. ``cwd`` and ``timeout`` are
        defaults an individual ``TurnRequest`` may override. Passing
        ``session_id`` adopts a conversation the provider already has, so the
        first turn resumes rather than starts fresh. ``previous_usage`` is
        that conversation's last report, including raw cumulative totals,
        when it was run outside this session. ``skill_scope`` limits
        which skills every turn's CLI may discover; a scope the provider's
        ``profile.skill_scopes`` does not list raises
        ``ProviderCapabilityError`` here, before any turn runs. ``mcp_scope``
        limits which MCP servers every turn's CLI may connect to
        (``McpScope.SESSION``: only those in ``TurnRequest.mcp_servers``) and
        is checked against ``profile.mcp_scopes`` the same way.
        ``config_scope`` limits which of the user's own CLI configuration
        (settings, hooks, global instructions, memory) every turn loads, and
        is checked against ``profile.config_scopes``; a provider with
        ``profile.config_home_files`` additionally needs its state root
        pointed at a home from ``prepare_config_home``.
        """
        if skill_scope not in agent.profile.skill_scopes:
            msg = f"{agent.profile.name} cannot limit skills to scope {skill_scope.value!r}"
            raise ProviderCapabilityError(msg)
        if mcp_scope not in agent.profile.mcp_scopes:
            msg = f"{agent.profile.name} cannot limit MCP servers to scope {mcp_scope.value!r}"
            raise ProviderCapabilityError(msg)
        if config_scope not in agent.profile.config_scopes:
            msg = f"{agent.profile.name} cannot limit user configuration to scope {config_scope.value!r}"
            raise ProviderCapabilityError(msg)
        self.skill_scope = skill_scope
        self.config_scope = config_scope
        self.mcp_scope = mcp_scope
        self._agent = agent
        self._cwd = cwd
        self._timeout = timeout
        self.session_id: str | None = session_id
        self.last_result: TurnResult | None = None
        self._usage_by_session: dict[str, ProviderUsage] = {}
        if session_id is not None and previous_usage is not None:
            self._usage_by_session[session_id] = previous_usage
        self._lock = threading.Lock()
        self._handle: CommandHandle | None = None
        self._cancel_requested = False
        self._idle = threading.Event()
        self._idle.set()

    @property
    def profile(self) -> ProviderProfile:
        """Capability description of the provider this session talks to.

        Re-exported from the agent so a caller can ask what a turn will support
        without reaching into the session's private agent reference.
        """
        return self._agent.profile

    def adopt(self, session_id: str, *, previous_usage: ProviderUsage | None = None) -> bool:
        """Continue an existing provider conversation on the next turn.

        ``previous_usage`` seeds an external conversation's last raw report.
        Reports from conversations already seen by this session are retained.
        Returns ``False`` when the provider cannot resume, or when a turn is
        in flight and switching conversations would race it.
        """
        if not self.profile.supports_resume:
            return False
        with self._lock:
            if not self._idle.is_set():
                return False
            self.session_id = session_id
            if previous_usage is not None:
                self._usage_by_session[session_id] = previous_usage
        return True

    def forget(self) -> bool:
        """Start the next turn as a fresh conversation.

        Returns ``False``, changing nothing, when a turn is in flight, which is
        the rule ``adopt`` already follows: the running turn is about to write
        the conversation id it was given, so dropping it mid-flight would
        either be undone a moment later or discard an id nobody else has.
        """
        with self._lock:
            if not self._idle.is_set():
                return False
            self.session_id = None
        return True

    def cancel(self, grace_s: float = 5.0) -> None:
        """Stop the running turn: terminate, then kill after ``grace_s``.

        A turn is in flight from the moment ``turn()`` is entered, but the
        process handle only appears once the executor has spawned the CLI.
        Cancelling in that window (installing MCP servers, building argv,
        spawning) records the request instead of dropping it, and the process
        is terminated as soon as its handle is published. Calling this while no
        turn is running does nothing and leaves the next turn alone.
        """
        with self._lock:
            if self._idle.is_set():
                return
            self._cancel_requested = True
            handle = self._handle
        if handle is not None:
            handle.terminate()
        if self._idle.wait(grace_s):
            return
        with self._lock:
            handle = self._handle
        if handle is not None:
            handle.kill()

    def turn(self, request: TurnRequest | str) -> TurnResult:
        """Run one prompt and return everything it produced.

        The session counts as busy from here, not from the moment the process
        exists, so ``adopt`` and ``cancel`` both see a turn that is still
        installing MCP servers or building argv.
        """
        req = coerce_request(request)
        agent = self._agent
        cwd = req.cwd if req.cwd is not None else self._cwd
        timeout = req.timeout if req.timeout is not None else self._timeout
        env = dict(agent.env)
        if req.env:
            env.update(req.env)

        self._idle.clear()
        installation: McpInstallation | None = None
        try:
            self._check_capabilities(req)
            schema_inline, schema_path = self._resolve_schema(req.output_schema)
            workspace = req.mcp_workspace
            if workspace is None and cwd is not None:
                workspace = Path(cwd)
            installation = agent.provider.install_mcp(workspace, req.mcp_servers)
            argv = agent.provider.build_argv(
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
                    cwd=cwd,
                    skill_scope=self.skill_scope,
                    mcp_scope=self.mcp_scope,
                    config_scope=self.config_scope,
                    mcp_servers=tuple(req.mcp_servers),
                )
            )
            command = CommandRequest(argv=argv, stdin=req.prompt, cwd=cwd, env=env, timeout=timeout)
            return self._execute(command, expect_structured=req.output_schema is not None)
        finally:
            if installation is not None:
                note = installation.restore()
                if note:
                    agent.log(note)
            with self._lock:
                self._handle = None
                self._cancel_requested = False
            self._idle.set()

    def _execute(self, command: CommandRequest, *, expect_structured: bool) -> TurnResult:
        """Run one prepared command and turn what it printed into a result."""
        agent = self._agent
        handler = agent.event_handler
        resumed = self.session_id is not None
        skills = SkillTracker(self.profile)

        def emit(event: AgentEvent) -> None:
            skills.on_event(event)
            handler.on_event(event)

        argv = list(command.argv)
        parser = agent.provider.new_parser(
            emit,
            expect_structured=expect_structured,
        )
        if isinstance(parser, ContextualStreamParser):
            parser.configure(
                ParserContext(
                    previous_usage=(
                        self._usage_by_session.get(self.session_id) if self.session_id else None
                    ),
                    resumed=resumed,
                )
            )
        emit(RunStarted(tuple(argv)))
        started = time.monotonic()
        try:
            result = agent.executor.run(command, _ParserSink(parser, self._set_handle))
        except AgentShimError as error:
            self._abandon(parser, emit, error)
            raise
        duration_ms = int((time.monotonic() - started) * 1000)
        emit(RunFinished(result.returncode))
        parsed = parser.finish()
        self._remember_usage(parsed)
        self._adopt_then_report(parsed, result, argv, resumed=resumed)

        turn_result = TurnResult(
            text=parsed.text,
            structured_output=parsed.structured_output,
            session_id=self.session_id,
            resumed=resumed,
            usage=parsed.usage,
            cost_usd=parsed.cost_usd,
            duration_ms=duration_ms,
            exit_code=result.returncode,
            skills=skills.summary(),
        )
        self.last_result = turn_result
        return turn_result

    def _abandon(
        self,
        parser: StreamParser,
        emit: Callable[[AgentEvent], None],
        error: AgentShimError,
    ) -> None:
        """Close out a run the executor could not finish.

        A timeout or a transport failure ends the stream mid-turn, but the run
        still started, so the boundary event and ``parser.finish()`` are owed to
        the caller exactly as on a clean exit. The session id matters most: a
        turn that named the conversation before it timed out is still
        resumable, and dropping the id would strand it. A timeout also carries
        the partial reading, since that is the only way back to it once the
        error is raised.
        """
        emit(RunFinished(None))
        partial = parser.finish()
        self._remember_usage(partial)
        if partial.session_id:
            self.session_id = partial.session_id
        if isinstance(error, CliTimeoutError):
            error.partial = partial

    def _remember_usage(self, parsed: ParsedTurn) -> None:
        """Retain raw totals even on a failed invocation, without erasing a baseline."""
        session_id = parsed.session_id or self.session_id
        if session_id is not None and parsed.usage.raw is not None:
            self._usage_by_session[session_id] = parsed.usage

    def _adopt_then_report(
        self,
        parsed: ParsedTurn,
        result: CommandResult,
        argv: Sequence[str],
        *,
        resumed: bool,
    ) -> None:
        """Adopt the conversation the run named, then raise on a nonzero exit.

        Adoption comes first because a provider that named the session and then
        failed leaves a conversation the caller can still resume; raising
        before adopting strands it. The exception is a conversation the
        provider says is gone, which there is no point resuming.
        """
        error: AgentShimError | None = None
        if result.returncode != 0:
            error = self._agent.provider.classify_exit(
                CliExitError(
                    argv,
                    result.returncode,
                    result.stdout,
                    result.stderr,
                    kind=parsed.error_kind,
                    detail=parsed.error or "",
                ),
                resumed=resumed,
            )
        if parsed.session_id and not isinstance(error, SessionResumeError):
            self.session_id = parsed.session_id
        if error is not None:
            raise error

    def _set_handle(self, handle: CommandHandle | None) -> None:
        """Publish the running process, honouring a cancel that already fired.

        ``cancel`` can arrive before the executor has anything to stop. It
        records the request rather than dropping it, and this is where that
        record is paid out.
        """
        with self._lock:
            self._handle = handle
            cancelled = self._cancel_requested
        if handle is not None and cancelled:
            handle.terminate()

    def _check_capabilities(self, req: TurnRequest) -> None:
        profile = self.profile
        if req.reasoning_effort is not None and not profile.supports_reasoning_effort:
            msg = f"{profile.name} does not support reasoning effort"
            raise ProviderCapabilityError(msg)
        if req.output_schema is not None and profile.output_schema is OutputSchemaStyle.NONE:
            msg = f"{profile.name} does not support a native output schema"
            raise ProviderCapabilityError(msg)
        if req.mcp_servers and profile.mcp is McpMechanism.NONE:
            msg = f"{profile.name} does not support MCP servers"
            raise ProviderCapabilityError(msg)

    def _resolve_schema(self, schema: OutputSchema | None) -> tuple[str | None, str | None]:
        if schema is None:
            return None, None
        profile = self.profile
        dialect = (
            profile.schema_dialect if profile.schema_dialect is not None else SchemaDialect.STRICT
        )
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
