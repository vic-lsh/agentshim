"""``CodexAppServerTransport``: one long-lived ``codex app-server`` per conversation.

``open`` starts the process through the agent's executor, performs the
``initialize`` handshake, then ``thread/start`` (or ``thread/resume`` for a
conversation to continue). A turn is a ``turn/start`` followed by pulling the
server's messages until ``turn/completed``. Everything runs on the calling
thread: the only other thread that ever touches the pipe is whichever one
calls ``interrupt``, and writes are serialized by the ``Channel``.

The transport never retries and never starts over; it reports. A failed turn
is a ``TurnFailedError`` classified from ``codexErrorInfo``, a thread Codex no
longer has is a ``SessionResumeError``, and the session above decides what to
do about either.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field, replace
from enum import Enum
from importlib import metadata
from typing import TYPE_CHECKING, cast

from agentshim.core.clock import SystemClock
from agentshim.core.env import interactive_env
from agentshim.core.errors import (
    AgentShimError,
    FailureKind,
    NoRunningTurnError,
    ProviderCapabilityError,
    SchemaDialectError,
    SessionResumeError,
    SessionStateError,
    TurnCancelledError,
    TurnFailedError,
    TurnTimeoutError,
)
from agentshim.core.events import (
    ApprovalDenied,
    Lifecycle,
    ProviderError,
    SessionStarted,
    SteerConsumed,
    SteerDelivered,
    SteerRejected,
    TurnInterrupted,
    UsageReport,
)
from agentshim.core.permissions import ApprovalPolicy
from agentshim.core.pricing import cost_usd, price_for
from agentshim.core.schema import dialect_problems
from agentshim.core.skills import SkillTracker
from agentshim.core.turn import TurnResult
from agentshim.execution.host import HostCommandExecutor
from agentshim.execution.process import SpawnRequest
from agentshim.providers.codex.provider import scope_overrides, shell_path_config

from .approvals import answer_request
from .channel import Channel, Deadline, Expired, Gone
from .errors import classify_turn_error, error_text
from .items import ItemEvents
from .mcp import server_keys, thread_config
from .permissions import check_applied, codex_permissions
from .profile import APP_SERVER_PROFILE
from .protocol import (
    INITIALIZED,
    AccountRateLimitsUpdatedNotification,
    ClientInfo,
    ErrorNotification,
    ErrorResponse,
    InitializeParams,
    ItemCompletedNotification,
    ItemStartedNotification,
    McpServerStartupState,
    McpServerStatusUpdatedNotification,
    Notification,
    Response,
    ServerRequest,
    ThreadItemAgentMessage,
    ThreadItemUserMessage,
    ThreadResumeParams,
    ThreadResumeResponse,
    ThreadStartParams,
    ThreadStartResponse,
    ThreadTokenUsageUpdatedNotification,
    TurnCompletedNotification,
    TurnInterruptParams,
    TurnStartedNotification,
    TurnStartParams,
    TurnStartResponse,
    TurnStatus,
    TurnSteerParams,
    UserInputText,
    WarningNotification,
)
from .rate_limits import rate_limit_statuses
from .usage import Totals, baseline_from, totals_of, turn_usage, zero_totals

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from agentshim.core.clock import Clock
    from agentshim.core.conversation import Conversation, ConversationSpec
    from agentshim.core.events import AgentEvent
    from agentshim.core.ids import IdAllocator
    from agentshim.core.mcp import McpServer
    from agentshim.core.pricing import ModelPricing, PricingTable
    from agentshim.core.profile import ProviderProfile
    from agentshim.core.turn import TurnRequest
    from agentshim.core.usage import ProviderUsage
    from agentshim.execution.executor import CommandExecutor
    from agentshim.execution.process import Process

    from ._wire import JsonObject, JsonValue
    from .permissions import CodexPermissions
    from .protocol import (
        AskForApproval,
        ClientRequestParams,
        SandboxPolicy,
        ServerMessage,
        ThreadItem,
        Turn,
        TurnError,
    )

#: Error code Codex answers a ``thread/resume`` of a thread it does not have.
_RESUME_REFUSALS = ("no rollout found", "invalid session id", "invalid thread id")
#: Seconds the server may take to answer the start-up requests.
DEFAULT_STARTUP_TIMEOUT_S = 60.0
#: Seconds each step of ``close`` waits (stdin closed, terminated, killed).
DEFAULT_CLOSE_TIMEOUT_S = 5.0
#: Seconds an interrupted turn may take to report that it ended.
DEFAULT_INTERRUPT_GRACE_S = 10.0
_MS_PER_S = 1000


def _client_version() -> str:
    try:
        return metadata.version("agentshim")
    except metadata.PackageNotFoundError:  # pragma: no cover - always installed in a checkout
        return "0"


class CodexAppServerTransport:
    """A ``Transport`` over ``codex app-server``, one process per conversation.

    Construction resolves the binary on *executor* and runs its health check
    once, like ``CliAgent``, so a broken install fails here. *executor* is the
    agent's own (possibly confined) executor: the process is started through
    its ``spawn``, so a confinement wraps it like any other command.

    The timeouts are in seconds on *clock*: ``startup_timeout`` bounds the
    handshake, ``close_timeout`` each step of shutting the process down, and
    ``interrupt_grace`` how long an interrupted or timed-out turn may take to
    wind down before the conversation is written off.
    """

    # Each argument is an independent documented option of the public constructor.
    def __init__(  # noqa: PLR0913
        self,
        *,
        executor: CommandExecutor | None = None,
        env: Mapping[str, str] | None = None,
        check_timeout: float = 15.0,
        log: Callable[[str], None] | None = None,
        clock: Clock | None = None,
        ids: IdAllocator | None = None,
        pricing: PricingTable | None = None,
        startup_timeout: float = DEFAULT_STARTUP_TIMEOUT_S,
        close_timeout: float = DEFAULT_CLOSE_TIMEOUT_S,
        interrupt_grace: float = DEFAULT_INTERRUPT_GRACE_S,
    ) -> None:
        """Check the install on *executor* (the local host by default).

        *ids* is accepted so this class satisfies ``StreamTransportFactory``;
        Codex's server allocates every thread and turn id, so it is unused.
        """
        del ids
        self._executor: CommandExecutor = (
            executor if executor is not None else HostCommandExecutor()
        )
        self._env: dict[str, str] = dict(env) if env is not None else interactive_env()
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._pricing = pricing
        self._startup_timeout = startup_timeout
        self._close_timeout = close_timeout
        self._interrupt_grace = interrupt_grace
        self._log: Callable[[str], None] = log if log is not None else _discard
        profile = APP_SERVER_PROFILE
        self._binary = self._executor.find_binary(profile.binary, self._env)
        self._executor.check_binary(self._binary, self._env, timeout=check_timeout)
        self._log(f"{profile.display_name} app-server ready at {self._binary}")

    @property
    def profile(self) -> ProviderProfile:
        """What Codex can do over ``app-server``."""
        return APP_SERVER_PROFILE

    def open(self, spec: ConversationSpec) -> Conversation:
        """Start the process and a thread, or resume ``spec.resume_id``.

        Raises:
            ProviderCapabilityError: *spec* asks for something this transport
                cannot do, or Codex did not apply the permissions asked for.
            SessionResumeError: Codex has no thread ``spec.resume_id``.
            TurnFailedError: The process would not start or answer.
            TurnTimeoutError: The handshake took longer than ``startup_timeout``.
        """
        settings = self._settings_for(spec)
        request = SpawnRequest(argv=self._argv(spec, settings), cwd=spec.cwd, env=dict(self._env))

        def spawn() -> Process:
            try:
                return self._executor.spawn(request)
            except OSError as error:
                msg = f"could not start codex app-server: {error}"
                raise TurnFailedError(msg, kind=FailureKind.OTHER, detail=str(error)) from error

        conversation = _Conversation(spawn, self._clock, spec, settings, self._timeouts())
        try:
            conversation.start()
        except BaseException:
            conversation.close()
            raise
        return conversation

    def _settings_for(self, spec: ConversationSpec) -> _Settings:
        profile = self.profile
        if spec.permissions.mode not in profile.native_permission_modes:
            msg = f"{profile.name} cannot enforce native permission mode {spec.permissions.mode.value!r}"
            raise ProviderCapabilityError(msg)
        for scope, allowed in (
            (spec.skill_scope, profile.skill_scopes),
            (spec.mcp_scope, profile.mcp_scopes),
            (spec.config_scope, profile.config_scopes),
        ):
            if scope not in allowed:
                msg = f"{profile.name} over app-server does not support scope {scope.value!r}"
                raise ProviderCapabilityError(msg)
        config = thread_config(spec.mcp_servers)
        model = spec.model
        return _Settings(
            permissions=codex_permissions(spec.permissions),
            config=config,
            mcp_keys=server_keys(spec.mcp_servers),
            servers=tuple(spec.mcp_servers),
            pricing=None if model is None else price_for(profile.name, model, self._pricing),
            pricing_table=self._pricing,
            version=_client_version(),
        )

    def _argv(self, spec: ConversationSpec, settings: _Settings) -> list[str]:
        """``codex app-server`` plus the ``-c`` overrides that keep the user's setup out.

        Sandbox, approvals and model travel in the requests, not here. What is
        left is the launcher's ``PATH`` for the commands the agent runs and the
        skill and MCP scopes, rendered by the one-shot provider's own helpers.
        """
        return [
            self._binary,
            "app-server",
            *shell_path_config(self._env),
            *scope_overrides(
                self._env,
                spec.cwd,
                skill_scope=spec.skill_scope,
                mcp_scope=spec.mcp_scope,
                given_keys=settings.mcp_keys,
            ),
        ]

    def _timeouts(self) -> _Timeouts:
        return _Timeouts(self._startup_timeout, self._close_timeout, self._interrupt_grace)


@dataclass(frozen=True)
class _Timeouts:
    startup: float
    close: float
    interrupt_grace: float


@dataclass(frozen=True)
class _Settings:
    """What the transport worked out from a spec before starting anything."""

    permissions: CodexPermissions
    config: JsonObject | None
    mcp_keys: frozenset[str]
    servers: tuple[McpServer, ...]
    pricing: ModelPricing | None
    pricing_table: PricingTable | None
    version: str


def _no_items() -> list[ThreadItem]:
    return []


@dataclass(frozen=True)
class _Steer:
    """A message sent with ``turn/steer`` whose outcome is still to be reported."""

    client_id: str
    text: str


@dataclass
class _Turn:
    """Everything one running turn has learned so far. Only the turn's thread touches it."""

    request: TurnRequest
    started_at: float
    start_id: int = 0
    turn_id: str | None = None
    items: list[ThreadItem] = field(default_factory=_no_items)
    total: Totals | None = None
    completed: Turn | None = None
    error: TurnError | None = None
    fail: TurnFailedError | None = None
    structured: bool = False


class _Conversation:
    """One thread on one process.

    State is split by who touches it. The turn thread owns the channel's read
    side, the baseline and the per-turn state. ``_state`` guards what
    ``interrupt`` and ``close`` also read: whether a turn runs, which turn, and
    whether an interrupt has been asked for or sent.
    """

    def __init__(
        self,
        spawn: Callable[[], Process],
        clock: Clock,
        spec: ConversationSpec,
        settings: _Settings,
        timeouts: _Timeouts,
    ) -> None:
        self._spawn = spawn
        self._channel = Channel(spawn(), self._emit)
        self._clock = clock
        self._spec = spec
        self._settings = settings
        self._timeouts = timeouts
        self._items = ItemEvents()
        self._state = threading.Lock()
        self._running = False
        self._closed = False
        self._turn_id: str | None = None
        self._interrupt_asked = False
        self._interrupt_sent = False
        #: ``turn/steer`` requests awaiting their reply, by request id; guarded by ``_state``.
        self._steer_requests: dict[int, _Steer] = {}
        #: Accepted steers not yet seen as items of the turn, by client id.
        self._steer_items: dict[str, _Steer] = {}
        self._steer_count = 0
        self._thread_id: str | None = None
        self._model = spec.model
        self._pricing = settings.pricing
        self._baseline: Totals | None = (
            zero_totals() if spec.resume_id is None else baseline_from(spec.previous_usage)
        )
        self._turns_run = 0
        self._sink: Callable[[AgentEvent], None] | None = None
        self._deferred: list[AgentEvent] = []
        self._skills: SkillTracker | None = None
        self._failed_servers: dict[str, str] = {}
        self._deferred_failure: TurnFailedError | None = None
        self._stuck = False

    # -- Conversation protocol

    @property
    def conversation_id(self) -> str | None:
        """The Codex thread id."""
        return self._thread_id

    def turn(self, request: TurnRequest, emit: Callable[[AgentEvent], None]) -> TurnResult:
        """Run one turn to completion on the calling thread."""
        self._check_request(request)
        schema = self._output_schema(request)
        self._begin()
        try:
            return self._run(request, emit, schema)
        finally:
            self._end()

    def interrupt(self) -> None:
        """Ask the server to end the running turn. Thread-safe; idle conversations ignore it."""
        with self._state:
            if not self._running or self._closed:
                return
        self._ask_for_interrupt()

    def steer(self, text: str) -> None:
        """Send ``turn/steer`` for the running turn. Thread-safe.

        The server folds the message into the turn after the work in flight (a
        running command finishes first). Its answer arrives on the turn's
        thread: ``SteerDelivered`` when accepted, ``SteerRejected`` with the
        server's reason otherwise, then ``SteerConsumed`` when the message
        becomes an item of the turn. Raises ``NoRunningTurnError`` while no
        turn is running, before the server has named the turn, and once it has
        reported the turn over.
        """
        with self._state:
            turn_id = self._turn_id
            if self._closed or not self._running or turn_id is None:
                msg = "no turn is running that can take a message"
                raise NoRunningTurnError(msg)
            self._steer_count += 1
            steer = _Steer(f"steer-{self._steer_count}", text)
            params = TurnSteerParams(
                thread_id=self._thread_id or "",
                expected_turn_id=turn_id,
                input=(UserInputText(text=text),),
                client_user_message_id=steer.client_id,
            )
            try:
                request_id = self._channel.request(params)
            except Gone as gone:
                msg = "the codex app-server is gone, so the message was not delivered"
                raise NoRunningTurnError(msg) from gone
            self._steer_requests[request_id] = steer

    def close(self) -> None:
        """Close stdin, then terminate and kill the process if it lingers. Idempotent."""
        with self._state:
            if self._closed:
                return
            self._closed = True
        _stop(self._channel.process, self._timeouts.close)

    # -- start-up

    def start(self) -> None:
        """Handshake, then start or resume the thread."""
        self._boot()

    def _boot(self) -> None:
        """Handshake on the current channel, then start the thread or resume it."""
        deadline = Deadline(self._clock, self._timeouts.startup)
        try:
            self._handshake(deadline)
            self._open_thread(deadline)
        except Expired:
            msg = f"codex app-server did not finish starting within {self._timeouts.startup}s"
            raise TurnTimeoutError(self._timeouts.startup, msg) from None
        except Gone as gone:
            raise self._gone_error(gone) from None

    def _handshake(self, deadline: Deadline) -> None:
        info = ClientInfo(name="agentshim", version=self._settings.version)
        self._call(InitializeParams(client_info=info), deadline)
        self._channel.notify(INITIALIZED)

    def _open_thread(self, deadline: Deadline) -> None:
        spec, perms = self._spec, self._settings.permissions
        resume_id = self._thread_id or spec.resume_id
        if resume_id is None:
            result = self._call(
                ThreadStartParams(
                    cwd=spec.cwd,
                    model=spec.model,
                    sandbox=perms.sandbox,
                    approval_policy=perms.approval,
                    config=self._settings.config,
                    ephemeral=False,
                ),
                deadline,
            )
            started = ThreadStartResponse.from_wire(result)
            self._adopt(started.thread.id, started.model, started.sandbox, started.approval_policy)
            return
        result = self._call(
            ThreadResumeParams(
                thread_id=resume_id,
                cwd=spec.cwd,
                model=spec.model,
                sandbox=perms.sandbox,
                approval_policy=perms.approval,
                config=self._settings.config,
                exclude_turns=True,
            ),
            deadline,
            refusal=resume_id,
        )
        resumed = ThreadResumeResponse.from_wire(result)
        self._adopt(resumed.thread.id, resumed.model, resumed.sandbox, resumed.approval_policy)

    def _adopt(
        self, thread_id: str, model: str, sandbox: SandboxPolicy, approval: AskForApproval
    ) -> None:
        """Take the thread the server reports, after checking the permissions it applied."""
        check_applied(self._settings.permissions, sandbox, approval)
        self._thread_id = thread_id
        if self._pricing is None:
            self._pricing = price_for(APP_SERVER_PROFILE.name, model, self._settings.pricing_table)

    def _call(
        self, params: ClientRequestParams, deadline: Deadline, *, refusal: str | None = None
    ) -> JsonValue:
        """Send a request and wait for its answer, handling whatever else arrives first."""
        request_id = self._channel.request(params)
        while True:
            message = self._channel.read(deadline)
            if isinstance(message, (Response, ErrorResponse)) and message.id == request_id:
                self._channel.settle(request_id)
                if isinstance(message, ErrorResponse):
                    raise self._rpc_error(params.METHOD, message, refusal)
                return message.result
            self._handle(message, None)

    @staticmethod
    def _rpc_error(method: str, message: ErrorResponse, refusal: str | None) -> AgentShimError:
        text = message.error.message
        if refusal is not None and any(marker in text for marker in _RESUME_REFUSALS):
            return SessionResumeError(
                ("codex", "app-server"), message.error.code, refusal, detail=text
            )
        return TurnFailedError(
            f"codex {method} failed: {text}", kind=FailureKind.OTHER, detail=text
        )

    # -- request checks

    def _check_request(self, request: TurnRequest) -> None:
        if request.extra_args:
            msg = "extra_args cannot be passed to a running codex app-server"
            raise ProviderCapabilityError(msg)
        if request.env is not None:
            msg = "a turn cannot change the environment of a running codex app-server"
            raise ProviderCapabilityError(msg)
        if request.mcp_servers and tuple(request.mcp_servers) != self._settings.servers:
            msg = "MCP servers are fixed for the life of a codex app-server conversation"
            raise ProviderCapabilityError(msg)

    @staticmethod
    def _output_schema(request: TurnRequest) -> JsonValue | None:
        schema = request.output_schema
        if schema is None:
            return None
        dialect = APP_SERVER_PROFILE.schema_dialect
        problems = [] if dialect is None else dialect_problems(schema.schema, dialect)
        if problems:
            raise SchemaDialectError(problems)
        return cast("JsonValue", json.loads(json.dumps(schema.schema)))

    # -- turns

    def _begin(self) -> None:
        with self._state:
            if self._closed:
                msg = "conversation is closed"
                raise SessionStateError(msg)
            if self._running:
                msg = "a turn is already running in this conversation"
                raise SessionStateError(msg)
            if self._stuck:
                msg = "the previous turn did not stop, so this conversation cannot take another"
                raise TurnFailedError(msg, kind=FailureKind.OTHER, detail=msg)
            self._running = True
            self._turn_id = None
            self._interrupt_asked = False
            self._interrupt_sent = False
            self._steer_items.clear()

    def _end(self) -> None:
        self._reject_unanswered_steers()
        self._sink = None
        self._skills = None
        with self._state:
            self._running = False
            self._turn_id = None

    def _run(
        self, request: TurnRequest, emit: Callable[[AgentEvent], None], schema: JsonValue | None
    ) -> TurnResult:
        self._ensure_process()
        self._raise_standing_failure()
        self._skills = SkillTracker(APP_SERVER_PROFILE)
        self._sink = emit
        for event in self._deferred:
            self._emit(event)
        self._deferred.clear()
        turn = _Turn(request, self._clock.monotonic(), structured=schema is not None)
        thread_id = self._thread_id or ""
        self._emit(SessionStarted(thread_id))
        deadline = Deadline(self._clock, request.timeout)
        try:
            turn.start_id = self._channel.request(self._turn_params(request, thread_id, schema))
            while turn.completed is None:
                self._handle(self._channel.read(deadline), turn)
        except Expired:
            self._wind_down(turn)
            raise TurnTimeoutError(request.timeout or 0.0) from None
        except Gone as gone:
            raise self._gone_error(gone) from None
        except BaseException:
            self._wind_down(turn)
            raise
        return self._finish(turn, thread_id)

    def _ensure_process(self) -> None:
        """Between turns, replace a process that died by resuming its thread in a new one.

        A crash, a kill or an exit while idle leaves a channel that can only
        fail. The thread lives on disk, so a new ``codex app-server`` can pick
        it up with ``thread/resume``. If Codex refuses, ``SessionResumeError``
        propagates and the session replaces the conversation. The token
        baseline is kept: the resumed thread reports the same cumulative totals.
        """
        if self._thread_id is None:
            return
        try:
            while (message := self._channel.poll()) is not None:
                self._handle(message, None)
        except Gone:
            pass
        else:
            return
        self._channel.drain()
        old = self._channel.process
        process = self._spawn()
        channel = Channel(process, self._emit)
        with self._state:
            closed = self._closed
            if not closed:
                self._channel = channel
        if closed:
            _stop(process, self._timeouts.close)
            msg = "the conversation was closed during the turn"
            raise TurnCancelledError(msg)
        _stop(old, self._timeouts.close)
        self._items = ItemEvents()
        self._failed_servers.clear()
        self._deferred_failure = None
        try:
            self._boot()
        except BaseException:
            _stop(process, self._timeouts.close)
            raise

    def _raise_standing_failure(self) -> None:
        if self._failed_servers:
            name, error = next(iter(self._failed_servers.items()))
            raise _mcp_failure(name, error)
        failure, self._deferred_failure = self._deferred_failure, None
        if failure is not None:
            raise failure

    def _turn_params(
        self, request: TurnRequest, thread_id: str, schema: JsonValue | None
    ) -> TurnStartParams:
        spec, perms = self._spec, self._settings.permissions
        return TurnStartParams(
            thread_id=thread_id,
            input=(UserInputText(text=request.prompt),),
            cwd=request.cwd or spec.cwd,
            model=spec.model,
            effort=request.reasoning_effort or spec.reasoning_effort,
            approval_policy=perms.approval,
            sandbox_policy=perms.policy,
            output_schema=schema,
        )

    # -- messages

    def _handle(self, message: ServerMessage, turn: _Turn | None) -> None:
        if isinstance(message, ServerRequest):
            self._on_request(message, turn)
        elif isinstance(message, Notification):
            self._on_notification(message, turn)
        else:
            self._on_reply(message, turn)

    def _on_reply(self, message: Response | ErrorResponse, turn: _Turn | None) -> None:
        if message.id is None or not isinstance(message.id, int):
            return
        method = self._channel.settle(message.id)
        with self._state:
            steer = self._steer_requests.pop(message.id, None)
        if steer is not None:
            self._on_steer_reply(steer, message)
            return
        if method is None or turn is None or message.id != turn.start_id:
            return  # an interrupt's acknowledgement, or a reply nobody is waiting for
        if isinstance(message, ErrorResponse):
            raise self._rpc_error(method, message, None)
        self._learn_turn_id(turn, TurnStartResponse.from_wire(message.result).turn.id)

    def _reject_unanswered_steers(self) -> None:
        """Report the steers whose answer never came before the turn ended."""
        with self._state:
            lost = list(self._steer_requests.values())
            self._steer_requests.clear()
            self._steer_items.clear()
        for steer in lost:
            self._emit(SteerRejected(steer.text, "the turn ended before codex answered"))

    def _on_steer_reply(self, steer: _Steer, message: Response | ErrorResponse) -> None:
        if isinstance(message, ErrorResponse):
            self._emit(SteerRejected(steer.text, message.error.message))
            return
        with self._state:
            self._steer_items[steer.client_id] = steer
        self._emit(SteerDelivered(steer.text))

    def _on_request(self, request: ServerRequest, turn: _Turn | None) -> None:
        answer = answer_request(request, self._spec.approvals)
        self._channel.reply(answer.response)
        refusal = answer.refusal
        if refusal is None:
            self._emit(Lifecycle("unsupported_request", request.method))
        elif self._spec.approvals is ApprovalPolicy.DENY:
            self._emit(ApprovalDenied(refusal.kind, refusal.detail))
        else:
            msg = f"codex asked for {refusal.kind} approval: {refusal.detail}"
            failure = TurnFailedError(msg, kind=FailureKind.OTHER, detail=msg)
            if turn is None:
                self._deferred_failure = failure
            else:
                turn.fail = turn.fail or failure
                self._ask_for_interrupt()

    def _on_notification(self, note: Notification, turn: _Turn | None) -> None:
        params = note.params
        if isinstance(params, McpServerStatusUpdatedNotification):
            self._on_mcp_status(params, turn)
        elif isinstance(params, WarningNotification):
            self._emit(Lifecycle("warning", params.message))
        elif isinstance(params, AccountRateLimitsUpdatedNotification):
            for status in rate_limit_statuses(params.rate_limits):
                self._emit(status)
        elif turn is not None:
            self._on_turn_notification(params, turn)

    def _on_turn_notification(self, params: object, turn: _Turn) -> None:
        if isinstance(params, TurnStartedNotification):
            self._on_turn_started(params, turn)
        elif isinstance(params, (ItemStartedNotification, ItemCompletedNotification)):
            self._on_item(params, turn)
        elif isinstance(params, ThreadTokenUsageUpdatedNotification):
            if self._is_mine(params.thread_id, params.turn_id, turn):
                turn.total = totals_of(params.token_usage.total)
        elif isinstance(params, ErrorNotification):
            self._on_error(params, turn)
        elif isinstance(params, TurnCompletedNotification):
            self._on_turn_completed(params, turn)

    def _is_mine(self, thread_id: str, turn_id: str | None, turn: _Turn) -> bool:
        """Whether a notification belongs to this thread and, once known, this turn."""
        if thread_id != self._thread_id:
            return False
        return turn.turn_id is not None and (turn_id is None or turn_id == turn.turn_id)

    def _on_turn_started(self, params: TurnStartedNotification, turn: _Turn) -> None:
        if params.thread_id == self._thread_id and turn.turn_id in (None, params.turn.id):
            self._learn_turn_id(turn, params.turn.id)
            self._emit(Lifecycle("turn_started", params.turn.id))

    def _on_item(
        self, params: ItemStartedNotification | ItemCompletedNotification, turn: _Turn
    ) -> None:
        if not self._is_mine(params.thread_id, params.turn_id, turn):
            return
        now = self._clock.monotonic()
        self._note_steer_item(params.item)
        if isinstance(params, ItemStartedNotification):
            events = self._items.started(params.item, now)
        else:
            turn.items.append(params.item)
            events = self._items.completed(params.item, now)
        for event in events:
            self._emit(event)

    def _note_steer_item(self, item: ThreadItem) -> None:
        """Report a steered message the first time it shows up as an item of the turn."""
        if not isinstance(item, ThreadItemUserMessage) or item.client_id is None:
            return
        with self._state:
            steer = self._steer_items.pop(item.client_id, None)
        if steer is not None:
            self._emit(SteerConsumed(steer.text))

    def _on_error(self, params: ErrorNotification, turn: _Turn) -> None:
        if not self._is_mine(params.thread_id, params.turn_id, turn):
            return
        prefix = "retrying: " if params.will_retry else ""
        self._emit(ProviderError(prefix + params.error.message))
        if not params.will_retry:
            turn.error = params.error

    def _on_turn_completed(self, params: TurnCompletedNotification, turn: _Turn) -> None:
        if params.thread_id != self._thread_id:
            return
        if turn.turn_id is None:
            self._learn_turn_id(turn, params.turn.id)
        if params.turn.id == turn.turn_id:
            turn.completed = params.turn
            with self._state:
                self._turn_id = None  # a steer for a finished turn would only be refused

    def _on_mcp_status(
        self, params: McpServerStatusUpdatedNotification, turn: _Turn | None
    ) -> None:
        if params.thread_id not in (None, self._thread_id):
            return
        if params.name not in self._settings.mcp_keys:
            return
        if params.status == McpServerStartupState.FAILED:
            error = params.error or "failed to start"
            self._failed_servers.setdefault(params.name, error)
            self._emit(Lifecycle("mcp_server_failed", f"{params.name}: {error}"))
            if turn is not None:
                turn.fail = turn.fail or _mcp_failure(params.name, error)
                self._ask_for_interrupt()

    # -- interrupting

    def _learn_turn_id(self, turn: _Turn, turn_id: str) -> None:
        if turn.turn_id is not None:
            return
        turn.turn_id = turn_id
        with self._state:
            self._turn_id = turn_id
            self._send_pending_interrupt()

    def _ask_for_interrupt(self) -> None:
        with self._state:
            self._interrupt_asked = True
            self._send_pending_interrupt()

    def _send_pending_interrupt(self) -> None:
        """Send ``turn/interrupt`` if one is asked for, unsent, and the turn is known.

        The caller holds ``_state``. A process that is already gone has nothing
        left to interrupt, so a failed write is dropped.
        """
        if not self._interrupt_asked or self._interrupt_sent or self._turn_id is None:
            return
        self._interrupt_sent = True
        try:
            self._channel.request(
                TurnInterruptParams(thread_id=self._thread_id or "", turn_id=self._turn_id)
            )
        except Gone:
            return

    def _wind_down(self, turn: _Turn) -> None:
        """Stop a turn that is being abandoned and wait, within bounds, for the server to agree.

        If the server never reports the turn over, the conversation is written
        off: another ``turn/start`` on it could collide with the one still running.
        """
        if turn.turn_id is None or turn.completed is not None:
            return
        self._ask_for_interrupt()
        self._sink = _drop
        deadline = Deadline(self._clock, self._timeouts.interrupt_grace)
        try:
            while turn.completed is None:
                self._handle(self._channel.read(deadline), turn)
        except (Expired, Gone, AgentShimError):
            self._stuck = True

    # -- finishing

    def _finish(self, turn: _Turn, thread_id: str) -> TurnResult:
        completed = turn.completed
        if completed is None:  # pragma: no cover - the loop only ends once it is set
            msg = "turn ended without a completion"
            raise AssertionError(msg)
        self._channel.settle(turn.start_id)
        if turn.fail is not None:
            raise turn.fail
        status = completed.status
        if status not in (TurnStatus.COMPLETED, TurnStatus.INTERRUPTED):
            raise self._turn_failure(turn, completed)
        interrupted = status == TurnStatus.INTERRUPTED
        resumed = self._turns_run > 0 or self._spec.resume_id is not None
        usage = self._usage(turn)
        text = final_text(_agent_messages(turn.items) or _agent_messages(completed.items))
        self._turns_run += 1
        self._emit(Lifecycle("turn_completed", f"status={_text(status)}"))
        self._emit(UsageReport(usage, usage.total_cost_usd))
        if interrupted:
            self._emit(TurnInterrupted())
        duration_ms = int((self._clock.monotonic() - turn.started_at) * _MS_PER_S)
        skills = self._skills.summary() if self._skills is not None else None
        result = TurnResult(
            text=text,
            structured_output=_structured(text) if turn.structured else None,
            session_id=thread_id,
            resumed=resumed,
            usage=usage,
            cost_usd=usage.total_cost_usd,
            duration_ms=duration_ms,
            exit_code=0,
            interrupted=interrupted,
        )
        return result if skills is None else replace(result, skills=skills)

    def _turn_failure(self, turn: _Turn, completed: Turn) -> TurnFailedError:
        error = completed.error or turn.error
        if error is None:
            detail = f"turn ended with status {_text(completed.status)}"
            return TurnFailedError(
                f"codex turn failed: {detail}", kind=FailureKind.OTHER, detail=detail
            )
        detail = error_text(error)
        return TurnFailedError(
            f"codex turn failed: {detail}", kind=classify_turn_error(error), detail=detail
        )

    def _usage(self, turn: _Turn) -> ProviderUsage:
        usage = turn_usage(turn.total, self._baseline)
        pricing = self._pricing
        if usage.increment_known and pricing is not None:
            usage = replace(usage, total_cost_usd=cost_usd(usage.tokens, pricing))
        if turn.total is not None:
            self._baseline = turn.total
        return usage

    def _gone_error(self, gone: Gone) -> AgentShimError:
        with self._state:
            closed = self._closed
        if closed:
            return TurnCancelledError("the conversation was closed during the turn")
        self._channel.drain()
        tail = self._channel.stderr_tail()
        code = self._channel.returncode if gone.returncode is None else gone.returncode
        msg = f"codex app-server exited with code {code}"
        detail = tail or msg
        return TurnFailedError(
            f"{msg}: {tail}" if tail else msg, kind=FailureKind.OTHER, detail=detail
        )

    # -- events

    def _emit(self, event: AgentEvent) -> None:
        """Deliver *event* to the running turn, or hold it for the next one."""
        skills = self._skills
        if skills is not None:
            skills.on_event(event)
        sink = self._sink
        if sink is None:
            self._deferred.append(event)
        else:
            sink(event)


def _stop(process: Process, wait: float) -> None:
    """Close stdin, then terminate and kill the process if it lingers."""
    process.close_stdin()
    if process.wait(wait) is None:
        process.terminate()
        if process.wait(wait) is None:
            process.kill()
            process.wait(wait)


def _agent_messages(items: Sequence[ThreadItem]) -> list[ThreadItemAgentMessage]:
    return [item for item in items if isinstance(item, ThreadItemAgentMessage)]


def final_text(messages: Sequence[ThreadItemAgentMessage]) -> str:
    """The answer among a turn's agent messages.

    The last message Codex marked ``final_answer``; failing that the last one
    that is not commentary, then the last of any kind, then nothing.
    """
    chosen = (
        [m for m in messages if m.phase == "final_answer"]
        or [m for m in messages if m.phase != "commentary"]
        or list(messages)
    )
    return chosen[-1].text if chosen else ""


def _text(status: str) -> str:
    """A status as the wire spells it (an enum member would print its class name)."""
    return str(status.value) if isinstance(status, Enum) else status


def _structured(text: str) -> object | None:
    try:
        return json.loads(text)
    except ValueError:
        return None


def _mcp_failure(name: str, error: str) -> TurnFailedError:
    msg = f"MCP server {name!r} failed to start: {error}"
    return TurnFailedError(msg, kind=FailureKind.OTHER, detail=msg)


def _drop(event: AgentEvent) -> None:
    """Event sink for a turn being abandoned."""


def _discard(message: str) -> None:
    """Default log sink: drop the message."""


__all__ = ["CodexAppServerTransport"]
