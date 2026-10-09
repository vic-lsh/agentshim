"""The agent: a provider reached through a transport, handing out sessions.

Composition layer. ``Agent`` is where a provider name, an executor and a
confinement become a ``Transport``, and where the injected clock and id
allocator are bound to every ``Session`` it makes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentshim.core.clock import SystemClock
from agentshim.core.conversation import ConversationSpec, TransportKind
from agentshim.core.ids import RandomIds
from agentshim.core.profile import ConfigScope, McpScope, SkillScope
from agentshim.core.session_policy import PolicyConfig, RetryPolicy
from agentshim.execution.confinement import confine
from agentshim.execution.host import HostCommandExecutor
from agentshim.oneshot import OneShotTransport
from agentshim.providers import get_stream_transport
from agentshim.session import Session, map_mcp_servers

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from agentshim.core.checkpoints import CheckpointStore
    from agentshim.core.clock import Clock
    from agentshim.core.conversation import Transport
    from agentshim.core.events import AgentEventHandler
    from agentshim.core.ids import IdAllocator
    from agentshim.core.mcp import McpServer
    from agentshim.core.permissions import ApprovalPolicy, NativePermissions
    from agentshim.core.profile import ProviderProfile
    from agentshim.core.usage import ProviderUsage
    from agentshim.execution.confinement import Confinement
    from agentshim.execution.executor import CommandExecutor


class Agent:
    """A provider plus how it runs: transport, permissions, clock, and recovery policy.

    ``Agent("claude", permissions=..., approvals=...)`` reaches the provider
    through the one-shot transport; passing a ``Transport`` instance uses it
    as is. ``permissions`` and ``approvals`` have no defaults: what the agent
    may do is always the caller's decision.
    """

    # Each argument is an independent documented option of the public constructor.
    def __init__(  # noqa: PLR0913
        self,
        provider: str | Transport,
        *,
        model: str | None = None,
        executor: CommandExecutor | None = None,
        confinement: Confinement | None = None,
        permissions: NativePermissions,
        approvals: ApprovalPolicy,
        clock: Clock | None = None,
        ids: IdAllocator | None = None,
        retry: RetryPolicy | None = None,
        event_handlers: Sequence[AgentEventHandler] = (),
        env: Mapping[str, str] | None = None,
        log: Callable[[str], None] | None = None,
        check_timeout: float = 15.0,
        transport: TransportKind = TransportKind.ONE_SHOT,
    ) -> None:
        """Build the transport and bind the policy inputs.

        With a provider *name*, the transport is chosen by *transport*: the
        ``OneShotTransport`` (one process per turn, the default for now) or
        the provider's long-lived ``TransportKind.STREAM`` transport, on
        *executor* (the local host by default). A *confinement* wraps that
        executor so every command runs inside it, and its ``env`` becomes the
        agent's environment, so *env* may not also be given. With a ready-made
        ``Transport`` none of *executor*, *confinement*, *env* applies and
        passing one, or a *transport* kind, is an error: they would be silently
        ignored.
        """
        self._permissions = permissions
        self._approvals = approvals
        self._model = model
        self._confinement = confinement
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._ids: IdAllocator = ids if ids is not None else RandomIds()
        self._retry = retry if retry is not None else RetryPolicy()
        self._event_handlers = tuple(event_handlers)
        self._log = log
        if isinstance(provider, str):
            self._transport: Transport = self._by_name(
                provider, transport, executor, confinement, env, check_timeout
            )
        else:
            if (
                executor is not None
                or confinement is not None
                or env is not None
                or transport is not TransportKind.ONE_SHOT
            ):
                msg = (
                    "executor, confinement, env and transport belong to the transport "
                    "that was passed in"
                )
                raise ValueError(msg)
            self._transport = provider

    # Each argument is an independent constructor option, passed through.
    def _by_name(  # noqa: PLR0913
        self,
        provider: str,
        kind: TransportKind,
        executor: CommandExecutor | None,
        confinement: Confinement | None,
        env: Mapping[str, str] | None,
        check_timeout: float,
    ) -> Transport:
        base = executor if executor is not None else HostCommandExecutor()
        if confinement is not None:
            if env is not None:
                msg = (
                    "env conflicts with confinement: the confinement supplies the agent environment"
                )
                raise ValueError(msg)
            base = confine(base, confinement)
            env = dict(confinement.env)
        if kind is TransportKind.STREAM:
            return get_stream_transport(
                provider,
                executor=base,
                env=env,
                clock=self._clock,
                ids=self._ids,
                check_timeout=check_timeout,
                log=self._log,
            )
        return OneShotTransport(
            provider, executor=base, env=env, check_timeout=check_timeout, log=self._log
        )

    @property
    def transport(self) -> Transport:
        """The transport sessions open conversations on."""
        return self._transport

    @property
    def profile(self) -> ProviderProfile:
        """What the provider can do."""
        return self._transport.profile

    # Each keyword is an independent documented session option.
    def session(  # noqa: PLR0913
        self,
        cwd: str,
        *,
        mcp_servers: Sequence[McpServer] = (),
        reasoning_effort: str | None = None,
        checkpoints: CheckpointStore | None = None,
        checkpoint_key: str | None = None,
        idle_release_after: float | None = None,
        resume_id: str | None = None,
        previous_usage: ProviderUsage | None = None,
        skill_scope: SkillScope = SkillScope.ALL,
        mcp_scope: McpScope = McpScope.ALL,
        config_scope: ConfigScope = ConfigScope.ALL,
    ) -> Session:
        """Open a session working in *cwd* (a host path).

        ``mcp_servers`` and ``reasoning_effort`` are fixed for the session.
        With ``checkpoints`` and ``checkpoint_key`` the session resumes the
        conversation saved there and saves it after every turn; ``resume_id``
        (with ``previous_usage``) names a conversation explicitly and wins over
        a checkpoint. ``idle_release_after`` seconds of inactivity release the
        live conversation before the next turn. The scope arguments are
        transitional, for the one-shot transport only.
        """
        agent_path = self._confinement.agent_path if self._confinement is not None else None
        spec = ConversationSpec(
            cwd=agent_path(cwd) if agent_path is not None else cwd,
            model=self._model,
            permissions=self._permissions,
            approvals=self._approvals,
            mcp_servers=(
                map_mcp_servers(mcp_servers, agent_path)
                if agent_path is not None
                else tuple(mcp_servers)
            ),
            reasoning_effort=reasoning_effort,
            resume_id=resume_id,
            previous_usage=previous_usage,
            skill_scope=skill_scope,
            mcp_scope=mcp_scope,
            config_scope=config_scope,
        )
        profile = self.profile
        return Session(
            self._transport,
            spec,
            clock=self._clock,
            ids=self._ids,
            policy=PolicyConfig(
                retry=self._retry,
                renewal=profile.renewal,
                supports_resume=profile.supports_resume,
                idle_release_after=idle_release_after,
            ),
            event_handlers=self._event_handlers,
            checkpoints=checkpoints,
            checkpoint_key=checkpoint_key,
            previous_usage=previous_usage,
            agent_path=agent_path,
            host_cwd=cwd,
            log=self._log,
        )
