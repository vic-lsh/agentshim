import inspect
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeVar

from agentshim.trajectory import NullTrajectoryRecorder, TrajectoryRecorderProtocol

_T = TypeVar("_T")
_READABLE_NAME_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

_AGENT_REGISTRY: dict[str, Any] = {}


def register_provider(*names: str) -> Any:
    """Decorator to register a coding agent provider.

    Args:
        *names: List of provider names/aliases (case-insensitive).
    """

    def decorator(cls: _T) -> _T:
        for name in names:
            _AGENT_REGISTRY[name.lower()] = cls
        return cls

    return decorator


def _available_providers() -> list[str]:
    return sorted(_AGENT_REGISTRY)


def _resolve_provider(provider: str) -> Any:
    provider_key = provider.lower()
    agent_cls = _AGENT_REGISTRY.get(provider_key)
    if agent_cls is None:
        raise ValueError(
            f"Unknown coding agent provider '{provider}'. Available providers: {_available_providers()}"
        )
    return agent_cls


def _readable_name_from_class_name(class_name: str) -> str:
    trimmed = class_name
    for suffix in ("CodingAgent", "Agent"):
        if trimmed.endswith(suffix):
            trimmed = trimmed[: -len(suffix)]
            break
    return _READABLE_NAME_BOUNDARY.sub(" ", trimmed).strip() or class_name


class BaseCodingAgent(ABC):
    """Abstract base class for coding agents."""

    recorder: TrajectoryRecorderProtocol = NullTrajectoryRecorder()
    event_handler: Any | None = None

    @property
    def readable_name(self) -> str:
        """Human-readable name for logs and UI."""
        return _readable_name_from_class_name(self.__class__.__name__)

    @property
    def backend_class_name(self) -> str:
        """Concrete backend class name."""
        return self.__class__.__name__

    def start_session(
        self,
        cwd: str | None = None,
        timeout: int = 300,
        silent: bool = False,
    ) -> "BaseAgentSession":
        """Open a stateful session if supported by the backend."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support start_session()")

    @abstractmethod
    def generate(
        self,
        prompt: str,
        cwd: str | None = None,
        timeout: int = 300,
        silent: bool = False,
    ) -> str:
        """One-shot prompt → reply. Equivalent to a fresh ``start_session``
        followed by a single ``session.generate(prompt)``; no conversation
        state is retained. Use ``start_session`` for multi-turn flows.

        Args:
            prompt: The prompt to send to the agent.
            cwd: Optional working directory context.
            timeout: Timeout in seconds.
            silent: If True, suppress stdout printing of the agent's output.

        Returns:
            Generated text.
        """


class BaseAgentSession(ABC):
    """Abstract base class for stateful coding-agent sessions."""

    session_id: str | None = None

    @abstractmethod
    def generate(
        self,
        prompt: str,
        cwd: str | None = None,
        timeout: int | None = None,
        silent: bool | None = None,
        on_process_started: Callable[[Any], None] | None = None,
    ) -> str:
        """Send ``prompt`` within an existing chat session."""


class CodingAgent(BaseCodingAgent):
    """Concrete provider-routing coding agent.

    Construct with a provider name and shared options, and this facade will
    instantiate the appropriate registered backend internally.
    """

    def __init__(
        self,
        provider: str,
        model: str | None = None,
        recorder: TrajectoryRecorderProtocol | None = None,
        event_handler: Any | None = None,
        mcp_servers: list[Any] | None = None,
        sandbox: Any = False,
        **kwargs: Any,
    ):
        self.provider = provider.lower()
        provider_cls = _resolve_provider(self.provider)

        candidate_kwargs: dict[str, Any] = dict(kwargs)
        if model is not None:
            candidate_kwargs["model"] = model
        if recorder is not None:
            candidate_kwargs["recorder"] = recorder
        if event_handler is not None:
            candidate_kwargs["event_handler"] = event_handler
        if mcp_servers is not None:
            candidate_kwargs["mcp_servers"] = mcp_servers
        if sandbox is not False:
            candidate_kwargs["sandbox"] = sandbox

        signature = inspect.signature(provider_cls.__init__)
        parameters = signature.parameters
        accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())

        supported_kwargs: dict[str, Any] = {}
        unsupported: list[str] = []
        for name, value in candidate_kwargs.items():
            if accepts_kwargs or name in parameters:
                supported_kwargs[name] = value
            else:
                unsupported.append(name)

        if unsupported:
            unsupported_args = ", ".join(sorted(unsupported))
            raise TypeError(
                f"{provider_cls.__name__} does not accept argument(s): {unsupported_args}"
            )

        self._backend: BaseCodingAgent = provider_cls(**supported_kwargs)

    @property
    def backend(self) -> BaseCodingAgent:
        """Return the concrete backend instance."""
        return self._backend

    @property
    def readable_name(self) -> str:
        return self._backend.readable_name

    @property
    def backend_class_name(self) -> str:
        return self._backend.backend_class_name

    @property
    def model(self) -> Any:
        return getattr(self._backend, "model", None)

    @model.setter
    def model(self, value: Any) -> None:
        self._backend.model = value  # type: ignore[attr-defined]

    @property
    def recorder(self) -> TrajectoryRecorderProtocol:
        return self._backend.recorder

    @recorder.setter
    def recorder(self, value: TrajectoryRecorderProtocol) -> None:
        self._backend.recorder = value

    @property
    def event_handler(self) -> Any | None:
        return getattr(self._backend, "event_handler", None)

    @event_handler.setter
    def event_handler(self, value: Any | None) -> None:
        self._backend.event_handler = value

    def start_session(
        self,
        cwd: str | None = None,
        timeout: int = 300,
        silent: bool = False,
    ) -> BaseAgentSession:
        return self._backend.start_session(cwd=cwd, timeout=timeout, silent=silent)

    def generate(
        self,
        prompt: str,
        cwd: str | None = None,
        timeout: int = 300,
        silent: bool = False,
    ) -> str:
        return self._backend.generate(prompt, cwd=cwd, timeout=timeout, silent=silent)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)
