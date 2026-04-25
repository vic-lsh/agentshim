from __future__ import annotations

import inspect
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from agentshim.events import AgentEventHandler
from agentshim.mcp_config import McpServerConfig
from agentshim.sandbox import SandboxConfig

_T = TypeVar("_T")
_READABLE_NAME_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_PROVIDER_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _readable_name_from_class_name(class_name: str) -> str:
    trimmed = class_name
    for suffix in ("CodingAgent", "Agent"):
        if trimmed.endswith(suffix):
            trimmed = trimmed[: -len(suffix)]
            break
    return _READABLE_NAME_BOUNDARY.sub(" ", trimmed).strip() or class_name


class BaseCodingAgent(ABC):
    """Abstract base class for coding agents."""

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
    ) -> BaseAgentSession:
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


ProviderClass = type[BaseCodingAgent]


class ProviderRegistry:
    """Registry for coding-agent provider classes.

    Provider names are normalized to lowercase. Each provider has one canonical
    name plus zero or more aliases. Registration is import-driven: importing a
    module with a ``@register_provider(...)`` decorator mutates this registry
    for the current Python process.
    """

    def __init__(self) -> None:
        self._providers: dict[str, ProviderClass] = {}
        self._canonical_names: dict[str, str] = {}

    def _normalize_name(self, name: str) -> str:
        if not isinstance(name, str):
            raise TypeError(f"provider name must be a string, got {type(name).__name__}")
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("provider name must not be empty")
        if not _PROVIDER_NAME_PATTERN.fullmatch(normalized):
            raise ValueError(f"invalid provider name '{name}'; use lowercase letters, digits, hyphens, or underscores")
        return normalized

    def _normalize_names(self, canonical_name: str, aliases: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
        canonical = self._normalize_name(canonical_name)
        normalized_aliases = tuple(
            dict.fromkeys(self._normalize_name(alias) for alias in aliases if alias != canonical_name)
        )
        return canonical, normalized_aliases

    def _validate_provider_class(self, cls: type[Any]) -> ProviderClass:
        if not inspect.isclass(cls):
            raise TypeError(f"registered provider must be a class, got {type(cls).__name__}")
        if not issubclass(cls, BaseCodingAgent):
            raise TypeError(f"{cls.__name__} must inherit from BaseCodingAgent")
        if inspect.isabstract(cls):
            raise TypeError(f"{cls.__name__} must be concrete before registration")
        return cls

    def register(
        self,
        cls: type[Any],
        *,
        canonical_name: str,
        aliases: tuple[str, ...] = (),
        overwrite: bool = False,
    ) -> ProviderClass:
        provider_cls = self._validate_provider_class(cls)
        canonical, normalized_aliases = self._normalize_names(canonical_name, aliases)
        all_names = (canonical, *normalized_aliases)

        collisions = [
            name for name in all_names if name in self._providers and self._providers[name] is not provider_cls
        ]
        if collisions and not overwrite:
            raise ValueError(
                "provider name(s) already registered: "
                + ", ".join(sorted(collisions))
                + ". Pass overwrite=True to replace them."
            )

        for name in all_names:
            self._providers[name] = provider_cls
            self._canonical_names[name] = canonical

        return provider_cls

    def list_providers(self) -> list[str]:
        """Return the sorted canonical provider names."""
        return sorted(set(self._canonical_names.values()))

    def get_provider_class(self, name: str) -> ProviderClass:
        normalized = self._normalize_name(name)
        provider_cls = self._providers.get(normalized)
        if provider_cls is None:
            raise ValueError(f"Unknown coding agent provider '{name}'. Available providers: {self.list_providers()}")
        return provider_cls

    def get_canonical_name(self, name: str) -> str:
        normalized = self._normalize_name(name)
        canonical = self._canonical_names.get(normalized)
        if canonical is None:
            raise ValueError(f"Unknown coding agent provider '{name}'. Available providers: {self.list_providers()}")
        return canonical


_PROVIDER_REGISTRY = ProviderRegistry()


def register_provider(
    canonical_name: str,
    *extra_names: str,
    aliases: tuple[str, ...] = (),
    overwrite: bool = False,
) -> Callable[[type[_T]], _T]:
    """Decorator to register a coding agent provider.

    Registration is import-driven: the decorated class becomes available only
    after its defining module has been imported in the current process.

    Args:
        canonical_name: Stable provider id exposed by ``list_providers()``.
        *extra_names: Backward-compatible positional aliases.
        aliases: Additional aliases for the same provider.
        overwrite: Whether to replace an existing registration for any name.
    """

    all_aliases = (*extra_names, *aliases)
    _PROVIDER_REGISTRY._normalize_names(canonical_name, all_aliases)

    def decorator(cls: type[_T]) -> _T:
        return _PROVIDER_REGISTRY.register(
            cls,
            canonical_name=canonical_name,
            aliases=all_aliases,
            overwrite=overwrite,
        )

    return decorator


def list_providers() -> list[str]:
    """Return the sorted canonical provider names."""
    return _PROVIDER_REGISTRY.list_providers()


def get_provider_class(name: str) -> ProviderClass:
    """Resolve *name* to a registered provider class."""
    return _PROVIDER_REGISTRY.get_provider_class(name)


class CodingAgent(BaseCodingAgent):
    """Concrete provider-routing coding agent.

    Construct with a provider name and shared options, and this facade will
    instantiate the appropriate registered backend internally.
    """

    def __init__(
        self,
        provider: str,
        model: str | None = None,
        event_handler: AgentEventHandler | None = None,
        event_handlers: Sequence[AgentEventHandler] | None = None,
        mcp_servers: Sequence[McpServerConfig] | None = None,
        sandbox: bool | SandboxConfig | None = False,
        backend_kwargs: dict[str, Any] | None = None,
    ) -> None:
        requested_provider = _PROVIDER_REGISTRY.get_canonical_name(provider)
        self.provider = requested_provider
        provider_cls = get_provider_class(provider)

        portable_kwargs: dict[str, Any] = {}
        if model is not None:
            portable_kwargs["model"] = model
        if event_handler is not None:
            portable_kwargs["event_handler"] = event_handler
        if event_handlers is not None:
            portable_kwargs["event_handlers"] = list(event_handlers)
        if mcp_servers is not None:
            portable_kwargs["mcp_servers"] = list(mcp_servers)
        if sandbox is not None and sandbox is not False:
            portable_kwargs["sandbox"] = sandbox

        advanced_kwargs = dict(backend_kwargs or {})
        overlapping_keys = sorted(portable_kwargs.keys() & advanced_kwargs.keys())
        if overlapping_keys:
            raise ValueError(
                "backend_kwargs must not override portable CodingAgent arguments: " + ", ".join(overlapping_keys)
            )

        self._backend: BaseCodingAgent = provider_cls(**portable_kwargs, **advanced_kwargs)

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
