from abc import ABC, abstractmethod
from typing import Any, TypeVar

from agentshim.trajectory import NullTrajectoryRecorder, TrajectoryRecorderProtocol

_T = TypeVar("_T")

AGENT_REGISTRY: dict[str, Any] = {}


def register_provider(*names: str) -> Any:
    """Decorator to register a coding agent provider.

    Args:
        *names: List of provider names/aliases (case-insensitive).
    """

    def decorator(cls: _T) -> _T:
        for name in names:
            AGENT_REGISTRY[name.lower()] = cls
        return cls

    return decorator


class CodingAgent(ABC):
    """Abstract base class for coding agents."""

    recorder: TrajectoryRecorderProtocol = NullTrajectoryRecorder()
    event_handler: Any | None = None

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
