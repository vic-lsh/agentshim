from abc import ABC, abstractmethod
from typing import Any, TypeVar

from .trajectory import NullTrajectoryRecorder, TrajectoryRecorderProtocol

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
        """Generate text/code based on a prompt.

        Args:
            prompt: The prompt to send to the agent.
            cwd: Optional working directory context.
            timeout: Timeout in seconds.
            silent: If True, suppress stdout printing of the agent's output.

        Returns:
            Generated text.
        """
