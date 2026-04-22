"""Thin litellm wrapper with automatic token tracking and trajectory recording."""

import os
from typing import Any

from agentshim.subagent import litellm_call_with_retry
from agentshim.trajectory import TrajectoryRecorderProtocol


class LiteLLMClient:
    """litellm completion wrapper that accumulates token usage and records to trajectory.

    Every ``complete()`` call:
    - Delegates to ``litellm_call_with_retry`` for retry-on-network-error logic.
    - Accumulates prompt/completion/total tokens in ``_token_usage``.
    - Calls ``recorder.record_token_usage()`` with the running total.

    The caller is responsible for managing conversation history (``messages``).
    This avoids coupling: pass the full list each call, append results yourself.
    """

    def __init__(
        self,
        model: str,
        location: str | None = None,
        recorder: TrajectoryRecorderProtocol | None = None,
    ):
        self.model = model
        self.location = location
        self.recorder = recorder
        self._token_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def complete(self, messages: list[dict[str, Any]], label: str = "llm call") -> str:
        """Call litellm with *messages*, accumulate tokens, and record to trajectory.

        Args:
            messages: Full conversation history to send (caller-managed).
            label: Human-readable label for retry/logging messages.

        Returns:
            Assistant response text.

        Raises:
            Exception: Any non-retried litellm error propagates to the caller.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "cache": {"no-cache": True},
        }
        loc = self.location or os.environ.get("VERTEX_LOCATION")
        if loc:
            kwargs["vertex_location"] = loc

        result = litellm_call_with_retry(kwargs, label=label, token_acc=self._token_usage)

        if self.recorder and hasattr(self.recorder, "record_token_usage"):
            self.recorder.record_token_usage(self._token_usage.copy())  # type: ignore[reportArgumentType]

        return result
