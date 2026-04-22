"""Trajectory protocol and null implementation for the agent_cli library.

This module provides the protocol interface and no-op implementation that
standalone library code can use without depending on app_operator.

The full TrajectoryRecorder implementation lives in app_operator/trajectory.py.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable


class TokenUsage(TypedDict):
    """Token usage statistics from an LLM call."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class FaultInjectionMetadata(TypedDict):
    """Metadata about fault injection for embedding in trajectory JSON."""

    enabled: bool
    num_faults_requested: int
    num_faults_injected: int
    faults: list[Any]
    failed_injections: list[Any]
    fault_ids: list[str]
    categories: list[str]
    severities: list[str]


@runtime_checkable
class TrajectoryRecorderProtocol(Protocol):
    """Protocol for trajectory recorders."""

    def start_phase(self, phase: Any, context: dict[str, Any] | None = None) -> None: ...
    def end_phase(self, status: str | None = None) -> None: ...
    def add_system_message(self, content: str) -> None: ...
    def add_user_message(self, content: str) -> None: ...

    def add_assistant_message(self, content: str, duration: float | None = None) -> None: ...

    def add_tool_call(
        self,
        tool: str,
        args: dict[str, Any],
        stdout: str = "",
        stderr: str = "",
        exit_code: int | None = None,
        duration: float | None = None,
    ) -> None: ...
    def set_phase_status(self, status: str) -> None: ...
    def set_agent_name(self, agent_name: str) -> None: ...
    def set_prompt_version(self, version: str) -> None: ...
    def record_fallback(self) -> None: ...
    def record_prompt_kwargs(self, kwargs: dict[str, Any]) -> None: ...
    def record_rendered_prompt(self, rendered_prompt: str) -> None: ...
    def record_fault_injection(self, metadata: FaultInjectionMetadata) -> None: ...
    def record_token_usage(self, usage: TokenUsage) -> None: ...
    def finalize(self, status: str = "completed") -> Path: ...

    def phase(self, phase: Any, context: dict[str, Any] | None = None) -> Any: ...


class NullTrajectoryRecorder(TrajectoryRecorderProtocol):
    """No-op recorder that formally implements TrajectoryRecorderProtocol.

    Used as a default when no real recorder is needed (e.g. in tests).
    Every method is a no-op, so callers never need to check for None.
    """

    def start_phase(self, phase: Any, context: dict[str, Any] | None = None) -> None:
        pass

    def end_phase(self, status: str | None = None) -> None:
        pass

    def add_system_message(self, content: str) -> None:
        pass

    def add_user_message(self, content: str) -> None:
        pass

    def add_assistant_message(self, content: str, duration: float | None = None) -> None:
        pass

    def add_tool_call(
        self,
        tool: str,
        args: dict[str, Any],
        stdout: str = "",
        stderr: str = "",
        exit_code: int | None = None,
        duration: float | None = None,
    ) -> None:
        pass

    def set_phase_status(self, status: str) -> None:
        pass

    def set_agent_name(self, agent_name: str) -> None:
        pass

    def set_prompt_version(self, version: str) -> None:
        pass

    def record_fallback(self) -> None:
        pass

    def record_prompt_kwargs(self, kwargs: dict[str, Any]) -> None:
        pass

    def record_rendered_prompt(self, rendered_prompt: str) -> None:
        pass

    def record_fault_injection(self, metadata: FaultInjectionMetadata) -> None:
        pass

    def record_token_usage(self, usage: TokenUsage) -> None:
        pass

    @contextmanager
    def phase(self, phase: Any, context: dict[str, Any] | None = None):
        yield self

    def finalize(self, status: str = "completed") -> Path:
        return Path("/dev/null")


def _call_id_provider() -> int | None:
    return None


def _run_id_provider() -> str | None:
    return None


def register_context_providers(
    call_id_fn: Callable[[], int | None],
    run_id_fn: Callable[[], str | None],
) -> None:
    """Register live context providers for trajectory correlation.

    Called by app_operator.trajectory at import time to wire in the real
    thread-local implementations. Standalone consumers get None from the
    default def stubs, which is correct outside of app_operator.
    """
    global _call_id_provider, _run_id_provider
    _call_id_provider = call_id_fn
    _run_id_provider = run_id_fn


def get_current_call_id() -> int | None:
    """Return the current call ID, or None if no trajectory is active."""
    return _call_id_provider()


def get_run_id() -> str | None:
    """Return the current run ID, or None if no trajectory is active."""
    return _run_id_provider()
