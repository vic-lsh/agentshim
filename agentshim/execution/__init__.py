"""Process transport: how a command is run, not what command to run."""

from __future__ import annotations

from .executor import (
    CallbackCommandStreamSink,
    CommandExecutor,
    CommandHandle,
    CommandRequest,
    CommandResult,
    CommandStreamSink,
    NullSink,
)
from .host import HostCommandExecutor, ProcessCommandHandle
from .transform import TransformingExecutor

__all__ = [
    "CallbackCommandStreamSink",
    "CommandExecutor",
    "CommandHandle",
    "CommandRequest",
    "CommandResult",
    "CommandStreamSink",
    "HostCommandExecutor",
    "NullSink",
    "ProcessCommandHandle",
    "TransformingExecutor",
]
