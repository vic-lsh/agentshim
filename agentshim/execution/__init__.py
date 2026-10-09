"""Process transport: how a command is run, not what command to run."""

from __future__ import annotations

from .confinement import Confinement, confine
from .docker import DockerExecConfinement
from .executor import (
    CallbackCommandStreamSink,
    CommandExecutor,
    CommandHandle,
    CommandRequest,
    CommandResult,
    CommandStreamSink,
    NullSink,
)
from .host import HostCommandExecutor, HostProcess, ProcessCommandHandle
from .process import (
    Process,
    ProcessExited,
    ProcessOutput,
    SpawnRequest,
    StderrLine,
    StdoutLine,
)
from .transform import TransformingExecutor

__all__ = [
    "CallbackCommandStreamSink",
    "CommandExecutor",
    "CommandHandle",
    "CommandRequest",
    "CommandResult",
    "CommandStreamSink",
    "Confinement",
    "DockerExecConfinement",
    "HostCommandExecutor",
    "HostProcess",
    "NullSink",
    "Process",
    "ProcessCommandHandle",
    "ProcessExited",
    "ProcessOutput",
    "SpawnRequest",
    "StderrLine",
    "StdoutLine",
    "TransformingExecutor",
    "confine",
]
