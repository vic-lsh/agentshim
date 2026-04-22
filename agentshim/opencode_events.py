"""Compatibility shim for older ``agentshim.opencode_events`` imports."""

from .opencode.events import OpencodeEvent, StepFinishEvent, StepStartEvent, TextEvent, ToolUseEvent

__all__ = [
    "OpencodeEvent",
    "StepFinishEvent",
    "StepStartEvent",
    "TextEvent",
    "ToolUseEvent",
]
