"""Compatibility shim for older ``agentshim.gemini_events`` imports."""

from .gemini.events import GeminiEvent, InitEvent, MessageEvent, ToolResultEvent, ToolUseEvent

__all__ = [
    "GeminiEvent",
    "InitEvent",
    "MessageEvent",
    "ToolResultEvent",
    "ToolUseEvent",
]
