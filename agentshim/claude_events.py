"""Compatibility shim for older ``agentshim.claude_events`` imports."""

from .claude.events import (
    ClaudeEvent,
    MultiEvent,
    ResultEvent,
    SystemEvent,
    TextEvent,
    ToolResultEvent,
    ToolUseEvent,
)

__all__ = [
    "ClaudeEvent",
    "MultiEvent",
    "ResultEvent",
    "SystemEvent",
    "TextEvent",
    "ToolResultEvent",
    "ToolUseEvent",
]
