"""Codex thread items as agentshim events.

An item has a start and a completion, and the events follow them: a tool item
becomes a ``ToolCall`` when it starts and a ``ToolResult`` when it completes;
an agent message becomes ``AssistantText`` and a reasoning item ``Reasoning``
when they complete (the streamed deltas are not forwarded, so a message is
reported once, whole). A shell command that reads a ``SKILL.md`` also yields a
``SkillInvoked``, the only way a skill load shows (see ``..skills``).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from agentshim.core.events import (
    AssistantText,
    Reasoning,
    ToolCall,
    ToolResult,
)
from agentshim.providers.codex.parser import COMMAND_TOOL
from agentshim.providers.codex.skills import skill_reads

from .protocol import (
    ThreadItemAgentMessage,
    ThreadItemCommandExecution,
    ThreadItemDynamicToolCall,
    ThreadItemFileChange,
    ThreadItemMcpToolCall,
    ThreadItemReasoning,
    ThreadItemWebSearch,
)

if TYPE_CHECKING:
    from agentshim.core.events import AgentEvent

    from .protocol import ThreadItem

FILE_CHANGE_TOOL = "file_change"
MCP_TOOL = "mcp_tool_call"
DYNAMIC_TOOL = "dynamic_tool_call"
WEB_SEARCH_TOOL = "web_search"

#: Item statuses (command, file change, MCP call) that mean the tool did not do its work.
_FAILED_STATUSES = ("failed", "declined")
_MS = 1000.0


class ItemEvents:
    """Turns item notifications into events, pairing each completion with its start.

    Durations come from Codex's own ``durationMs`` when the item has one, and
    otherwise from the *now* (monotonic seconds) the caller supplies at start and
    completion, so nothing here reads a clock.
    """

    def __init__(self) -> None:
        """Start with no tool in flight."""
        self._started: dict[str, float] = {}

    def started(self, item: ThreadItem, now: float) -> list[AgentEvent]:
        """Events for an item that has just begun."""
        if isinstance(item, ThreadItemCommandExecution):
            self._started[item.id] = now
            events: list[AgentEvent] = [ToolCall(item.id, COMMAND_TOOL, {"command": item.command})]
            events += skill_reads(item.command, item.id)
            return events
        call = _tool_call(item)
        if call is None:
            return []
        self._started[call.tool_id or ""] = now
        return [call]

    def completed(self, item: ThreadItem, now: float) -> list[AgentEvent]:
        """Events for an item that has just finished."""
        if isinstance(item, ThreadItemAgentMessage):
            return [AssistantText(item.text)] if item.text.strip() else []
        if isinstance(item, ThreadItemReasoning):
            text = "\n".join(item.summary or item.content)
            return [Reasoning(text)] if text.strip() else []
        if isinstance(item, ThreadItemCommandExecution):
            failed = _failed(item.status) or bool(item.exit_code)
            return [
                self._result(
                    item.id,
                    COMMAND_TOOL,
                    item.aggregated_output or "",
                    item.exit_code,
                    item.duration_ms,
                    now,
                    failed=failed,
                )
            ]
        summary = _tool_summary(item)
        if summary is None:
            return []
        tool, tool_id, output, duration_ms, failed = summary
        return [self._result(tool_id, tool, output, None, duration_ms, now, failed=failed)]

    # One value per field of a ToolResult plus the clock reading; a bundle would only rename them.
    def _result(  # noqa: PLR0913
        self,
        tool_id: str,
        tool: str,
        output: str,
        exit_code: int | None,
        duration_ms: int | None,
        now: float,
        *,
        failed: bool,
    ) -> ToolResult:
        """Build a result, routing a failure's output onto ``stderr`` as every provider does."""
        began = self._started.pop(tool_id, None)
        if duration_ms is not None:
            duration_s: float | None = duration_ms / _MS
        else:
            duration_s = None if began is None else max(now - began, 0.0)
        if failed and exit_code is None:
            exit_code = 1
        return ToolResult(
            tool_id=tool_id,
            tool=tool,
            stdout="" if failed else output,
            stderr=output if failed else "",
            exit_code=exit_code,
            duration_s=duration_s,
        )


def _tool_call(item: ThreadItem) -> ToolCall | None:
    if isinstance(item, ThreadItemFileChange):
        changes = [{"path": change.path} for change in item.changes]
        return ToolCall(item.id, FILE_CHANGE_TOOL, {"changes": changes})
    if isinstance(item, ThreadItemMcpToolCall):
        args = {"server": item.server, "tool": item.tool, "arguments": item.arguments}
        return ToolCall(item.id, MCP_TOOL, args)
    if isinstance(item, ThreadItemDynamicToolCall):
        return ToolCall(item.id, DYNAMIC_TOOL, {"tool": item.tool, "arguments": item.arguments})
    if isinstance(item, ThreadItemWebSearch):
        return ToolCall(item.id, WEB_SEARCH_TOOL, {"query": item.query})
    return None


def _tool_summary(item: ThreadItem) -> tuple[str, str, str, int | None, bool] | None:
    """``(tool, id, output, duration_ms, failed)`` of a non-command tool item."""
    if isinstance(item, ThreadItemFileChange):
        output = "\n".join(change.path for change in item.changes)
        return FILE_CHANGE_TOOL, item.id, output, None, _failed(item.status)
    if isinstance(item, ThreadItemMcpToolCall):
        if item.error is not None:
            output = item.error.message
        elif item.result is not None:
            output = _json(item.result.content)
        else:
            output = ""
        return MCP_TOOL, item.id, output, item.duration_ms, _failed(item.status)
    if isinstance(item, ThreadItemDynamicToolCall):
        return DYNAMIC_TOOL, item.id, "", item.duration_ms, item.success is False
    if isinstance(item, ThreadItemWebSearch):
        return WEB_SEARCH_TOOL, item.id, item.query, None, False
    return None


def _json(value: object) -> str:
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


def _failed(status: str) -> bool:
    return status in _FAILED_STATUSES
