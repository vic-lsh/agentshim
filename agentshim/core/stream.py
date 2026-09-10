"""Line-level helpers every provider stream parser needs."""

from __future__ import annotations

import json
import time
from typing import Any, cast


def parse_json_object(line: str) -> dict[str, Any] | None:
    """Decode one stdout line into a JSON object.

    Returns ``None`` for blank lines, invalid JSON, and valid JSON that is
    not an object (a bare ``42`` or ``[1,2]`` on stdout is provider noise,
    not an event, and must not crash the turn).
    """
    stripped = line.strip()
    if not stripped:
        return None
    try:
        data: object = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return cast("dict[str, Any]", data)


class ToolTracker:
    """Pairs a provider's tool-call frames with its tool-result frames.

    Result frames usually carry only the call id, so the tool name and the
    elapsed time have to be remembered from the call.
    """

    def __init__(self, *, unknown_name: str = "Tool") -> None:
        """Start a tracker with no calls recorded.

        ``unknown_name`` is what ``name()`` reports for a result frame whose
        call was never seen, so a caller always has a displayable label even
        when the provider's frames do not pair up.
        """
        self._names: dict[str, str] = {}
        self._starts: dict[str, float] = {}
        self._unknown = unknown_name

    def start(self, tool_id: str | None, tool: str) -> None:
        """Record a tool call so a later result can be attributed to it."""
        if tool_id is None:
            return
        self._names[tool_id] = tool
        self._starts[tool_id] = time.monotonic()

    def name(self, tool_id: str | None) -> str:
        """Return the tool name recorded for ``tool_id``."""
        if tool_id is None:
            return self._unknown
        return self._names.get(tool_id, self._unknown)

    def duration(self, tool_id: str | None) -> float | None:
        """Return seconds since the matching call, or ``None`` if unpaired."""
        if tool_id is None:
            return None
        start = self._starts.get(tool_id)
        if start is None:
            return None
        return time.monotonic() - start
