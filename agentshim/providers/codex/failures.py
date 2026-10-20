"""How Codex says why a turn failed, mapped to ``FailureKind``.

``codex exec --json`` reports a failure only as the text of an ``error`` or
``turn.failed`` event, so the classification reads that text. The phrases are
Codex's own: it names a usage limit ("You've hit your usage limit",
``usage_limit_reached``), retries a lost stream ("stream disconnected",
"Reconnecting...") and, once its retries run out, reports
"exceeded retry limit, last status: <status>".

A prompt the context window cannot hold arrives two ways. A request the
Responses API refuses is relayed as the API's own JSON error body, whose
``error.code`` is ``context_length_exceeded``; that code is read from the
parsed body, never searched for in prose. An overflow Codex's own accounting
catches is its message "Codex ran out of room in the model's context window".
"""

from __future__ import annotations

import json
import re
from typing import Any, cast

from agentshim.core.errors import FailureKind

#: The Responses API's ``error.code`` for a prompt longer than the window.
CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
#: Codex's own message for ``CodexErr::ContextWindowExceeded``.
_CONTEXT_WINDOW = re.compile(r"ran out of room in the model's context window", re.IGNORECASE)

_USAGE_LIMIT = re.compile(
    r"hit your usage limit|usage_limit_(?:reached|exceeded)|insufficient_quota", re.IGNORECASE
)
_AUTH = re.compile(r"\b(?:401|403)\b|unauthorized|forbidden|not logged in", re.IGNORECASE)
_TRANSIENT = re.compile(
    r"exceeded retry limit|stream disconnected|reconnecting|server_overloaded|overloaded"
    r"|internal_server_error|internal server error|service unavailable|bad gateway"
    r"|too many requests|rate limit|at capacity|http_connection_failed"
    r"|\bstatus:? (?:429|5\d\d)\b",
    re.IGNORECASE,
)


def classify_failure(text: str) -> FailureKind:
    """Classify the error text of one failed Codex turn.

    The order matters: a usage limit is also reported with a 429 status, and
    an exhausted retry can end on a 401, so the specific kinds are checked
    before the generic transient one. A JSON error body is classified by its
    code before any phrase is looked for.
    """
    code = api_error_code(text)
    if code == CONTEXT_LENGTH_EXCEEDED or _CONTEXT_WINDOW.search(text):
        return FailureKind.CONTEXT_OVERFLOW
    if _USAGE_LIMIT.search(text):
        return FailureKind.USAGE_LIMIT
    if _AUTH.search(text):
        return FailureKind.AUTH
    if _TRANSIENT.search(text):
        return FailureKind.TRANSIENT
    return FailureKind.OTHER


def api_error_code(text: str) -> str | None:
    """The ``error.code`` of *text* when it is an API error body, else ``None``."""
    try:
        body: object = json.loads(text)
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    error = cast("dict[str, Any]", body).get("error")
    if not isinstance(error, dict):
        return None
    code = cast("dict[str, Any]", error).get("code")
    return code if isinstance(code, str) else None
