"""How Codex says why a turn failed, mapped to ``FailureKind``.

``codex exec --json`` reports a failure only as the text of an ``error`` or
``turn.failed`` event, so the classification reads that text. The phrases are
Codex's own: it names a usage limit ("You've hit your usage limit",
``usage_limit_reached``), retries a lost stream ("stream disconnected",
"Reconnecting...") and, once its retries run out, reports
"exceeded retry limit, last status: <status>".
"""

from __future__ import annotations

import re

from agentshim.core.errors import FailureKind

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
    before the generic transient one.
    """
    if _USAGE_LIMIT.search(text):
        return FailureKind.USAGE_LIMIT
    if _AUTH.search(text):
        return FailureKind.AUTH
    if _TRANSIENT.search(text):
        return FailureKind.TRANSIENT
    return FailureKind.OTHER
