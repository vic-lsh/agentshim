"""How Claude Code says why a turn failed, mapped to ``FailureKind``.

Claude reports a failed API request three ways, newest first: an ``error``
kind on the synthetic assistant message it writes for the failure, an
``api_error_status`` on the final ``result`` frame, and the text
``API Error: <status> <body>`` in that frame's ``result``. Older builds only
have the text, so all three are read. A turn whose ``StructuredOutput`` calls
never validated ends with the ``result`` subtype
``error_max_structured_output_retries`` instead.

A prompt the context window cannot hold is named by the ``result`` frame's
``terminal_reason``. Builds without that field write ``Prompt is too long``
as the message, sometimes followed by why the CLI's own compaction failed
(which can be an API 503), so the overflow is checked before the status.
"""

from __future__ import annotations

import re

from agentshim.core.errors import FailureKind

#: Assistant ``error`` kinds that a later request may not hit.
_TRANSIENT_KINDS = frozenset({"overloaded", "rate_limit", "server_error"})
#: Assistant ``error`` kinds that need the user to sign in or fix an account.
_AUTH_KINDS = frozenset(
    {
        "authentication_failed",
        "oauth_org_not_allowed",
        "account_on_hold",
        "verification_required",
        "cloud_credential_error",
    }
)
_USAGE_LIMIT_KINDS = frozenset({"billing_error"})

#: Anthropic API error types that name an overload or a server fault.
_TRANSIENT_TYPES = re.compile(r"\b(?:overloaded_error|rate_limit_error|api_error)\b")
_AUTH_TYPES = re.compile(r"\b(?:authentication_error|permission_error)\b")
_API_ERROR_STATUS = re.compile(r"\bAPI Error: (\d{3})\b")
#: A 429 that is a subscription or spend quota rather than a rate limit.
#: Claude words these itself ("You've hit your limit", "usage limit reached")
#: or relays a gateway's own sentence about a spend limit or credit balance.
_QUOTA_TEXT = re.compile(r"hit your limit|usage limit|credit balance|spend limit", re.IGNORECASE)
_AUTH_STATUSES = frozenset({401, 403})
#: A request timeout or a rate limit; every 5xx (529 is an overload) is transient too.
_TRANSIENT_STATUSES = frozenset({408, 429})
_SERVER_ERRORS = range(500, 600)
#: The ``result`` subtype of a turn that used up its output-schema retries.
SCHEMA_RETRIES_SUBTYPE = "error_max_structured_output_retries"
#: ``terminal_reason`` values Claude Code groups as a context limit: the API
#: refused the prompt, the CLI's own estimate was over its blocking limit, or
#: the context refilled right after a compaction.
CONTEXT_LIMIT_REASONS = frozenset({"prompt_too_long", "blocking_limit", "rapid_refill_breaker"})
#: The CLI's message for the same, and the API's (``prompt is too long: N
#: tokens > M maximum``), for builds that write no ``terminal_reason`` or a
#: path that sets another one (a compaction that failed on its own error).
_CONTEXT_LIMIT_TEXT = re.compile(r"\bprompt is too long\b", re.IGNORECASE)


def classify_failure(
    *,
    subtype: str | None,
    api_error: str | None,
    status: int | None,
    text: str,
    terminal_reason: str | None = None,
) -> FailureKind:
    """Classify one failed turn from what Claude reported about it.

    ``subtype`` is the error ``result`` frame's subtype, ``api_error`` the
    assistant frame's ``error`` kind, ``status`` the
    result frame's ``api_error_status``, ``text`` the error text the
    stream (or, when the stream had none, stderr) carried, and
    ``terminal_reason`` why the CLI said it stopped. Tool output is never part
    of ``text``, so a tool that printed an API error cannot make the turn
    look like one.
    """
    if subtype == SCHEMA_RETRIES_SUBTYPE:
        return FailureKind.SCHEMA
    if terminal_reason in CONTEXT_LIMIT_REASONS or _CONTEXT_LIMIT_TEXT.search(text):
        return FailureKind.CONTEXT_OVERFLOW
    return _classify_api_failure(api_error=api_error, status=status, text=text)


def _classify_api_failure(*, api_error: str | None, status: int | None, text: str) -> FailureKind:
    """Classify a failure from the API error Claude reported, if it reported one."""
    if api_error in _AUTH_KINDS:
        return FailureKind.AUTH
    if api_error in _USAGE_LIMIT_KINDS:
        return FailureKind.USAGE_LIMIT
    if status is None:
        match = _API_ERROR_STATUS.search(text)
        status = int(match.group(1)) if match else None
    if (
        api_error in _TRANSIENT_KINDS
        or _is_transient_status(status)
        or _TRANSIENT_TYPES.search(text)
    ):
        return FailureKind.USAGE_LIMIT if _QUOTA_TEXT.search(text) else FailureKind.TRANSIENT
    if status in _AUTH_STATUSES or _AUTH_TYPES.search(text):
        return FailureKind.AUTH
    if _QUOTA_TEXT.search(text):
        return FailureKind.USAGE_LIMIT
    return FailureKind.OTHER


def _is_transient_status(status: int | None) -> bool:
    return status is not None and (status in _TRANSIENT_STATUSES or status in _SERVER_ERRORS)
