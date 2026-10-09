"""Why a Codex turn failed, as a ``FailureKind``.

Codex classifies a failure in ``TurnError.codexErrorInfo``. That is the first
source; when it says only ``other`` (it then carries the raw HTTP body in
``message``) the text classifier the one-shot path uses reads the message.

- usage or session budget exceeded: ``USAGE_LIMIT``.
- rate limit, server overloaded, internal server error, flex unavailable, and
  a lost connection or response stream (unless it ended in a client error
  other than 408 and 429): ``TRANSIENT``.
- ``unauthorized``, or a connection failure with HTTP 401 or 403: ``AUTH``.
- anything else: the text classifier, which returns ``OTHER`` for text it does
  not recognize.
"""

from __future__ import annotations

from agentshim.core.errors import FailureKind
from agentshim.providers.codex.failures import classify_failure

from .protocol import (
    CodexErrorInfoHttpConnectionFailed,
    CodexErrorInfoKind,
    CodexErrorInfoResponseStreamConnectionFailed,
    CodexErrorInfoResponseStreamDisconnected,
    CodexErrorInfoResponseTooManyFailedAttempts,
    TurnError,
)

_BY_KIND: dict[CodexErrorInfoKind, FailureKind] = {
    CodexErrorInfoKind.USAGE_LIMIT_EXCEEDED: FailureKind.USAGE_LIMIT,
    CodexErrorInfoKind.SESSION_BUDGET_EXCEEDED: FailureKind.USAGE_LIMIT,
    CodexErrorInfoKind.RATE_LIMIT_EXCEEDED: FailureKind.TRANSIENT,
    CodexErrorInfoKind.SERVER_OVERLOADED: FailureKind.TRANSIENT,
    CodexErrorInfoKind.INTERNAL_SERVER_ERROR: FailureKind.TRANSIENT,
    CodexErrorInfoKind.FLEX_UNAVAILABLE: FailureKind.TRANSIENT,
    CodexErrorInfoKind.UNAUTHORIZED: FailureKind.AUTH,
}

_AUTH_STATUSES = frozenset({401, 403})
#: Client errors a retry can outlast: request timeout and rate limiting.
_RETRYABLE_CLIENT_STATUSES = frozenset({408, 429})
_CLIENT_ERRORS = range(400, 500)


def error_text(error: TurnError) -> str:
    """The message and any additional details of *error*, as one string."""
    if error.additional_details:
        return f"{error.message}\n{error.additional_details}"
    return error.message


def classify_turn_error(error: TurnError) -> FailureKind:
    """Decide the ``FailureKind`` of a failed turn from *error*."""
    info = error.codex_error_info
    if isinstance(info, CodexErrorInfoKind):
        kind = _BY_KIND.get(info)
        if kind is not None:
            return kind
    elif isinstance(
        info,
        (
            CodexErrorInfoHttpConnectionFailed,
            CodexErrorInfoResponseStreamConnectionFailed,
            CodexErrorInfoResponseStreamDisconnected,
            CodexErrorInfoResponseTooManyFailedAttempts,
        ),
    ):
        status = info.http_status_code
        if status in _AUTH_STATUSES:
            return FailureKind.AUTH
        if status is None or status not in _CLIENT_ERRORS or status in _RETRYABLE_CLIENT_STATUSES:
            return FailureKind.TRANSIENT
    return classify_failure(error_text(error))
