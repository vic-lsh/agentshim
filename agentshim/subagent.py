"""Shared subagent primitive for isolated LLM calls.

Provides ``call_subagent()`` — a single fresh litellm call with its own
message list.  Used by RLM (isolated recursive calls) and the
SubagentCodingAgent (fan-out analysis calls).
"""

import os
import subprocess
import time
from typing import Any

import litellm
from loguru import logger

_NETWORK_ERROR_MARKERS = (
    "nameresolutionerror",
    "name or service not known",
    "transporterror",
    "apiconnectionerror",
    "connectionerror",
    "max retries exceeded",
)


def litellm_call_with_retry(
    kwargs: dict[str, Any],
    label: str,
    max_attempts: int = 3,
    token_acc: dict[str, int] | None = None,
) -> str:
    """Call litellm.completion with retry on transient network errors."""
    for attempt in range(max_attempts):
        try:
            response = litellm.completion(**kwargs)  # type: ignore[reportUnknownMemberType]
            if token_acc is not None:
                usage = getattr(response, "usage", None)
                if usage:
                    token_acc["prompt_tokens"] = token_acc.get("prompt_tokens", 0) + (
                        getattr(usage, "prompt_tokens", 0) or 0
                    )
                    token_acc["completion_tokens"] = token_acc.get("completion_tokens", 0) + (
                        getattr(usage, "completion_tokens", 0) or 0
                    )
                    token_acc["total_tokens"] = token_acc.get("total_tokens", 0) + (
                        getattr(usage, "total_tokens", 0) or 0
                    )
            return response.choices[0].message.content or ""  # type: ignore[reportAttributeAccessIssue]
        except Exception as e:
            if any(m in str(e).lower() for m in _NETWORK_ERROR_MARKERS) and attempt < max_attempts - 1:
                delay = 15 * (2**attempt)
                logger.warning(
                    f"[RLM] {label}: transient network error (attempt {attempt + 1}/{max_attempts}), "
                    f"retrying in {delay}s: {e}"
                )
                time.sleep(delay)
            else:
                raise

    return ""  # unreachable; satisfies type checker


def call_subagent(
    model: str,
    system_prompt: str,
    user_prompt: str,
    location: str | None = None,
    token_acc: dict[str, int] | None = None,
) -> str:
    """Make a completely fresh, isolated litellm call.

    Builds a new ``messages`` list from scratch (system + user), so there is
    no shared conversation history with any other call.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "cache": {"no-cache": True},
    }
    loc = location or os.environ.get("VERTEX_LOCATION")
    if loc:
        kwargs["vertex_location"] = loc

    try:
        return litellm_call_with_retry(kwargs, label="subagent call", token_acc=token_acc)
    except KeyboardInterrupt:
        raise
    except (TimeoutError, ConnectionError, subprocess.SubprocessError, OSError) as e:
        logger.error(f"[Subagent] LLM call failed: {type(e).__name__}: {e}")
        return f"Subagent call failed: {type(e).__name__}: {e}"
