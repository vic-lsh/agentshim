"""Codex's session files: ``<state root>/sessions/**/rollout-*.jsonl``, and ``archived_sessions``.

One JSON object per line. A ``token_count`` line carries the usage of one
model request, but not the model or the service tier: those come from earlier
lines (``turn_context`` for the model, ``thread_settings_applied`` for the
tier), so the parser carries them forward. The first ``session_meta`` names the
session and its working directory. Each ``token_count`` holds the request's
tokens (``last_token_usage``) and, usually, the thread's running totals
(``total_token_usage``), which are used to tell a new request from a repeat.

A forked or subagent session starts by copying its parent's history in a burst;
that history was billed in the parent, so it only sets the child's baseline.

Written from the CLI's session files and the public Codex documentation; the
layout can change with the CLI, and a format change is an agentshim release.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, cast

from agentshim.core.transcripts import TranscriptFormat, TranscriptRecord, parse_timestamp
from agentshim.core.usage import UsageSpeed, normalized_usage

from .speed import speed_of_tier

if TYPE_CHECKING:
    import datetime as dt
    from collections.abc import Mapping
    from pathlib import Path

    from agentshim.core.transcripts import TranscriptParser

Usage = dict[str, int]

#: Copied parent history arrives in a burst (0 to 40 ms apart); a child's own usage follows a model turn.
FORK_COPY_MAX_GAP_MS = 1000
_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)


@dataclass
class _State:
    """What later lines depend on; everything here is JSON."""

    model: str = ""
    speed: str = UsageSpeed.STANDARD.value
    session_id: str = ""
    cwd: str | None = None
    last_signature: str | None = None
    last_cumulative: Usage | None = None
    sequence_start: str | None = None
    malformed: int = 0
    saw_meta: bool = False
    suppressing: bool = False
    anchor_ms: int = 0


def _whole(value: object) -> int | None:
    """A non-negative whole number, as JSON may spell it (``5`` or ``5.0``); ``None`` otherwise."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    return int(value) if value >= 0 else None


def _usage(value: object) -> Usage | None:
    """A token usage object, or ``None`` if it is missing, ill-typed or self-contradictory."""
    if not isinstance(value, dict):
        return None
    raw = cast("Mapping[str, object]", value)
    if _whole(raw.get("input_tokens")) is None or _whole(raw.get("output_tokens")) is None:
        return None
    base = (cast("int", _whole(raw["input_tokens"])), cast("int", _whole(raw["output_tokens"])))
    defaults = {"total_tokens": base[0] + base[1]}
    parsed = {name: _whole(raw[name] if name in raw else defaults.get(name, 0)) for name in _FIELDS}
    if any(count is None for count in parsed.values()):
        return None
    usage = cast("Usage", parsed)
    if (
        usage["cached_input_tokens"] + usage["cache_write_input_tokens"] > usage["input_tokens"]
        or usage["reasoning_output_tokens"] > usage["output_tokens"]
        or usage["total_tokens"] < usage["input_tokens"] + usage["output_tokens"]
    ):
        return None
    return usage


def _advance(previous: Usage | None, last: Usage) -> Usage | None:
    return _usage({name: (previous[name] if previous else 0) + last[name] for name in _FIELDS})


def _signature(usage: Usage) -> str:
    return json.dumps(usage, sort_keys=True, separators=(",", ":"))


def _millis(stamp: dt.datetime) -> int:
    return round(stamp.timestamp() * 1000)


def _forked(payload: Mapping[str, Any]) -> bool:
    if isinstance(payload.get("forked_from_id"), str):
        return True
    node: object = payload.get("source")
    for name in ("subagent", "thread_spawn"):
        if not isinstance(node, dict):
            return False
        node = cast("Mapping[str, object]", node).get(name)
    return isinstance(node, dict) and isinstance(
        cast("Mapping[str, object]", node).get("parent_thread_id"), str
    )


class CodexTranscriptParser:
    """Reads one Codex rollout, line by line."""

    def __init__(self, path: Path, state: _State | None = None) -> None:
        """Start from the state of an earlier read, or from nothing."""
        self._fallback_session = path.stem
        self._state = state or _State()

    @property
    def malformed(self) -> int:
        """Usage records that were inconsistent or unreadable."""
        return self._state.malformed

    def reject(self) -> None:
        """Count a line the reader could not decode."""
        self._state.malformed += 1

    def snapshot(self) -> bytes:
        """The carried state, as JSON."""
        return json.dumps(asdict(self._state), separators=(",", ":")).encode()

    def feed(self, line: str) -> tuple[TranscriptRecord, ...]:
        """The request a ``token_count`` line records, if it is a new one."""
        try:
            data: object = json.loads(line)
        except (ValueError, RecursionError):
            self._state.malformed += 1
            return ()
        if not isinstance(data, dict):
            return ()
        fields = cast("Mapping[str, Any]", data)
        payload = fields.get("payload")
        if not isinstance(payload, dict):
            return ()
        body = cast("Mapping[str, Any]", payload)
        kind = fields.get("type")
        if kind == "session_meta":
            self._meta(fields, body)
        elif kind == "turn_context":
            self._context(body)
        elif body.get("type") == "thread_settings_applied":
            settings = body.get("thread_settings")
            if isinstance(settings, dict):
                tier = cast("Mapping[str, object]", settings).get("service_tier")
                self._state.speed = speed_of_tier(tier if isinstance(tier, str) else None).value
        elif body.get("type") == "token_count":
            record = self._count(fields, body)
            return () if record is None else (record,)
        return ()

    def _meta(self, line: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
        """The first ``session_meta`` names the session; copied ancestors' must not rename it."""
        state = self._state
        if state.saw_meta:
            return
        state.saw_meta = True
        name = payload.get("id") if payload.get("id") is not None else payload.get("session_id")
        if isinstance(name, str):
            state.session_id = name
        cwd = payload.get("cwd")
        if isinstance(cwd, str):
            state.cwd = cwd
        stamp = parse_timestamp(line.get("timestamp"))
        if stamp is not None and _forked(payload):
            state.suppressing = True
            state.anchor_ms = _millis(stamp)

    def _context(self, payload: Mapping[str, Any]) -> None:
        if isinstance(payload.get("model"), str):
            self._state.model = payload["model"]
        if isinstance(payload.get("cwd"), str):
            self._state.cwd = payload["cwd"]

    def _count(
        self, line: Mapping[str, Any], payload: Mapping[str, Any]
    ) -> TranscriptRecord | None:
        info = payload.get("info")
        stamp = parse_timestamp(line.get("timestamp"))
        if not isinstance(info, dict) or stamp is None or not self._state.model:
            return None
        details = cast("Mapping[str, object]", info)
        last = _usage(details.get("last_token_usage"))
        raw_total = details.get("total_token_usage")
        has_total = raw_total is not None
        total = _usage(raw_total) if has_total else None
        millis = _millis(stamp)
        if self._copying(millis, last, total, has_total=has_total):
            return None
        accepted = self._is_new_request(millis, last, total, has_total=has_total)
        return self._record(stamp, last) if accepted and last is not None else None

    def _rebase(self, total: Usage) -> None:
        self._state.last_cumulative = total
        if total["input_tokens"] + total["output_tokens"] == 0:
            self._state.sequence_start = None

    def _copying(
        self, millis: int, last: Usage | None, total: Usage | None, *, has_total: bool
    ) -> bool:
        """Whether this line is parent history copied into a fork, which sets a baseline only."""
        state = self._state
        if not state.suppressing:
            return False
        if millis - state.anchor_ms < FORK_COPY_MAX_GAP_MS:
            state.anchor_ms = millis
            signature = None if last is None else _signature(last)
            if total is not None:
                self._rebase(total)
            elif not has_total and last is not None and signature != state.last_signature:
                state.last_cumulative = _advance(state.last_cumulative, last)
            state.last_signature = signature
            return True
        state.suppressing = False
        state.last_signature = None
        return False

    def _malformed_line(self) -> bool:
        """Count an unusable usage record; always ``False``, so callers can ``return`` it."""
        self._state.malformed += 1
        return False

    def _is_new_request(
        self, millis: int, last: Usage | None, total: Usage | None, *, has_total: bool
    ) -> bool:
        """Whether a ``token_count`` is a request not seen before, updating the baseline as it goes."""
        previous = self._state.last_cumulative
        if total is not None:
            self._rebase(total)
        if last is None or (has_total and total is None):
            return self._malformed_line()
        if last["input_tokens"] + last["output_tokens"] == 0:
            # A context-size marker (compaction, fill_to_context_window), not a request.
            self._state.last_signature = None
            return False
        if total is not None and previous is not None and total == previous:
            return False
        if last["total_tokens"] != last["input_tokens"] + last["output_tokens"]:
            return self._malformed_line()
        signature = _signature(last)
        accepted = (
            self._reconciles(millis, last, total, previous)
            if total is not None
            else self._advances(last, signature, previous)
        )
        if accepted:
            self._state.last_signature = signature
        return accepted

    def _advances(self, last: Usage, signature: str, previous: Usage | None) -> bool:
        """For a line without running totals: a repeat of the last request is not a new one."""
        if signature == self._state.last_signature:
            return False
        advanced = _advance(previous, last)
        if advanced is None:
            return self._malformed_line()
        self._state.last_cumulative = advanced
        return True

    def _reconciles(self, millis: int, last: Usage, total: Usage, previous: Usage | None) -> bool:
        """Whether the running totals say *last* is the request that moved them."""
        state = self._state
        restarted = total == last
        start = json.dumps([millis, last], sort_keys=True, separators=(",", ":"))
        # A session resumed in a new process restarts its counters at this request; only the very
        # same event is a replay, since a later request can restart with identical counters.
        if restarted and start == state.sequence_start:
            state.last_cumulative = previous
            return False
        # Forks and paginated continuations inherit totals without copying a counter snapshot: bill
        # the latest request only, and require the unseen prefix to be valid usage too.
        if previous is None:
            reconciled = _usage({name: total[name] - last[name] for name in _FIELDS}) is not None
        else:
            reconciled = all(total[name] - previous[name] == last[name] for name in _FIELDS)
        if not restarted and not reconciled:
            return self._malformed_line()
        if restarted:
            state.sequence_start = start
        return True

    def _record(self, stamp: dt.datetime, last: Usage) -> TranscriptRecord:
        state = self._state
        tokens = normalized_usage(
            input_tokens=last["input_tokens"],
            output_tokens=last["output_tokens"],
            cache_read_input_tokens=last["cached_input_tokens"],
            cache_write_input_tokens=last["cache_write_input_tokens"],
            reasoning_output_tokens=last["reasoning_output_tokens"],
            turns=1,
        )
        return TranscriptRecord(
            stamp,
            state.model,
            state.session_id or self._fallback_session,
            state.cwd,
            tokens,
            UsageSpeed(state.speed),
        )


#: What each field of a saved state may hold, so a state from anywhere else is refused, not trusted.
_STATE_TYPES: dict[str, tuple[type, ...]] = {
    "model": (str,),
    "speed": (str,),
    "session_id": (str,),
    "cwd": (str, type(None)),
    "last_signature": (str, type(None)),
    "last_cumulative": (dict, type(None)),
    "sequence_start": (str, type(None)),
    "malformed": (int,),
    "saw_meta": (bool,),
    "suppressing": (bool,),
    "anchor_ms": (int,),
}


def _decode(state: bytes) -> _State | None:
    """The state a snapshot saved, or ``None`` if it is not one."""
    try:
        saved: object = json.loads(state)
    except ValueError:
        return None
    if not isinstance(saved, dict):
        return None
    fields = cast("dict[str, object]", saved)
    if set(fields) != set(_STATE_TYPES):
        return None
    for name, kinds in _STATE_TYPES.items():
        value = fields[name]
        if not isinstance(value, kinds) or (isinstance(value, bool) and bool not in kinds):
            return None
    loaded = _State(**cast("dict[str, Any]", fields))
    cumulative = loaded.last_cumulative
    if (
        loaded.speed not in {speed.value for speed in UsageSpeed}
        or loaded.malformed < 0
        or (cumulative is not None and _usage(cumulative) != cumulative)
    ):
        return None
    return loaded


def _parser(path: Path, state: bytes | None) -> TranscriptParser | None:
    if state is None:
        return CodexTranscriptParser(path)
    loaded = _decode(state)
    return None if loaded is None else CodexTranscriptParser(path, loaded)


TRANSCRIPTS = TranscriptFormat(
    subdirs=("sessions", "archived_sessions"),
    markers=(b'"token_count"', b'"turn_context"', b'"thread_settings_applied"', b'"session_meta"'),
    selected_fields={
        "type": True,
        "timestamp": True,
        "payload": {
            "type": True,
            "id": True,
            "session_id": True,
            "cwd": True,
            "model": True,
            "thread_settings": {"service_tier": True},
            "forked_from_id": True,
            "source": {"subagent": {"thread_spawn": {"parent_thread_id": True}}},
            "info": {"last_token_usage": True, "total_token_usage": True},
        },
    },
    parser=_parser,
    unreadable_suffixes=(".jsonl.zst",),
    shared_sessions=True,
)
