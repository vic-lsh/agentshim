"""Claude Code's session files: ``<state root>/projects/<project>/<session>.jsonl``.

One JSON object per line. An ``assistant`` line carries the model, the usage
of the request that wrote it, the session and the working directory. The CLI
writes one line per content block of a message, and each repeats the whole
usage, so a message is counted once, by ``message.id`` and ``requestId``.

A Task subagent writes its own file under ``<session>/subagents/``. Its
records belong to the parent session, which is named by the path.

Written from the CLI's session files and the public Claude Code documentation;
the layout can change with the CLI, and a format change is an agentshim release.
"""

from __future__ import annotations

import json
from collections import deque
from typing import TYPE_CHECKING, Any, cast

from agentshim.core.transcripts import TranscriptFormat, TranscriptRecord, parse_timestamp
from agentshim.core.usage import TokenUsage, UsageSpeed, normalized_usage

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from agentshim.core.transcripts import TranscriptParser

#: The CLI's model name for a message it wrote itself, which no API call produced.
SYNTHETIC_MODEL = "<synthetic>"
#: How many recent message keys a file's parser remembers. A repeat of one further back counts again.
KEY_WINDOW = 64
_SUBAGENTS = "subagents"


class ClaudeTranscriptParser:
    """Reads one Claude session file, line by line."""

    def __init__(self, path: Path, keys: tuple[str, ...] = (), malformed: int = 0) -> None:
        """Start from the state of an earlier read, or from nothing."""
        self._session = path.parent.parent.name if path.parent.name == _SUBAGENTS else None
        self._fallback_session = path.stem
        self._keys: deque[str] = deque(keys, maxlen=KEY_WINDOW)
        self._malformed = malformed

    @property
    def malformed(self) -> int:
        """Lines with usage that could not be used."""
        return self._malformed

    def reject(self) -> None:
        """Count a line the reader could not decode."""
        self._malformed += 1

    def snapshot(self) -> bytes:
        """The recent message keys and the malformed count, as JSON."""
        return json.dumps(
            {"keys": list(self._keys), "malformed": self._malformed}, separators=(",", ":")
        ).encode()

    def feed(self, line: str) -> tuple[TranscriptRecord, ...]:
        """The requests an ``assistant`` line records: its own, then each advisor call."""
        try:
            data: object = json.loads(line)
        except (ValueError, RecursionError):
            self._malformed += 1
            return ()
        request = _request(data)
        if request is None:
            return ()
        fields, message, usage, model = request
        stamp = parse_timestamp(fields.get("timestamp"))
        if stamp is None:
            self._malformed += 1
            return ()
        key = _key(message, fields)
        if key is not None:
            if key in self._keys:
                return ()
            self._keys.append(key)
        cwd = fields.get("cwd")
        first = TranscriptRecord(
            stamp,
            model,
            self._session or _text(fields.get("sessionId")) or self._fallback_session,
            cwd if isinstance(cwd, str) else None,
            _tokens(usage, turns=1),
            UsageSpeed.FAST if usage.get("speed") == "fast" else UsageSpeed.STANDARD,
        )
        return (first, *_advisors(usage, first))


def _request(
    data: object,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], str] | None:
    """The line, its message, usage and model, if it is an assistant line that billed a model call."""
    if not isinstance(data, dict):
        return None
    fields = cast("Mapping[str, Any]", data)
    message = fields.get("message")
    if fields.get("type") != "assistant" or not isinstance(message, dict):
        return None
    body = cast("Mapping[str, Any]", message)
    usage, model = body.get("usage"), body.get("model")
    if not isinstance(usage, dict) or not isinstance(model, str) or model in ("", SYNTHETIC_MODEL):
        return None
    return fields, body, cast("Mapping[str, Any]", usage), model


def _advisors(usage: Mapping[str, Any], executor: TranscriptRecord) -> list[TranscriptRecord]:
    """One record per advisor call, which runs on its own model and appears only in ``iterations``."""
    iterations = usage.get("iterations")
    found: list[TranscriptRecord] = []
    for item in cast("list[object]", iterations) if isinstance(iterations, list) else []:
        if not isinstance(item, dict):
            continue
        call = cast("Mapping[str, Any]", item)
        model = call.get("model")
        if call.get("type") != "advisor_message" or model in (None, "", SYNTHETIC_MODEL):
            continue
        if isinstance(model, str):
            found.append(
                TranscriptRecord(
                    executor.timestamp,
                    model,
                    executor.session_id,
                    executor.cwd,
                    _tokens(call, turns=0),
                    executor.speed,
                )
            )
    return found


def _key(message: Mapping[str, Any], line: Mapping[str, Any]) -> str | None:
    """What repeated content-block lines of one message share; ``None`` if the line names neither."""
    message_id, request_id = message.get("id"), line.get("requestId")
    if not isinstance(message_id, str) and not isinstance(request_id, str):
        return None
    return f"{_text(message_id)}:{_text(request_id)}"


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    try:
        number = int(value)
    except (OverflowError, ValueError):
        return 0
    return max(number, 0)


def _tokens(usage: Mapping[str, Any], *, turns: int) -> TokenUsage:
    """Claude reports uncached input apart from cache reads and writes; agentshim sums them."""
    read = _count(usage.get("cache_read_input_tokens"))
    write = _count(usage.get("cache_creation_input_tokens"))
    creation = usage.get("cache_creation")
    one_hour = (
        _count(cast("Mapping[str, Any]", creation).get("ephemeral_1h_input_tokens"))
        if isinstance(creation, dict)
        else 0
    )
    return normalized_usage(
        input_tokens=_count(usage.get("input_tokens")) + read + write,
        output_tokens=_count(usage.get("output_tokens")),
        cache_read_input_tokens=read,
        cache_write_input_tokens=write,
        cache_write_1h_input_tokens=one_hour,
        turns=turns,
    )


def _parser(path: Path, state: bytes | None) -> TranscriptParser | None:
    if state is None:
        return ClaudeTranscriptParser(path)
    try:
        saved = json.loads(state)
        keys, malformed = saved["keys"], saved["malformed"]
    except (ValueError, KeyError, TypeError):
        return None
    if (
        not isinstance(keys, list)
        or not all(isinstance(key, str) for key in cast("list[object]", keys))
        or isinstance(malformed, bool)
        or not isinstance(malformed, int)
        or malformed < 0
    ):
        return None
    return ClaudeTranscriptParser(path, tuple(cast("list[str]", keys)), malformed)


TRANSCRIPTS = TranscriptFormat(
    subdirs=("projects",),
    markers=(b'"usage"',),
    selected_fields={
        "type": True,
        "timestamp": True,
        "requestId": True,
        "sessionId": True,
        "cwd": True,
        "message": {"id": True, "model": True, "usage": True},
    },
    parser=_parser,
)
