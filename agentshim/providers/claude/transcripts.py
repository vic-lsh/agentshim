"""Claude Code's session files: ``<state root>/projects/<project>/<session>.jsonl``.

One JSON object per line. An ``assistant`` line carries the model, the usage
of the request that wrote it, the session and the working directory. The CLI
writes one line per content block of a message, and each repeats the usage the
stream had reached when it was written: the prompt side is the same on every
line, but ``output_tokens`` grows as the reply streams, from a few tokens on the
first line to the whole reply on the last. The lines of one message are told
apart by ``message.id`` and ``requestId``, and the message is counted once at
the largest count any line reported: the first line yields the request, each
later line yields only the tokens it added, so the records of a message always
add up to its final usage, however the file was read in pieces.

A Task subagent writes its own file under ``<session>/subagents/``. Its
records belong to the parent session, which is named by the path.

Written from the CLI's session files and the public Claude Code documentation;
the layout can change with the CLI, and a format change is an agentshim release.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from agentshim.core.transcripts import TranscriptFormat, TranscriptRecord, parse_timestamp
from agentshim.core.usage import TokenUsage, UsageSpeed, normalized_usage

from .cost import report_speed
from .model_data import SYNTHETIC_MODEL

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from agentshim.core.transcripts import TranscriptParser

#: How many recent message keys a file's parser remembers. A repeat of one further back counts again.
KEY_WINDOW = 64
_SUBAGENTS = "subagents"
_ADVISOR_CALL = "advisor_message"
_COUNT_FIELDS = 5


class _Counts(NamedTuple):
    """One model call's tokens by class, as separate counts that add up.

    A request is counted at the largest count any of its lines reported for each class, and a
    record carries only what a line added to that, so the classes are kept apart rather than
    nested the way :class:`TokenUsage` nests them.
    """

    uncached_input: int
    cache_read: int
    cache_write: int
    cache_write_1h: int
    output: int

    @classmethod
    def of(cls, tokens: TokenUsage) -> _Counts:
        return cls(
            tokens.uncached_input_tokens,
            tokens.cache_read_input_tokens,
            tokens.cache_write_input_tokens,
            tokens.cache_write_1h_input_tokens,
            tokens.output_tokens,
        )

    @property
    def empty(self) -> bool:
        return not any(self)

    def largest(self, other: _Counts) -> _Counts:
        """Each class at the larger of the two counts: a call's usage only grows."""
        return _Counts(*(max(a, b) for a, b in zip(self, other, strict=True)))

    def beyond(self, counted: _Counts) -> _Counts:
        """What this has over *counted*, class by class; a class that did not grow adds nothing."""
        return _Counts(*(max(a - b, 0) for a, b in zip(self, counted, strict=True)))

    def tokens(self, *, turns: int) -> TokenUsage:
        # Counts that only grow keep one-hour writes within writes. If a line ever moved writes
        # to the one-hour class without adding any, normalized_usage clamps the difference.
        return normalized_usage(
            input_tokens=self.uncached_input + self.cache_read + self.cache_write,
            output_tokens=self.output,
            cache_read_input_tokens=self.cache_read,
            cache_write_input_tokens=self.cache_write,
            cache_write_1h_input_tokens=self.cache_write_1h,
            turns=turns,
        )


class _Call(NamedTuple):
    """A model call a line reports: the request's own, then each advisor call."""

    model: str
    counts: _Counts


class _Added(NamedTuple):
    """What a line adds to its message's usage for one call."""

    model: str
    counts: _Counts
    turns: int


class ClaudeTranscriptParser:
    """Reads one Claude session file, line by line."""

    def __init__(
        self,
        path: Path,
        seen: Mapping[str, Sequence[_Counts]] | None = None,
        malformed: int = 0,
    ) -> None:
        """Start from the state of an earlier read, or from nothing."""
        self._session = path.parent.parent.name if path.parent.name == _SUBAGENTS else None
        self._fallback_session = path.stem
        #: The calls of each recent message and what was counted for each, oldest message first.
        self._seen: OrderedDict[str, tuple[_Counts, ...]] = OrderedDict(
            (key, tuple(counts)) for key, counts in (seen or {}).items()
        )
        self._malformed = malformed

    @property
    def malformed(self) -> int:
        """Lines with usage that could not be used."""
        return self._malformed

    def reject(self) -> None:
        """Count a line the reader could not decode."""
        self._malformed += 1

    def snapshot(self) -> bytes:
        """What was counted for the recent messages, and the malformed count, as JSON."""
        seen = [[key, [list(counts) for counts in calls]] for key, calls in self._seen.items()]
        return json.dumps(
            {"seen": seen, "malformed": self._malformed}, separators=(",", ":")
        ).encode()

    def feed(self, line: str) -> tuple[TranscriptRecord, ...]:
        """The tokens an ``assistant`` line adds: its own request, then each advisor call.

        The first line of a message yields the whole of what it reports. A later line of
        the same message yields only what grew since, which is usually some output tokens
        and nothing else.
        """
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
        session = self._session or _text(fields.get("sessionId")) or self._fallback_session
        cwd = fields.get("cwd")
        speed = report_speed(usage) or UsageSpeed.STANDARD
        return tuple(
            TranscriptRecord(
                stamp,
                added.model,
                session,
                cwd if isinstance(cwd, str) else None,
                added.counts.tokens(turns=added.turns),
                speed,
            )
            for added in self._count(_key(message, fields), _calls(usage, model))
        )

    def _count(self, key: str | None, calls: Sequence[_Call]) -> list[_Added]:
        """What *calls* add to the message *key*, which is counted at the largest of its lines.

        A call the message had not yet shown is added whole, even when empty, since the call
        happened. One it had shown is added only if it grew. The request's own call is the
        first, and carries the turn: a message counts one turn however many lines it has.
        """
        earlier = None if key is None else self._seen.pop(key, None)
        largest = list(earlier or ())
        added: list[_Added] = []
        for index, call in enumerate(calls):
            if index >= len(largest):
                largest.append(call.counts)
                added.append(_Added(call.model, call.counts, int(index == 0)))
                continue
            gained = call.counts.beyond(largest[index])
            largest[index] = largest[index].largest(call.counts)
            if not gained.empty:
                added.append(_Added(call.model, gained, 0))
        if key is not None:
            self._seen[key] = tuple(largest)
            while len(self._seen) > KEY_WINDOW:
                self._seen.popitem(last=False)
        return added


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


def _calls(usage: Mapping[str, Any], model: str) -> list[_Call]:
    """The request's own call, then one per advisor call, which runs on its own model.

    An advisor call appears only in ``iterations``.
    """
    calls = [_Call(model, _Counts.of(_tokens(usage)))]
    iterations = usage.get("iterations")
    for item in cast("list[object]", iterations) if isinstance(iterations, list) else []:
        if not isinstance(item, dict):
            continue
        call = cast("Mapping[str, Any]", item)
        advisor = call.get("model")
        if (
            call.get("type") == _ADVISOR_CALL
            and isinstance(advisor, str)
            and advisor not in ("", SYNTHETIC_MODEL)
        ):
            calls.append(_Call(advisor, _Counts.of(_tokens(call))))
    return calls


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


def _tokens(usage: Mapping[str, Any]) -> TokenUsage:
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
    )


def _saved_calls(entry: object) -> tuple[str, tuple[_Counts, ...]] | None:
    """One message of a saved state: its key and the counts of each call; ``None`` if malformed."""
    if not isinstance(entry, list) or len(cast("list[object]", entry)) != 2:  # noqa: PLR2004
        return None
    key, calls = cast("list[object]", entry)
    if not isinstance(key, str) or not isinstance(calls, list):
        return None
    counts = [_saved_counts(item) for item in cast("list[object]", calls)]
    if not counts or any(item is None for item in counts):
        return None
    return key, tuple(cast("list[_Counts]", counts))


def _saved_counts(item: object) -> _Counts | None:
    if not isinstance(item, list):
        return None
    numbers = cast("list[object]", item)
    if len(numbers) != _COUNT_FIELDS or not all(
        isinstance(number, int) and not isinstance(number, bool) and number >= 0
        for number in numbers
    ):
        return None
    return _Counts(*cast("list[int]", numbers))


def _parser(path: Path, state: bytes | None) -> TranscriptParser | None:
    if state is None:
        return ClaudeTranscriptParser(path)
    try:
        saved = json.loads(state)
        entries, malformed = saved["seen"], saved["malformed"]
    except (ValueError, KeyError, TypeError):
        return None
    if not isinstance(entries, list) or isinstance(malformed, bool):
        return None
    if not isinstance(malformed, int) or malformed < 0:
        return None
    seen: dict[str, tuple[_Counts, ...]] = {}
    for entry in cast("list[object]", entries):
        message = _saved_calls(entry)
        if message is None:
            return None
        seen[message[0]] = message[1]
    return ClaudeTranscriptParser(path, seen, malformed)


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
