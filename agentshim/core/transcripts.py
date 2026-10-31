"""Types for reading a provider CLI's own session files, outside any agentshim run.

Each CLI keeps a transcript of its conversations on disk. The usage in those
files is the only record of work done without agentshim, such as a person
using the CLI directly. :func:`agentshim.transcripts.read_transcript_usage`
turns them into :class:`TranscriptRecord` values; what is in a file, and where,
is each provider's :class:`TranscriptFormat`.

agentshim keeps no state between calls. The caller remembers a
:class:`FilePosition` per file and hands it back, and gets only what is new.
The positions of every root of a provider go in one mapping: a session file
that moves between roots is then read on from where the old path stopped.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from .usage import TokenUsage, UsageSpeed

#: The bytes before a resume point that must still hash the same for the file to count as unchanged.
GUARD_LENGTH = 64
_UINT32_MAX = 0xFFFFFFFF

_TIMESTAMP = re.compile(
    r"(\d{4})-(\d\d)-(\d\d)[T ](\d\d):(\d\d):(\d\d)(?:\.(\d+))?(Z|[+-]\d\d(?::?\d\d)?)?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TranscriptRoot:
    """A directory a provider keeps transcripts under.

    Attributes:
        provider: The provider that wrote them.
        path: The directory, resolved from the environment (it may not exist).
    """

    provider: str
    path: Path


@dataclass(frozen=True)
class TranscriptRecord:
    """One model request's usage, or the part of it one line of a transcript added.

    Attributes:
        timestamp: When the request was recorded (timezone-aware). For the
            tokens a later line of the file added, when that line was written.
        model: The model id as the transcript names it.
        session_id: The session it belongs to; a subagent's records carry
            their parent's session.
        cwd: The working directory the session ran in: an absolute path on the
            user's machine, so a personal one. ``None`` unless the caller asked
            for it with ``include_cwd``, and then only when the file says.
        tokens: The request's tokens. ``input_tokens`` includes cache reads and
            writes, as everywhere in agentshim. A request is counted once, at
            the most any line of the file reported: the records of one request
            add up to its final usage, the first holding the request and its
            ``turns``, and later ones (``turns`` 0) only what a later line added.
        speed: The service speed it was billed at.
        reported_usd: A cost the file states. ``None`` for every shipped
            format: a CLI's own figure is not trusted (see :mod:`agentshim.providers.claude.cost`).
    """

    timestamp: dt.datetime
    model: str
    session_id: str
    cwd: str | None
    tokens: TokenUsage
    speed: UsageSpeed
    reported_usd: float | None = None


@dataclass(frozen=True)
class FilePosition:
    """Where a read of one file stopped, and what is needed to continue it.

    The caller stores this as it is and passes it back; only the fields
    outside ``state`` mean anything to it.

    Attributes:
        size: The file's size when the read began.
        mtime_ns: Its modification time then, in nanoseconds.
        offset: Bytes of complete lines already read.
        guard_length: How many bytes before ``offset`` the guard covers.
        guard_hash: FNV-1a (32-bit) of those bytes.
        state: The format's parser state at ``offset``, opaque to the caller.
        complete: Whether the read reached the end of the file. False when the
            byte budget stopped it; the next call continues from ``offset``.
    """

    size: int
    mtime_ns: int
    offset: int
    guard_length: int
    guard_hash: int
    state: bytes
    complete: bool = True

    def __post_init__(self) -> None:
        """Reject a position that could not have come from a real read."""
        for name in ("size", "mtime_ns", "offset", "guard_length", "guard_hash"):
            value: int = getattr(self, name)
            if isinstance(value, bool) or value < 0:
                msg = f"{name} must be a non-negative int, got {value!r}"
                raise ValueError(msg)
        if self.guard_length > min(GUARD_LENGTH, self.offset):
            msg = "guard_length must not exceed the guard size or the offset"
            raise ValueError(msg)
        if self.guard_hash > _UINT32_MAX:
            msg = "guard_hash must fit in 32 bits"
            raise ValueError(msg)


@dataclass(frozen=True)
class FileScan:
    """What one read found in one file.

    Attributes:
        path: The file.
        position: Where to continue from next time.
        records: Requests found after the known position (or in the whole file
            when ``reset``). Each is final: it will not be reported again.
        tail_records: Requests in an unterminated last line, which a writer may
            still be completing. Provisional: the next read reports them again
            in ``records`` once the line ends, so a caller shows them but does
            not keep them.
        reset: The file was rewritten, truncated or its guard changed, so the
            known position could not be used and the whole file was read again.
            The caller drops every record it holds for this path.
        malformed: How many lines the format could not use, over the whole file
            so far (it replaces the caller's count, it is not a delta). Counts
            lines in ``tail_records``' line too, if it is unusable.
    """

    path: Path
    position: FilePosition
    records: tuple[TranscriptRecord, ...]
    tail_records: tuple[TranscriptRecord, ...]
    reset: bool
    malformed: int


@dataclass(frozen=True)
class TranscriptScan:
    """The result of one bounded read of a root.

    Attributes:
        files: The files read this call, in path order.
        unchanged: Files whose size and modification time match the known
            position of a finished read, and files that are earlier copies of
            a session already read further under another path, so nothing was
            read.
        skipped: Files too old to matter for the window.
        pending: Files not finished: the byte budget ran out first. Call again
            with the new positions until it is zero.
        failed: Files and directories that could not be read, or that the
            format cannot read. Never treated as empty: the caller keeps what
            it knows and reports the source as incomplete.
        found: Whether the root directory exists.
    """

    files: tuple[FileScan, ...] = ()
    unchanged: int = 0
    skipped: int = 0
    pending: int = 0
    failed: tuple[Path, ...] = ()
    found: bool = True


class TranscriptParser(Protocol):
    """Reads one file's lines in order and remembers what later lines depend on."""

    @property
    def malformed(self) -> int:
        """How many lines so far were unusable."""
        ...

    def feed(self, line: str) -> tuple[TranscriptRecord, ...]:
        """The requests one line records. Unusable lines count in :attr:`malformed`."""
        ...

    def reject(self) -> None:
        """Count one line the reader could not decode, so the parser never saw it."""
        ...

    def snapshot(self) -> bytes:
        """The parser's state, for :meth:`TranscriptFormat.parser` to continue from."""
        ...


@dataclass(frozen=True)
class TranscriptFormat:
    """What a provider's transcript files look like.

    Attributes:
        subdirs: Directories under the provider's state root that hold
            transcripts.
        markers: ASCII byte strings. A line with none of them cannot carry
            anything the parser needs (usage, or state such as the model in
            force), so it is skipped without decoding it.
        selected_fields: The fields the parser reads, as nested mappings with
            ``True`` for a whole value. A line too large to decode whole is
            reduced to these before the parser sees it.
        parser: Builds a parser for one file. Given the state of a previous
            read it continues from there; it returns ``None`` when it cannot
            read that state, and the file is then read again from the start.
        suffix: Which file names are transcripts.
        unreadable_suffixes: Transcript names agentshim cannot read (a
            compressed archive). They are reported in ``failed``, not skipped.
        session_file: The name every copy of one session's file has, for a
            provider that moves or copies a file between its directories after
            it was read. A file with the name of one already read is read on
            from there, so the move adds nothing twice. ``None`` when a file
            keeps its path.
    """

    subdirs: tuple[str, ...]
    markers: tuple[bytes, ...]
    selected_fields: Mapping[str, object]
    parser: Callable[[Path, bytes | None], TranscriptParser | None]
    suffix: str = ".jsonl"
    unreadable_suffixes: tuple[str, ...] = field(default=())
    session_file: Callable[[Path], str] | None = None

    def __post_init__(self) -> None:
        """Reject a format that would read nothing, or read non-ASCII markers."""
        if not self.subdirs or not self.markers:
            msg = "a transcript format needs subdirs and markers"
            raise ValueError(msg)
        if any(not marker or not marker.isascii() for marker in self.markers):
            msg = "markers must be non-empty ASCII"
            raise ValueError(msg)


def parse_timestamp(value: object) -> dt.datetime | None:
    """A transcript's ISO 8601 timestamp as an aware datetime, or ``None`` if it is not one.

    Accepts ``Z`` and numeric offsets and any number of fractional digits
    (``datetime.fromisoformat`` before Python 3.11 takes neither). A timestamp
    with no offset is read as UTC.
    """
    if not isinstance(value, str):
        return None
    match = _TIMESTAMP.match(value.strip())
    if match is None:
        return None
    year, month, day, hour, minute, second, fraction, zone = match.groups()
    try:
        moment = dt.datetime(
            int(year),
            int(month),
            int(day),
            int(hour),
            int(minute),
            int(second),
            int((fraction or "0")[:6].ljust(6, "0")),
            tzinfo=_offset(zone),
        )
    except (ValueError, OverflowError):
        return None
    return moment


def _offset(zone: str | None) -> dt.timezone:
    if zone is None or zone.upper() == "Z":
        return dt.timezone.utc
    digits = zone[1:].replace(":", "")
    delta = dt.timedelta(hours=int(digits[:2]), minutes=int(digits[2:] or "0"))
    return dt.timezone(-delta if zone[0] == "-" else delta)
