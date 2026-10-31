"""Read token usage out of a provider CLI's own session files.

This covers work done outside agentshim, such as a person using the CLI
directly. It is bounded and resumable, and keeps no state of its own:

* :func:`transcript_roots` says where a provider keeps its transcripts.
* :func:`read_transcript_usage` reads one root, given what the caller already
  knows about each file, and returns only what is new. It reads at most a byte
  budget per call, always stops at a line boundary, and says how many files
  are left, so a cold read of gigabytes finishes over several calls.

A file is read again from the start (``reset``) when it was rewritten or
truncated, which is detected by comparing a hash of the 64 bytes before the
resume point. A file that cannot be read is listed in ``failed`` and never
counts as empty.
"""

from __future__ import annotations

import json
import os
import re
from typing import TYPE_CHECKING, BinaryIO, cast

from agentshim.core.home import state_root
from agentshim.core.transcripts import (
    GUARD_LENGTH,
    FilePosition,
    FileScan,
    TranscriptFormat,
    TranscriptRecord,
    TranscriptRoot,
    TranscriptScan,
)

if TYPE_CHECKING:
    import datetime as dt
    from collections.abc import Iterator, Mapping
    from pathlib import Path

    from agentshim.core.transcripts import TranscriptParser
    from agentshim.providers.catalog import ProviderCatalog

__all__ = [
    "FilePosition",
    "FileScan",
    "TranscriptFormat",
    "TranscriptRecord",
    "TranscriptRoot",
    "TranscriptScan",
    "read_transcript_usage",
    "transcript_providers",
    "transcript_roots",
]

#: A file whose modification time is older than the window by more than this cannot hold a line in it.
_MTIME_SLACK_NS = 36 * 3600 * 1_000_000_000
_CHUNK_BYTES = 1 << 20
_MIN_CHUNK_BYTES = 1 << 16
#: A line larger than this is reduced to the fields the format reads before it is decoded.
_HUGE_LINE_BYTES = 8 << 20
_FNV_OFFSET = 0x811C9DC5
_FNV_PRIME = 0x01000193

_STRING = re.compile(rb'"[^"\\]*(?:\\.[^"\\]*)*"', re.DOTALL)
_TOKEN = re.compile(rb'"[^"\\]*(?:\\.[^"\\]*)*"|[\[\]{}]', re.DOTALL)
_SCALAR = re.compile(rb"[^,\]}\s]+")
_SPACE = re.compile(rb"\s*")


def transcript_providers(*, catalog: ProviderCatalog | None = None) -> tuple[str, ...]:
    """The providers in *catalog* whose transcripts can be read, in name order."""
    catalog = _catalog_or_default(catalog)
    return tuple(
        name for name in catalog.names() if _format(catalog, name, required=False) is not None
    )


def transcript_roots(
    provider: str, env: Mapping[str, str], *, catalog: ProviderCatalog | None = None
) -> tuple[TranscriptRoot, ...]:
    """The directories *provider* keeps transcripts under, for the environment *env*.

    Follows the CLI's own rule for its state root (an environment variable if
    set, else a directory under ``HOME``). The directories may not exist.
    A provider whose transcripts agentshim cannot read has no roots.

    Raises:
        ValueError: *provider* is not in the catalog.
    """
    catalog = _catalog_or_default(catalog)
    transcripts = _format(catalog, provider, required=False)
    if transcripts is None:
        return ()
    home = state_root(catalog.plugin(provider).describe(), env)
    if home is None:
        return ()
    return tuple(
        TranscriptRoot(provider, home / name, shared_sessions=transcripts.shared_sessions)
        for name in transcripts.subdirs
    )


def read_transcript_usage(
    root: TranscriptRoot,
    *,
    known: Mapping[Path, FilePosition],
    since: dt.datetime,
    byte_budget: int,
    catalog: ProviderCatalog | None = None,
) -> TranscriptScan:
    """Read the transcripts under *root* that may hold usage since *since*.

    Args:
        root: One of :func:`transcript_roots`.
        known: What the caller holds: the position returned for each file the
            last time it was read. A file with no entry is read from the start.
        since: Start of the window. Files last modified more than 36 hours
            before it are skipped.
        byte_budget: Stop after about this many bytes (at least 1), at a line
            boundary. The first file read always makes progress, so repeating
            the call with the returned positions finishes for any budget.
        catalog: Where to find the provider's format.

    Returns:
        What is new. Passing every returned ``position`` back as *known* and
        calling until ``pending`` is zero gives the same records as one read of
        each file whole, however the file grew or the budget split the work.

    Raises:
        ValueError: *byte_budget* is below 1, *since* is naive, or the root's
            provider has no transcript format.
    """
    if byte_budget < 1:
        msg = f"byte_budget must be at least 1, got {byte_budget}"
        raise ValueError(msg)
    if since.tzinfo is None:
        msg = "since must be timezone-aware"
        raise ValueError(msg)
    transcripts = _format(_catalog_or_default(catalog), root.provider, required=True)
    assert transcripts is not None  # noqa: S101 - required=True raises instead of returning None
    if not root.path.is_dir():
        return TranscriptScan(found=False)

    cutoff_ns = int(since.timestamp() * 1_000_000_000) - _MTIME_SLACK_NS
    failed: list[Path] = []
    listed = _list_files(root.path, transcripts, failed)
    walk = _Walk(transcripts, known, cutoff_ns, byte_budget, failed)
    for path in listed:
        walk.visit(path)
    return TranscriptScan(
        files=tuple(walk.scans),
        unchanged=walk.unchanged,
        skipped=walk.skipped,
        pending=walk.pending,
        failed=tuple(failed),
    )


class _Walk:
    """One call's pass over the listed files: what each was, and the budget left."""

    def __init__(
        self,
        fmt: TranscriptFormat,
        known: Mapping[Path, FilePosition],
        cutoff_ns: int,
        budget: int,
        failed: list[Path],
    ) -> None:
        self._fmt = fmt
        self._known = known
        self._cutoff_ns = cutoff_ns
        self._remaining = budget
        self._failed = failed
        self.scans: list[FileScan] = []
        self.unchanged = 0
        self.skipped = 0
        self.pending = 0

    def visit(self, path: Path) -> None:
        """Classify *path*, and read it if it is new and the budget allows."""
        try:
            status = path.stat()
        except OSError:
            self._failed.append(path)
            return
        if status.st_mtime_ns < self._cutoff_ns:
            self.skipped += 1
        elif path.name.endswith(self._fmt.unreadable_suffixes):
            self._failed.append(path)
        elif _unchanged(self._known.get(path), status):
            self.unchanged += 1
        elif self._remaining <= 0:
            self.pending += 1
        else:
            self._read(path, status)

    def _read(self, path: Path, status: os.stat_result) -> None:
        read = _read_file(path, status, self._known.get(path), self._fmt, self._remaining)
        if read is None:
            self._failed.append(path)
            return
        scan, used = read
        self._remaining -= used
        self.scans.append(scan)
        self.pending += not scan.position.complete


def _catalog_or_default(catalog: ProviderCatalog | None) -> ProviderCatalog:
    if catalog is not None:
        return catalog
    from agentshim.providers import DEFAULT_CATALOG  # noqa: PLC0415

    return DEFAULT_CATALOG


def _format(catalog: ProviderCatalog, provider: str, *, required: bool) -> TranscriptFormat | None:
    from agentshim.core.plugin import TranscriptSource  # noqa: PLC0415

    plugin = catalog.plugin(provider)
    found = plugin.transcripts() if isinstance(plugin, TranscriptSource) else None
    if found is None and required:
        msg = f"provider {provider!r} has no transcript format"
        raise ValueError(msg)
    return found


def _list_files(root: Path, transcripts: TranscriptFormat, failed: list[Path]) -> list[Path]:
    """Every transcript file under *root*, depth first, each directory in name order."""
    names = (transcripts.suffix, *transcripts.unreadable_suffixes)
    found: list[Path] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                children = sorted(entries, key=lambda entry: entry.name)
        except OSError:
            failed.append(directory)
            continue
        folders = [directory / entry.name for entry in children if _is_dir(entry)]
        found.extend(
            directory / entry.name
            for entry in children
            if entry.name.endswith(names) and not _is_dir(entry)
        )
        pending.extend(reversed(folders))
    return sorted(found)


def _is_dir(entry: os.DirEntry[str]) -> bool:
    try:
        return entry.is_dir(follow_symlinks=False)
    except OSError:
        return False


def _unchanged(previous: FilePosition | None, status: os.stat_result) -> bool:
    return (
        previous is not None
        and previous.complete
        and previous.size == status.st_size
        and previous.mtime_ns == status.st_mtime_ns
    )


def _fnv1a(data: bytes) -> int:
    value = _FNV_OFFSET
    for byte in data:
        value = ((value ^ byte) * _FNV_PRIME) & 0xFFFFFFFF
    return value


def _guard(handle: BinaryIO, offset: int) -> tuple[int, int]:
    """The length and hash of the bytes just before *offset*."""
    length = min(GUARD_LENGTH, offset)
    if length == 0:
        return 0, 0
    handle.seek(offset - length)
    window = handle.read(length)
    if len(window) != length:
        msg = "file shrank while it was being read"
        raise OSError(msg)
    return length, _fnv1a(window)


def _resume(
    handle: BinaryIO,
    path: Path,
    status: os.stat_result,
    previous: FilePosition | None,
    fmt: TranscriptFormat,
) -> tuple[int, TranscriptParser, bool]:
    """Where to start, with which parser, and whether the known position had to be dropped."""
    if previous is None or previous.offset == 0:
        parser = fmt.parser(path, None)
        assert parser is not None  # noqa: S101 - a parser always starts from nothing
        return 0, parser, False
    if previous.offset <= status.st_size and _guard(handle, previous.offset) == (
        previous.guard_length,
        previous.guard_hash,
    ):
        parser = fmt.parser(path, previous.state)
        if parser is not None:
            return previous.offset, parser, False
    fresh = fmt.parser(path, None)
    assert fresh is not None  # noqa: S101 - a parser always starts from nothing
    return 0, fresh, True


def _read_file(
    path: Path,
    status: os.stat_result,
    previous: FilePosition | None,
    fmt: TranscriptFormat,
    budget: int,
) -> tuple[FileScan, int] | None:
    """Read one file from the known position, or ``None`` if it cannot be read."""
    try:
        with path.open("rb") as handle:
            start, parser, reset = _resume(handle, path, status, previous, fmt)
            handle.seek(start)
            lines = _Lines(handle, start, budget)
            records: list[TranscriptRecord] = []
            gate = re.compile(b"|".join(re.escape(marker) for marker in fmt.markers))
            for line in lines:
                records.extend(_feed(parser, gate, fmt, line))
            snapshot = parser.snapshot()
            tail = fmt.parser(path, snapshot)
            assert tail is not None  # noqa: S101 - a parser reads its own snapshot
            tail_records = _feed(tail, gate, fmt, lines.tail) if lines.tail else ()
            length, digest = _guard(handle, lines.offset)
    except OSError:
        return None
    position = FilePosition(
        size=status.st_size,
        mtime_ns=status.st_mtime_ns,
        offset=lines.offset,
        guard_length=length,
        guard_hash=digest,
        state=snapshot,
        complete=lines.complete or lines.offset >= status.st_size,
    )
    scan = FileScan(
        path=path,
        position=position,
        records=tuple(records),
        tail_records=tuple(tail_records),
        reset=reset,
        malformed=tail.malformed,
    )
    return scan, lines.used


class _Lines:
    """Complete lines of a file from an offset, until the byte budget is spent.

    Iterating yields each line without its newline. ``offset`` then is the end
    of the last line yielded (counting its newline), ``tail`` the unterminated
    remainder if the file ended without a newline, and ``used`` the bytes the
    budget paid for. ``complete`` is whether the end of the file was reached.
    """

    def __init__(self, handle: BinaryIO, start: int, budget: int) -> None:
        self._handle = handle
        self._start = start
        self._budget = budget
        self.offset = start
        self.tail = b""
        self.complete = False

    @property
    def used(self) -> int:
        """Bytes consumed from the budget: whole lines read, plus any unterminated tail."""
        return self.offset - self._start + len(self.tail)

    def __iter__(self) -> Iterator[bytes]:
        partial = b""
        while True:
            chunk = self._handle.read(min(_CHUNK_BYTES, max(self._budget, _MIN_CHUNK_BYTES)))
            if not chunk:
                self.tail = partial
                self.complete = True
                return
            position = 0
            while (newline := chunk.find(b"\n", position)) != -1:
                line = partial + chunk[position:newline]
                partial = b""
                position = newline + 1
                self.offset += len(line) + 1
                yield line.removesuffix(b"\r")
                if self.offset - self._start >= self._budget:
                    return
            partial += chunk[position:]


def _feed(
    parser: TranscriptParser, gate: re.Pattern[bytes], fmt: TranscriptFormat, line: bytes
) -> tuple[TranscriptRecord, ...]:
    """The records one line holds, skipping it undecoded when no marker is in it."""
    if not gate.search(line):
        return ()
    if len(line) > _HUGE_LINE_BYTES:
        try:
            projected, _ = _project(line, 0, fmt.selected_fields)
        except ValueError:
            parser.reject()
            return ()
        return parser.feed(json.dumps(projected))
    return parser.feed(line.decode("utf-8", errors="replace"))


def _project(data: bytes, start: int, wanted: Mapping[str, object]) -> tuple[object, int]:
    """The object at *start* reduced to the fields in *wanted*, without decoding the rest.

    *wanted* maps a key to ``True`` (keep the value whole) or to a nested
    mapping (keep those fields of the object there). Everything else is
    skipped by scanning its extent only, so a multi-megabyte tool output
    costs one pass and no memory.

    Raises:
        ValueError: The bytes are not a JSON object.
    """
    position = _expect(data, start, b"{")
    kept: dict[str, object] = {}
    if data[position : position + 1] == b"}":
        return kept, position + 1
    while True:
        key_match = _STRING.match(data, position)
        if key_match is None:
            msg = "expected a key"
            raise ValueError(msg)
        key = json.loads(key_match.group())
        position = _expect(data, key_match.end(), b":")
        end = _value_end(data, position)
        if key in wanted:
            _keep(kept, key, wanted[key], data, position, end)
        position = _skip_space(data, end)
        separator = data[position : position + 1]
        if separator == b"}":
            return kept, position + 1
        position = _expect(data, position, b",")


def _keep(  # noqa: PLR0913 - the value's place in the bytes and the choice about it
    kept: dict[str, object], key: str, choice: object, data: bytes, start: int, end: int
) -> None:
    if choice is True:
        kept[key] = json.loads(data[start:end])
    elif data[start : start + 1] == b"{":
        kept[key], _ = _project(data, start, cast("Mapping[str, object]", choice))


def _expect(data: bytes, position: int, char: bytes) -> int:
    """The position after *char* and the whitespace following it, which must come after any space."""
    position = _skip_space(data, position)
    if data[position : position + 1] != char:
        msg = f"expected {char!r}"
        raise ValueError(msg)
    return _skip_space(data, position + 1)


def _skip_space(data: bytes, position: int) -> int:
    r"""Where the whitespace at *position* ends (``\s*`` always matches)."""
    match = _SPACE.match(data, position)
    return position if match is None else match.end()


def _value_end(data: bytes, start: int) -> int:
    """Where the JSON value that begins at *start* ends."""
    first = data[start : start + 1]
    if first == b'"':
        match = _STRING.match(data, start)
    elif first in (b"{", b"["):
        depth = 0
        for token in _TOKEN.finditer(data, start):
            opener = token.group() in (b"{", b"[")
            closer = token.group() in (b"}", b"]")
            depth += opener - closer
            if depth == 0 and (opener or closer):
                return token.end()
        msg = "unterminated value"
        raise ValueError(msg)
    else:
        match = _SCALAR.match(data, start)
    if match is None:
        msg = "expected a value"
        raise ValueError(msg)
    return match.end()
