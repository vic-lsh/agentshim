"""Synthetic session files for the transcript tests.

Every line here is made up by the tests from the documented shape of each CLI's
session file. None is copied from a real session.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentshim import (
    FilePosition,
    FileScan,
    TranscriptRecord,
    TranscriptRoot,
    read_transcript_usage,
    transcript_roots,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

#: Long before every file's mtime, so the date filter never interferes unless a test asks.
SINCE = dt.datetime(2026, 10, 1, tzinfo=dt.timezone.utc)
BIG_BUDGET = 1 << 40
_BASE_NS = int(dt.datetime(2026, 10, 20, tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000


def stamp(index: int) -> str:
    """An increasing timestamp, one second per index, up to an hour."""
    return f"2026-10-20T{index // 3600:02d}:{index // 60 % 60:02d}:{index % 60:02d}.250Z"


def claude_line(  # noqa: PLR0913 - one keyword per field of the line
    index: int,
    *,
    model: str = "claude-opus-5-5",
    message: str | None = None,
    request: str | None = None,
    session: str | None = "sess-1",
    cwd: str | None = "/work/project",
    input_tokens: int = 10,
    cache_read: int = 0,
    cache_write: int = 0,
    one_hour: int | None = None,
    output: int = 5,
    speed: str | None = None,
    iterations: list[dict[str, Any]] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """One ``assistant`` line, as Claude Code writes it for one content block."""
    usage: dict[str, Any] = {
        "input_tokens": input_tokens,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_write,
        "output_tokens": output,
    }
    if one_hour is not None:
        usage["cache_creation"] = {"ephemeral_1h_input_tokens": one_hour}
    if speed is not None:
        usage["speed"] = speed
    if iterations is not None:
        usage["iterations"] = iterations
    line: dict[str, Any] = {
        "type": "assistant",
        "timestamp": stamp(index),
        "message": {
            "id": message if message is not None else f"msg_{index}",
            "model": model,
            "usage": usage,
            "content": [{"type": "text", "text": "hello"}],
        },
    }
    if request is not None:
        line["requestId"] = request
    if session is not None:
        line["sessionId"] = session
    if cwd is not None:
        line["cwd"] = cwd
    line.update(extra or {})
    return json.dumps(line)


def codex_meta(
    index: int, *, session: str = "codex-sess", cwd: str = "/work/project", **payload: object
) -> str:
    """A ``session_meta`` line."""
    return json.dumps(
        {
            "type": "session_meta",
            "timestamp": stamp(index),
            "payload": {"id": session, "cwd": cwd, **payload},
        }
    )


def codex_context(index: int, model: str, **payload: object) -> str:
    """A ``turn_context`` line, which sets the model in force."""
    return json.dumps(
        {
            "type": "turn_context",
            "timestamp": stamp(index),
            "payload": {"model": model, "turn_id": f"turn-{index}", **payload},
        }
    )


def codex_tier(index: int, tier: str | None) -> str:
    """A ``thread_settings_applied`` line, which sets the service tier."""
    return json.dumps(
        {
            "type": "event_msg",
            "timestamp": stamp(index),
            "payload": {
                "type": "thread_settings_applied",
                "thread_settings": {"service_tier": tier},
            },
        }
    )


def codex_usage(
    inp: int = 100, cached: int = 0, write: int = 0, out: int = 10, reasoning: int = 0
) -> dict[str, int]:
    """A token usage object whose total is consistent."""
    return {
        "input_tokens": inp,
        "cached_input_tokens": cached,
        "cache_write_input_tokens": write,
        "output_tokens": out,
        "reasoning_output_tokens": reasoning,
        "total_tokens": inp + out,
    }


def codex_count(
    index: int, last: Mapping[str, int], total: Mapping[str, int] | None = None, *, ms: int = 0
) -> str:
    """A ``token_count`` line; *total* is the thread's running totals, when the CLI wrote them."""
    info: dict[str, Any] = {"last_token_usage": dict(last)}
    if total is not None:
        info["total_token_usage"] = dict(total)
    when = stamp(index).replace(".250Z", f".{ms:03d}Z")
    return json.dumps(
        {"type": "event_msg", "timestamp": when, "payload": {"type": "token_count", "info": info}}
    )


def add_usage(a: Mapping[str, int], b: Mapping[str, int]) -> dict[str, int]:
    """Two usage objects summed field by field."""
    return {name: a[name] + b[name] for name in a}


@dataclass
class Tree:
    """A temporary state root with a clock for file modification times."""

    root: Path
    clock: int = 0
    written: dict[Path, bytes] = field(default_factory=dict)

    def write(self, relative: str, data: bytes | str) -> Path:
        """Create or replace a file, giving it a modification time later than any before."""
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = data.encode() if isinstance(data, str) else data
        path.write_bytes(payload)
        self.touch(path)
        self.written[path] = payload
        return path

    def append(self, relative: str, data: bytes | str) -> Path:
        """Add to a file, as a writer does, moving its modification time forward."""
        path = self.root / relative
        payload = data.encode() if isinstance(data, str) else data
        with path.open("ab") as handle:
            handle.write(payload)
        self.touch(path)
        self.written[path] = self.written.get(path, b"") + payload
        return path

    def touch(self, path: Path, *, age_hours: int = 0) -> None:
        """Set a strictly increasing modification time (or one *age_hours* older than the base)."""
        self.clock += 1
        when = _BASE_NS + self.clock * 1_000_000 - age_hours * 3600 * 1_000_000_000
        os.utime(path, ns=(when, when))


@dataclass
class Collected:
    """What repeated reads of a tree add up to, kept the way a caller would keep it."""

    records: dict[Path, list[TranscriptRecord]] = field(default_factory=dict)
    positions: dict[Path, FilePosition] = field(default_factory=dict)
    malformed: dict[Path, int] = field(default_factory=dict)
    tails: dict[Path, tuple[TranscriptRecord, ...]] = field(default_factory=dict)
    resets: list[Path] = field(default_factory=list)
    calls: int = 0
    failed: set[Path] = field(default_factory=set)

    def take(self, files: tuple[FileScan, ...]) -> None:
        """Apply one scan's files: drop what a reset invalidates, keep what is new."""
        for scan in files:
            if scan.reset:
                self.records[scan.path] = []
                self.resets.append(scan.path)
            self.records.setdefault(scan.path, []).extend(scan.records)
            self.positions[scan.path] = scan.position
            self.malformed[scan.path] = scan.malformed
            self.tails[scan.path] = scan.tail_records


def collect(
    root: TranscriptRoot, collected: Collected | None = None, *, budget: int = BIG_BUDGET
) -> Collected:
    """Read until nothing is pending, accumulating as a caller would."""
    state = collected or Collected()
    limit = 10_000
    while True:
        scan = read_transcript_usage(root, known=state.positions, since=SINCE, byte_budget=budget)
        state.calls += 1
        state.take(scan.files)
        state.failed |= set(scan.failed)
        if scan.pending == 0:
            return state
        limit -= 1
        assert limit > 0, "reading did not terminate"


@contextmanager
def tree() -> Iterator[Tree]:
    """A fresh temporary state root."""
    with tempfile.TemporaryDirectory() as name:
        yield Tree(Path(name).resolve())


def claude_root(state: Tree) -> TranscriptRoot:
    """The Claude transcript root of a tree."""
    (root,) = transcript_roots("claude", {"CLAUDE_CONFIG_DIR": str(state.root)})
    return root


def codex_root(state: Tree, which: int = 0) -> TranscriptRoot:
    """A Codex transcript root of a tree (``sessions`` is 0, ``archived_sessions`` is 1)."""
    return transcript_roots("codex", {"CODEX_HOME": str(state.root)})[which]
