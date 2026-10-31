"""Session files shaped like what the CLIs write, generated whole.

``builders.py`` makes one idealised line at a time: every repeat of a message carries the same
usage, a session file is never moved, and nothing is half written. Real files are not like that,
and a test of an idealised input only proves the idealisation. These generators make the shapes
real files have, from a survey of their structure (field names and counts only; nothing here is
copied from a real session):

* **Claude.** The lines of one message are in a row. Their prompt side is identical, but
  ``output_tokens`` is what the stream had reached when the line was written, so it grows from a
  few tokens on the first line to the whole reply on the last. Most messages write the finished
  count on every line. An advisor call is in the usage only once it has run.
* **Codex.** A session file is moved from ``sessions`` to ``archived_sessions`` under the same
  name, copies of it are left behind, and the session keeps growing while all that happens.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING, NamedTuple

from hypothesis import strategies as st

from .builders import (
    add_usage,
    claude_line,
    codex_context,
    codex_count,
    codex_meta,
    codex_usage,
    stamp,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from agentshim import TranscriptRecord

    from .builders import Tree

ADVISOR_MODEL = "claude-fable-5-1"
MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5")


class Totals(NamedTuple):
    """Tokens by class, which add up across records, and the requests they came from."""

    uncached_input: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cache_write_1h: int = 0
    output: int = 0
    turns: int = 0

    def __add__(self, other: object) -> Totals:  # type: ignore[override]
        assert isinstance(other, Totals)
        return Totals(*(a + b for a, b in zip(self, other, strict=True)))


def total_of(records: Sequence[TranscriptRecord]) -> dict[str, Totals]:
    """What a set of records add up to, per model."""
    found: dict[str, Totals] = {}
    for record in records:
        tokens = record.tokens
        found[record.model] = found.get(record.model, Totals()) + Totals(
            tokens.uncached_input_tokens,
            tokens.cache_read_input_tokens,
            tokens.cache_write_input_tokens,
            tokens.cache_write_1h_input_tokens,
            tokens.output_tokens,
            tokens.turns,
        )
    return found


@dataclass(frozen=True)
class Advisor:
    """An advisor call: on its own model, and in the usage from line ``shows_on`` on."""

    shows_on: int
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class Streamed:
    """One message as the CLI writes it: several lines whose output count grows."""

    key: int
    model: str
    prompt: tuple[int, int, int, int]  # uncached input, cache read, cache write, one-hour writes
    outputs: tuple[int, ...]  # the count on each line, never falling
    advisor: Advisor | None = None

    def lines(self, start: int) -> list[str]:
        """The lines of this message, with timestamps from index *start*."""
        uncached, read, write, one_hour = self.prompt
        written = []
        for index, output in enumerate(self.outputs):
            calls = []
            if self.advisor is not None and index >= self.advisor.shows_on:
                calls.append(
                    {
                        "type": "advisor_message",
                        "model": ADVISOR_MODEL,
                        "input_tokens": self.advisor.input_tokens,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "output_tokens": self.advisor.output_tokens,
                    }
                )
            last = index == len(self.outputs) - 1
            written.append(
                claude_line(
                    start + index,
                    model=self.model,
                    message=f"msg_{self.key}",
                    request=f"req_{self.key}",
                    input_tokens=uncached,
                    cache_read=read,
                    cache_write=write,
                    one_hour=one_hour,
                    output=output,
                    speed="standard" if last else None,
                    iterations=[{"type": "message", "output_tokens": output}, *calls]
                    if last or calls
                    else None,
                    extra={"apiBlockIndex": index, "parentUuid": f"uuid-{self.key}-{index}"},
                )
            )
        return written

    def final(self) -> dict[str, Totals]:
        """The usage of the finished message: the prompt once, the output it ended at."""
        uncached, read, write, one_hour = self.prompt
        found = {
            self.model: Totals(uncached, read, write, min(one_hour, write), max(self.outputs), 1)
        }
        if self.advisor is not None:
            found[ADVISOR_MODEL] = Totals(
                self.advisor.input_tokens, 0, 0, 0, self.advisor.output_tokens, 0
            )
        return found


_TOKENS = st.integers(min_value=0, max_value=200_000)


@st.composite
def streamed_messages(draw: st.DrawFn, key: int) -> Streamed:
    """A message with 1 to 6 lines, whose output grows on about half of them."""
    count = draw(st.integers(min_value=1, max_value=6))
    write = draw(_TOKENS)
    if draw(st.booleans()) and count > 1:
        first = draw(st.integers(min_value=1, max_value=20))
        rest = sorted(
            draw(st.lists(st.integers(first, 8000), min_size=count - 1, max_size=count - 1))
        )
        outputs = (first, *rest)
    else:
        outputs = (draw(st.integers(min_value=0, max_value=8000)),) * count
    advisor = None
    if draw(st.integers(min_value=0, max_value=5)) == 0:
        advisor = Advisor(
            draw(st.integers(min_value=0, max_value=count - 1)),
            draw(st.integers(min_value=0, max_value=5000)),
            draw(st.integers(min_value=1, max_value=5000)),
        )
    return Streamed(
        key,
        draw(st.sampled_from(MODELS)),
        (draw(_TOKENS), draw(_TOKENS), write, draw(st.integers(min_value=0, max_value=write))),
        outputs,
        advisor,
    )


@dataclass(frozen=True)
class ClaudeSession:
    """A session file's messages in order, with a user line between them."""

    messages: tuple[Streamed, ...]
    text: str = field(init=False)

    def __post_init__(self) -> None:
        """Lay the messages out as the file's text."""
        lines: list[str] = []
        for message in self.messages:
            lines.append(
                f'{{"type":"user","timestamp":"{stamp(len(lines))}","message":{{"content":"x"}}}}'
            )
            lines.extend(message.lines(len(lines)))
        object.__setattr__(self, "text", "".join(line + "\n" for line in lines))

    def final(self) -> dict[str, Totals]:
        """What the records of the whole file must add up to."""
        found: dict[str, Totals] = {}
        for message in self.messages:
            for model, totals in message.final().items():
                found[model] = found.get(model, Totals()) + totals
        return found


@st.composite
def claude_sessions(draw: st.DrawFn) -> ClaudeSession:
    """A session of 1 to 8 messages, which together stay inside an hour of timestamps."""
    count = draw(st.integers(min_value=1, max_value=8))
    return ClaudeSession(tuple(draw(streamed_messages(key)) for key in range(count)))


@st.composite
def cut_points(draw: st.DrawFn, size: int) -> list[int]:
    """Ascending byte offsets that cut a file of *size* bytes into the pieces a writer appends."""
    cuts = draw(st.lists(st.integers(min_value=0, max_value=size), max_size=6))
    return sorted(set(cuts))


def pieces(data: bytes, cuts: Sequence[int]) -> list[bytes]:
    """*data* cut at *cuts*, every piece non-empty, in order."""
    edges = [0, *cuts, len(data)]
    return [data[a:b] for a, b in pairwise(edges) if b > a]


# ---------------------------------------------------------------- Codex: a session that moves

LIVE = "sessions/2026/10/20/rollout-2026-10-20T00-00-00-abc.jsonl"
ARCHIVED = "archived_sessions/rollout-2026-10-20T00-00-00-abc.jsonl"


def rollout(requests: int) -> list[str]:
    """The lines of one Codex rollout with *requests* model requests, each a unique second."""
    lines = [codex_meta(0), codex_context(1, "gpt-6-luna")]
    total = codex_usage(0, out=0)
    for number in range(requests):
        last = codex_usage(100 + number, cached=10, out=5 + number)
        total = add_usage(total, last)
        lines.append(codex_count(2 + number, last, total))
    return lines


class Step(NamedTuple):
    """One thing that happens to a session's files, or a read of one root."""

    kind: str  # append, read, move, copy, drop_copy
    amount: int = 0  # lines to append, or the root to read


@st.composite
def file_moves(draw: st.DrawFn) -> list[Step]:
    """A history of appends, moves between the two directories, copies left behind and reads."""
    steps = []
    for _ in range(draw(st.integers(min_value=1, max_value=14))):
        kind = draw(
            st.sampled_from(["append", "append", "read", "read", "move", "copy", "drop_copy"])
        )
        amount = (
            draw(st.integers(min_value=1, max_value=4))
            if kind == "append"
            else draw(st.integers(min_value=0, max_value=1))
        )
        steps.append(Step(kind, amount))
    return steps


@dataclass
class Moving:
    """A session file on disk that is appended to, moved between roots, and copied."""

    state: Tree
    lines: list[str]
    written: int = 0
    where: str = LIVE
    copy: str | None = None

    def other(self, path: str) -> str:
        """The path in the other directory."""
        return ARCHIVED if path == LIVE else LIVE

    def append(self, count: int) -> None:
        """Write the next *count* lines of the rollout to the file."""
        take = self.lines[self.written : self.written + count]
        if not take:
            return
        text = "".join(line + "\n" for line in take)
        if self.written == 0:
            self.state.write(self.where, text)
        else:
            self.state.append(self.where, text)
        self.written += len(take)

    def move(self) -> None:
        """Rename the file into the other directory, as Codex archives and unarchives."""
        if self.written == 0:
            return
        target = self.other(self.where)
        (self.state.root / target).parent.mkdir(parents=True, exist_ok=True)
        (self.state.root / self.where).replace(self.state.root / target)
        self.where = target
        if self.copy == target:
            self.copy = None

    def copy_aside(self) -> None:
        """Leave a copy of the file as it is now in the other directory."""
        if self.written == 0:
            return
        target = self.other(self.where)
        path: Path = self.state.root / target
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.state.root / self.where, path)
        self.copy = target

    def drop_copy(self) -> None:
        """Delete the copy, if there is one."""
        if self.copy is not None:
            (self.state.root / self.copy).unlink()
            self.copy = None
