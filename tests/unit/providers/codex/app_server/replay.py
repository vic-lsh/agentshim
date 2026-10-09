"""A fake Codex server that replays what a recorded session's server said.

The transcripts in ``tests/fixtures/codex_app_server`` are the real CLI's own
words. ``ReplayPeer`` serves a stretch of one: each time the client writes, it
checks the message is the one the recording's client wrote next (method and
parameters, modulo the request id and the settings the transport chooses for
itself), then plays back everything the server said before the client spoke
again, with the client's request id put into the replies.

Two things are deliberately not held to the recording. The sandbox and approval
policy the transport sends differ from the research driver's (it asked for
``untrusted`` where the transport asks ``on-request``), so those fields are not
compared and the server's echo of them is rewritten to what the client asked,
as a server that honoured the request would say. And ``clientInfo`` names the
research driver.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from agentshim.execution.process import StdoutLine

if TYPE_CHECKING:
    from collections.abc import Sequence

    from agentshim.testing.process import PeerOutput

FIXTURES = Path(__file__).resolve().parents[5] / "tests" / "fixtures" / "codex_app_server"

#: Fields compared nowhere: what the transport chooses for itself.
SKIPPED = frozenset({"clientInfo", "capabilities", "sandbox", "approvalPolicy", "sandboxPolicy"})
#: Fields the transport may send that the research driver did not.
EXTRAS = frozenset(
    {
        "model",
        "cwd",
        "ephemeral",
        "excludeTurns",
        "required",
        "approvalPolicy",
        "sandboxPolicy",
        "sandbox",
    }
)

_SANDBOXES = {
    "danger-full-access": {"type": "dangerFullAccess"},
    "read-only": {"type": "readOnly", "networkAccess": False},
    "workspace-write": {
        "type": "workspaceWrite",
        "writableRoots": [],
        "networkAccess": False,
        "excludeSlashTmp": False,
        "excludeTmpdirEnvVar": False,
    },
}


@dataclass(frozen=True)
class Entry:
    """One recorded line: ``out`` is the client's, ``in`` the server's."""

    direction: str
    message: dict[str, object]


def load(name: str) -> list[Entry]:
    """The entries of fixture ``session_<name>.jsonl``."""
    path = FIXTURES / f"session_{name}.jsonl"
    entries: list[Entry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        entries.append(Entry(raw["dir"], raw["msg"]))
    return entries


def mismatch(
    recorded: object,
    sent: object,
    *,
    skip: frozenset[str],
    extras: frozenset[str],
    path: str = "params",
) -> str | None:
    """Describe the first way *sent* differs from *recorded*, or ``None`` if equivalent.

    Every key the recording has must be sent with an equivalent value, except
    those in *skip*; the client may send keys the recording lacks only if they
    are in *extras*.
    """
    if isinstance(recorded, dict) and isinstance(sent, dict):
        return _object_mismatch(recorded, sent, skip, extras, path)
    if isinstance(recorded, list) and isinstance(sent, list):
        return _list_mismatch(recorded, sent, skip, extras, path)
    return None if recorded == sent else f"{path} is {sent!r}, recorded {recorded!r}"


def _object_mismatch(
    recorded: dict[str, object],
    sent: dict[str, object],
    skip: frozenset[str],
    extras: frozenset[str],
    path: str,
) -> str | None:
    for key, value in recorded.items():
        if key in skip:
            continue
        if key not in sent:
            return f"{path}.{key} was not sent"
        problem = mismatch(value, sent[key], skip=skip, extras=extras, path=f"{path}.{key}")
        if problem:
            return problem
    unexpected = [key for key in sent if key not in recorded and key not in extras]
    return f"{path} has unexpected {unexpected}" if unexpected else None


def _list_mismatch(
    recorded: list[object],
    sent: list[object],
    skip: frozenset[str],
    extras: frozenset[str],
    path: str,
) -> str | None:
    if len(recorded) != len(sent):
        return f"{path} has {len(sent)} items, recorded {len(recorded)}"
    for index, (a, b) in enumerate(zip(recorded, sent, strict=True)):
        problem = mismatch(a, b, skip=skip, extras=extras, path=f"{path}[{index}]")
        if problem:
            return problem
    return None


class ReplayPeer:
    """Serves *entries*; raises ``AssertionError`` the moment the client departs from them."""

    def __init__(
        self,
        entries: Sequence[Entry],
        *,
        skip: frozenset[str] = frozenset(),
        extras: frozenset[str] = frozenset(),
        tolerated: frozenset[str] = frozenset(),
    ) -> None:
        """Replay *entries*; *skip* and *extras* widen what is compared.

        A client request whose method is in *tolerated* may also appear where
        the recording has none (the transport reacts to something the research
        driver ignored); the server says nothing back.
        """
        self._entries = deque(entries)
        self._skip = SKIPPED | skip
        self._extras = EXTRAS | extras
        self._tolerated = tolerated
        self._ids: dict[object, object] = {}
        self._buffer = ""
        self._asked: dict[str, object] = {}
        self.sent: list[dict[str, object]] = []

    @property
    def finished(self) -> bool:
        """Whether the client has said everything the recording's client did."""
        return not any(entry.direction == "out" for entry in self._entries)

    def on_start(self) -> list[PeerOutput]:
        """Play any server lines that come before the client's first."""
        return self._play()

    def on_stdin(self, data: str) -> list[PeerOutput]:
        """Check each line the client wrote against the recording and play the reply."""
        out: list[PeerOutput] = []
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self._expect(json.loads(line))
                out += self._play()
        return out

    def on_stdin_closed(self) -> list[PeerOutput]:
        """Nothing more is said once the client closes the pipe."""
        return []

    def _expect(self, sent: dict[str, object]) -> None:
        self.sent.append(sent)
        recorded_next = self._entries[0].message if self._entries else {}
        if sent.get("method") in self._tolerated and recorded_next.get("method") != sent.get(
            "method"
        ):
            return
        if not self._entries or self._entries[0].direction != "out":
            msg = f"the recording has no client message here, but the client sent {sent!r}"
            raise AssertionError(msg)
        recorded = self._entries.popleft().message
        if recorded.get("method") != sent.get("method"):
            msg = f"client sent {sent!r}, the recording's client sent {recorded!r}"
            raise AssertionError(msg)
        if "result" in recorded or "error" in recorded:  # a reply to a server request
            assert sent == recorded, f"client replied {sent!r}, recording {recorded!r}"
            return
        problem = mismatch(
            recorded.get("params"), sent.get("params"), skip=self._skip, extras=self._extras
        )
        assert problem is None, f"{sent.get('method')}: {problem}"
        if "id" in recorded:
            self._ids[recorded["id"]] = sent["id"]
            self._asked[str(sent["id"])] = sent.get("params")

    def _play(self) -> list[PeerOutput]:
        out: list[PeerOutput] = []
        while self._entries and self._entries[0].direction == "in":
            message = dict(self._entries.popleft().message)
            if "method" not in message and message.get("id") in self._ids:
                original = message["id"]
                message["id"] = self._ids[original]
                self._honour_request(message)
            out.append(StdoutLine(json.dumps(message) + "\n"))
        return out

    def _honour_request(self, reply: dict[str, object]) -> None:
        """Make a thread's echoed sandbox and approval policy say what the client asked."""
        result = reply.get("result")
        asked = self._asked.get(str(reply["id"]))
        if (
            not isinstance(result, dict)
            or not isinstance(asked, dict)
            or "approvalPolicy" not in result
        ):
            return
        if "approvalPolicy" in asked:
            result["approvalPolicy"] = asked["approvalPolicy"]
        if asked.get("sandbox") in _SANDBOXES:
            result["sandbox"] = _SANDBOXES[asked["sandbox"]]  # type: ignore[index]
