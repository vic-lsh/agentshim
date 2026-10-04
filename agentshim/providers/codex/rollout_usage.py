"""Recover invocation usage from newly appended Codex rollout request records."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from agentshim.core.stream import parse_json_object
from agentshim.core.usage import ProviderUsage, TokenUsage, normalized_usage

if TYPE_CHECKING:
    from collections.abc import Mapping


class RolloutUsage:
    """Snapshot file positions before a command and sum only its billed requests.

    The stream's terminal total can be absent on suspension or reset by
    compaction. Rollout ``last_token_usage`` records survive both cases. File
    positions exclude previous invocations, including externally adopted ones.
    """

    def __init__(self, env: Mapping[str, str], session_id: str | None) -> None:
        """Capture existing rollout positions in the command's provider home."""
        home = env.get("CODEX_HOME")
        if home is None and env.get("HOME") is not None:
            home = str(Path(env["HOME"]) / ".codex")
        self._directory = Path(home) / "sessions" if home is not None else None
        self._session_id = session_id
        self._offsets: dict[Path, tuple[int, dict[str, Any] | None]] = {}
        for path in self._files():
            snapshot = _snapshot(path)
            if snapshot is not None:
                self._offsets[path] = snapshot

    def _files(self) -> list[Path]:
        if self._directory is None:
            return []
        return list(self._directory.glob("**/rollout-*.jsonl"))

    def read(self, session_id: str | None) -> ProviderUsage | None:
        """Read appended records for this conversation, never a prior turn."""
        session_id = session_id or self._session_id
        if session_id is None:
            return None
        for path in self._files():
            if not path.name.endswith(f"-{session_id}.jsonl"):
                continue
            try:
                with path.open("rb") as stream:
                    offset, previous = self._offsets.get(path, (0, None))
                    if offset < 0 or stream.seek(0, 2) < offset:
                        return None
                    stream.seek(offset)
                    appended = stream.read().decode("utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            usage = _request_usage(appended, previous)
            if usage is not None:
                return usage
        return None


def _request_usage(text: str, previous: dict[str, Any] | None) -> ProviderUsage | None:
    tokens = TokenUsage()
    raw = None
    for line in text.splitlines():
        record = parse_json_object(line)
        if record is None or record.get("type") != "event_msg":
            continue
        payload = _object(record.get("payload"))
        if not isinstance(payload, dict) or payload.get("type") != "token_count":
            continue
        info = _object(payload.get("info"))
        if not isinstance(info, dict):
            continue
        last = _object(info.get("last_token_usage"))
        total = _object(info.get("total_token_usage"))
        if not isinstance(last, dict) or not isinstance(total, dict) or total == previous:
            continue
        previous = total
        if not any(last.values()):
            continue
        counts = {key: value for key, value in last.items() if isinstance(value, int)}
        tokens += normalized_usage(
            input_tokens=counts.get("input_tokens", 0),
            output_tokens=counts.get("output_tokens", 0),
            cache_read_input_tokens=counts.get("cached_input_tokens", 0),
            cache_write_input_tokens=counts.get("cache_write_input_tokens", 0),
            reasoning_output_tokens=counts.get("reasoning_output_tokens", 0),
            turns=1,
        )
        raw = total
    if raw is None:
        return None
    return ProviderUsage(tokens=tokens, provider="codex", raw=raw, increment_known=True)


def _snapshot(path: Path) -> tuple[int, dict[str, Any] | None] | None:
    try:
        with path.open("rb") as stream:
            size = stream.seek(0, 2)
            stream.seek(max(0, size - 65536))
            tail = stream.read()
    except OSError:
        return None
    if tail and not tail.endswith(b"\n"):
        # An incomplete prior record has no safe invocation boundary.
        return (-1, None)
    previous = None
    for line in tail.decode("utf-8", errors="replace").splitlines():
        record = parse_json_object(line)
        if record is None or record.get("type") != "event_msg":
            continue
        payload = _object(record.get("payload"))
        if payload is None or payload.get("type") != "token_count":
            continue
        info = _object(payload.get("info"))
        if info is not None:
            previous = _object(info.get("total_token_usage"))
    return size, previous


def _object(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return cast("dict[str, Any]", value)
