"""Provider request accounting through the public session API."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from agentshim import CliAgent, CliTimeoutError, CommandRequest
from agentshim.testing import FakeExecutor, FakeRun
from hypothesis import given
from hypothesis import strategies as st


@given(st.lists(st.tuples(st.integers(1, 10000), st.integers(0, 10000)), min_size=1, max_size=10))
def test_rollout_request_usage_survives_resume_compaction_and_duplicate_snapshots(
    requests: list[tuple[int, int]],
) -> None:
    with TemporaryDirectory() as root:
        home = Path(root)
        directory = home / "sessions" / "2026" / "10" / "04"
        directory.mkdir(parents=True)
        path = directory / "rollout-example-session.jsonl"
        path.write_text(_event(90000, 90000, 1), encoding="utf-8")
        index = 0

        def run(_request: CommandRequest) -> FakeRun:
            nonlocal index
            total = 90000
            previous = _event(90000, 90000, 1) if index == 0 else _event(0, 0, 0)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(previous)
                for input_tokens, output_tokens in requests:
                    total += input_tokens
                    record = _event(input_tokens, total, output_tokens)
                    stream.write(record)
                    stream.write(record)
                    stream.write(_event(0, 0, 0))
                    total = 0
            index += 1
            return FakeRun(stdout=['{"type":"thread.started","thread_id":"session"}\n'])

        session = CliAgent(
            "codex", executor=FakeExecutor(run), env={"CODEX_HOME": root}
        ).start_session(session_id="session")
        for _ in range(2):
            result = session.turn("continue")
            assert result.usage.increment_known
            assert result.usage.tokens.input_tokens == sum(item[0] for item in requests)
            assert result.usage.tokens.output_tokens == sum(item[1] for item in requests)
            assert result.usage.tokens.turns == len(requests)
        assert index == 2


def test_rollout_usage_is_emitted_when_executor_times_out(tmp_path: Path) -> None:
    directory = tmp_path / "sessions"
    directory.mkdir()
    path = directory / "rollout-example-session.jsonl"

    def run(_request: CommandRequest) -> FakeRun:
        path.write_text(_event(100, 100, 20), encoding="utf-8")
        return FakeRun(timeout=True)

    session = CliAgent(
        "codex", executor=FakeExecutor(run), env={"CODEX_HOME": str(tmp_path)}
    ).start_session(session_id="session")
    with pytest.raises(CliTimeoutError) as failure:
        session.turn("continue")
    assert failure.value.partial is not None
    assert failure.value.partial.usage.increment_known
    assert failure.value.partial.usage.tokens.input_tokens == 100


def _event(input_tokens: int, total: int, output_tokens: int) -> str:
    return (
        json.dumps(
            {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "last_token_usage": {
                            "input_tokens": input_tokens,
                            "cached_input_tokens": input_tokens // 2,
                            "output_tokens": output_tokens,
                        },
                        "total_token_usage": {"input_tokens": total},
                    },
                },
            }
        )
        + "\n"
    )
