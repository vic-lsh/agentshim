"""End-to-end usage propagation for the claude provider.

Existing tests cover ``_process_stdout`` / ``_handle_event`` in isolation,
but nothing exercises the full ``start_session() -> generate() ->
agent.last_usage`` path that the SREGym cli_agent driver depends on. This
reproduces that path with a scripted executor so we can assert token usage
AND step count (``turns``) survive onto ``agent.last_usage``.
"""

from __future__ import annotations

from agentshim.claude import ClaudeCodeCodingAgent
from agentshim.executor import CommandRequest, CommandResult, CommandStreamSink


class _FakeHandle:
    def terminate(self) -> None: ...

    def kill(self) -> None: ...


class _ScriptedExecutor:
    """Streams a fixed list of stdout lines through the sink, then exits."""

    def __init__(self, stdout_lines: list[str], returncode: int = 0) -> None:
        self._lines = stdout_lines
        self._returncode = returncode

    def find_binary(self, binary_name: str, env: dict[str, str]) -> str:
        return f"/usr/local/bin/{binary_name}"

    def check_binary(self, binary_path: str, env: dict[str, str], *, timeout: int) -> None:
        return None

    def run(self, request: CommandRequest, sink: CommandStreamSink) -> CommandResult:
        sink.started(_FakeHandle())
        buf: list[str] = []
        for line in self._lines:
            sink.stdout(line)
            buf.append(line)
        return CommandResult(returncode=self._returncode, stdout="".join(buf), stderr="")


_ASSISTANT_LINE = '{"type":"assistant","message":{"content":[{"type":"text","text":"hi"}]}}\n'
_RESULT_LINE = (
    '{"type":"result","result":"done","num_turns":9,'
    '"usage":{"input_tokens":7000,"output_tokens":120,'
    '"cache_creation_input_tokens":30,"cache_read_input_tokens":20},'
    '"total_cost_usd":0.0456}\n'
)


def test_generate_populates_agent_last_usage_with_turns_and_tokens() -> None:
    agent = ClaudeCodeCodingAgent(
        model="m",
        executor=_ScriptedExecutor([_ASSISTANT_LINE, _RESULT_LINE]),
    )
    session = agent.start_session(cwd="/tmp", timeout=30)
    session.generate("do the thing")

    u = agent.last_usage
    assert u.provider == "claude"
    assert u.tokens.turns == 9  # step count
    assert u.tokens.output_tokens == 120
    assert u.tokens.input_tokens == 7000 + 50  # cache folded into input
    assert u.tokens.cached_input_tokens == 50
    assert u.total_cost_usd == 0.0456


def test_coding_agent_wrapper_exposes_backend_last_usage() -> None:
    """Regression: the portable ``CodingAgent`` wrapper (what the SREGym
    driver constructs via ``CodingAgent(provider="claude")``) must surface
    ``last_usage`` from its backend. Previously the wrapper had no
    ``last_usage`` attribute, so ``generate`` populated it on the hidden
    backend while callers read ``None`` off the wrapper.
    """
    from agentshim import CodingAgent

    agent = CodingAgent(
        provider="claude",
        model="m",
        executor=_ScriptedExecutor([_ASSISTANT_LINE, _RESULT_LINE]),
    )
    session = agent.start_session(cwd="/tmp", timeout=30)
    session.generate("do the thing")

    assert agent.last_usage is not None, "wrapper must expose backend last_usage, not None"
    assert agent.last_usage.tokens.turns == 9
    assert agent.last_usage.tokens.output_tokens == 120
    assert agent.last_usage.total_cost_usd == 0.0456


def test_generate_without_result_event_leaves_zero_usage_not_none() -> None:
    """If the CLI exits without emitting a result event (e.g. killed after
    submit), last_usage should degrade to a zero ProviderUsage — never None.
    """
    agent = ClaudeCodeCodingAgent(
        model="m",
        executor=_ScriptedExecutor([_ASSISTANT_LINE]),  # no result line
    )
    session = agent.start_session(cwd="/tmp", timeout=30)
    session.generate("do the thing")

    assert agent.last_usage is not None
    assert agent.last_usage.tokens.turns == 0
