"""Session/agent usage accounting across all cli_agent providers.

Verifies that each provider's generation session populates ``self.usage``
from its native event stream, and that the normalized ``ProviderUsage``
obeys the crucible invariant (cached ⊆ input).
"""

from __future__ import annotations

from agentshim.claude import ClaudeGenerationSession
from agentshim.claude_events import ResultEvent as ClaudeResultEvent
from agentshim.codex import CodexGenerationSession
from agentshim.codex_events import TurnCompletedEvent
from agentshim.copilot import CopilotGenerationSession
from agentshim.copilot_events import TurnEndEvent as CopilotTurnEndEvent
from agentshim.copilot_events import UsageEvent as CopilotUsageEvent
from agentshim.gemini import GeminiGenerationSession
from agentshim.gemini_events import MessageEvent
from agentshim.opencode import OpencodeGenerationSession
from agentshim.opencode_events import StepFinishEvent
from agentshim.usage import ProviderUsage, TokenUsage


class _StubLogger:
    """Minimal stand-in for loguru's logger — avoids initializing loguru."""

    class _Opt:
        def info(self, *args, **kwargs) -> None: ...

    def __init__(self) -> None:
        self._opt = self._Opt()

    def opt(self, raw: bool = False) -> _StubLogger._Opt:
        return self._opt

    def info(self, *args, **kwargs) -> None: ...

    def warning(self, *args, **kwargs) -> None: ...

    def bind(self, **kwargs) -> _StubLogger:
        return self


def _make_session(cls):
    return cls(
        binary_name="stub",
        env={},
        log_prefix="[stub]",
        cmd=["stub"],
        logger=_StubLogger(),
        silent=True,
    )


class TestClaudeSessionUsage:
    def test_result_event_populates_usage_and_folds_cache_into_input(self):
        session = _make_session(ClaudeGenerationSession)
        event = ClaudeResultEvent(
            result="done",
            num_turns=4,
            usage={
                "input_tokens": 100,
                "output_tokens": 40,
                "cache_creation_input_tokens": 30,
                "cache_read_input_tokens": 20,
            },
            total_cost_usd=0.5,
        )
        session._handle_event(event)

        assert isinstance(session.usage, ProviderUsage)
        assert session.usage.provider == "claude"
        assert session.usage.total_cost_usd == 0.5
        # cached = 30 + 20 = 50; input_tokens = 100 + cached = 150
        assert session.usage.tokens.input_tokens == 150
        assert session.usage.tokens.output_tokens == 40
        assert session.usage.tokens.cached_input_tokens == 50
        assert session.usage.tokens.turns == 4
        # Invariant: cached ⊆ input
        assert session.usage.tokens.cached_input_tokens <= session.usage.tokens.input_tokens
        assert session.final_usage == {
            "input_tokens": 100,
            "output_tokens": 40,
            "cache_creation_input_tokens": 30,
            "cache_read_input_tokens": 20,
        }
        assert session.total_cost_usd == 0.5

    def test_result_event_populates_duration_ms(self):
        session = _make_session(ClaudeGenerationSession)
        event = ClaudeResultEvent(
            result="done",
            duration_ms=18431,
        )
        session._handle_event(event)
        assert session.duration_ms == 18431

    def test_result_event_without_usage_degrades_to_zero(self):
        session = _make_session(ClaudeGenerationSession)
        session._handle_event(ClaudeResultEvent(result="done"))
        assert session.usage.provider == "claude"
        assert session.usage.tokens == TokenUsage()
        assert session.final_usage is None
        assert session.total_cost_usd is None


class TestCodexSessionUsage:
    def test_turn_completed_events_accumulate(self):
        session = _make_session(CodexGenerationSession)
        session._handle_event(TurnCompletedEvent(input_tokens=100, cached_input_tokens=10, output_tokens=50))
        session._handle_event(TurnCompletedEvent(input_tokens=200, cached_input_tokens=30, output_tokens=80))

        assert session.usage.provider == "codex"
        assert session.usage.tokens.input_tokens == 300
        assert session.usage.tokens.output_tokens == 130
        assert session.usage.tokens.cached_input_tokens == 40
        assert session.usage.tokens.turns == 2

    def test_turn_completed_populates_final_usage(self):
        session = _make_session(CodexGenerationSession)
        session._handle_event(
            TurnCompletedEvent(
                input_tokens=100,
                cached_input_tokens=10,
                output_tokens=50,
                usage={"input_tokens": 100, "cached_input_tokens": 10, "output_tokens": 50},
            )
        )

        assert session.final_usage == {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 10,
            "cache_creation_input_tokens": 0,
        }
        assert session.total_cost_usd is None


class TestOpencodeSessionUsage:
    def test_step_finish_events_accumulate_with_cache_and_reasoning(self):
        session = _make_session(OpencodeGenerationSession)
        session._handle_event(
            StepFinishEvent(
                reason="tool-calls",
                cost=0.01,
                tokens={"input": 100, "output": 20, "reasoning": 5, "cache": {"read": 0, "write": 0}},
            )
        )
        session._handle_event(
            StepFinishEvent(
                reason="stop",
                cost=0.02,
                tokens={"input": 50, "output": 10, "reasoning": 3, "cache": {"read": 200, "write": 0}},
            )
        )

        assert session.usage.provider == "opencode"
        # input_tokens = (100 + 0) + (50 + 200) = 350; cached = 0 + 200 = 200
        assert session.usage.tokens.input_tokens == 350
        assert session.usage.tokens.cached_input_tokens == 200
        # output = 20+5 + 10+3 = 38 (reasoning folded into output)
        assert session.usage.tokens.output_tokens == 38
        assert session.usage.tokens.turns == 2
        # Cost accumulates across steps.
        assert session.usage.total_cost_usd == 0.03


class TestCopilotSessionUsage:
    def test_usage_events_accumulate_and_turn_end_updates_turns(self):
        session = _make_session(CopilotGenerationSession)
        session._handle_event(
            CopilotUsageEvent(
                model="gpt-5.4",
                input_tokens=100,
                output_tokens=20,
                cache_read_tokens=10,
                cache_write_tokens=5,
                reasoning_tokens=3,
            )
        )
        session._handle_event(CopilotTurnEndEvent(turn_id="1"))
        session._handle_event(
            CopilotUsageEvent(
                model="gpt-5.4",
                input_tokens=50,
                output_tokens=10,
                cache_read_tokens=0,
                cache_write_tokens=0,
                reasoning_tokens=2,
            )
        )
        session._handle_event(CopilotTurnEndEvent(turn_id="2"))

        assert session.usage.provider == "copilot"
        assert session.usage.tokens.input_tokens == 165
        assert session.usage.tokens.cached_input_tokens == 15
        assert session.usage.tokens.output_tokens == 35
        assert session.usage.tokens.turns == 2
        assert session.usage.total_cost_usd is None


class TestGeminiSessionUsage:
    def test_assistant_messages_count_as_turns_tokens_stay_zero(self):
        session = _make_session(GeminiGenerationSession)
        session._handle_event(MessageEvent(role="assistant", content="one"))
        session._handle_event(MessageEvent(role="assistant", content="two"))
        session._handle_event(MessageEvent(role="user", content="skip"))

        assert session.usage.provider == "gemini"
        assert session.usage.tokens.turns == 2
        assert session.usage.tokens.input_tokens == 0
        assert session.usage.tokens.output_tokens == 0
        assert session.usage.tokens.cached_input_tokens == 0
        assert session.usage.total_cost_usd is None


class TestTokenUsageInvariants:
    def test_to_dict_has_expected_keys(self):
        d = TokenUsage(input_tokens=1, output_tokens=2, cached_input_tokens=3, turns=4).to_dict()
        assert d == {"input_tokens": 1, "output_tokens": 2, "cached_input_tokens": 3, "turns": 4}

    def test_provider_usage_to_dict(self):
        pu = ProviderUsage(tokens=TokenUsage(turns=1), total_cost_usd=0.5, provider="claude")
        assert pu.to_dict() == {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "turns": 1,
            "total_cost_usd": 0.5,
            "provider": "claude",
        }

    def test_addition(self):
        a = TokenUsage(input_tokens=1, output_tokens=2, cached_input_tokens=1, turns=1)
        b = TokenUsage(input_tokens=3, output_tokens=4, cached_input_tokens=2, turns=1)
        c = a + b
        assert c == TokenUsage(input_tokens=4, output_tokens=6, cached_input_tokens=3, turns=2)
