"""opencode against the real binary.

Set ``AGENTSHIM_E2E_OPENCODE_MODEL`` to a ``provider/model`` when the model
in the user's own opencode config is not the one to test against.
"""

from __future__ import annotations

import pytest
from agentshim import CliAgent

from tests.e2e.conftest import OPENCODE_MODEL_VAR, model_from_env, requires_cli

pytestmark = [pytest.mark.e2e, requires_cli("opencode")]


def _agent() -> CliAgent:
    return CliAgent("opencode", model=model_from_env(OPENCODE_MODEL_VAR))


def test_a_single_turn_answers_and_reports_usage() -> None:
    result = _agent().run("Reply with exactly: pong")
    assert "pong" in result.text.lower()
    assert result.session_id is not None
    assert result.usage.tokens.input_tokens > 0
    assert result.usage.tokens.cached_input_tokens <= result.usage.tokens.input_tokens


def test_a_second_turn_resumes_the_first() -> None:
    session = _agent().start_session()
    session.turn("Remember the word 'juniper'. Reply with 'ok'.")
    assert session.session_id is not None
    second = session.turn("What word did I ask you to remember? Reply with just the word.")
    assert second.resumed is True
    assert "juniper" in second.text.lower()
