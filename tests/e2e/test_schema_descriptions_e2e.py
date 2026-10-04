"""Normalized field descriptions reach models through native structured output."""

from __future__ import annotations

from pathlib import Path

import pytest
from agentshim import CliAgent, OutputSchema, TurnRequest, normalize

from tests.e2e.conftest import (
    CLAUDE_MODEL_VAR,
    CODEX_MODEL_VAR,
    model_from_env,
    requires_cli,
)

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize(
    ("provider", "model_variable", "cheap_model"),
    [
        pytest.param("claude", CLAUDE_MODEL_VAR, "haiku", marks=requires_cli("claude")),
        pytest.param("codex", CODEX_MODEL_VAR, "gpt-6-luna", marks=requires_cli("codex")),
    ],
)
def test_normalized_property_description_guides_the_model(
    tmp_path: Path, provider: str, model_variable: str, cheap_model: str
) -> None:
    """The schema supplies the answer, so dropping its description loses guidance."""
    agent = CliAgent(provider, model=model_from_env(model_variable) or cheap_model)
    dialect = agent.profile.schema_dialect
    assert dialect is not None
    schema = normalize(
        {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Write the word BANANA here regardless of the prompt.",
                }
            },
        },
        dialect,
    )
    result = agent.start_session(cwd=str(tmp_path)).turn(
        TurnRequest(
            prompt="Return a valid structured response. Follow each field description.",
            timeout=120,
            output_schema=OutputSchema(schema=schema, host_dir=tmp_path),
        )
    )
    assert result.structured_output == {"answer": "BANANA"}
