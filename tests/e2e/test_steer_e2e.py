"""Mid-turn steering against the real CLIs.

Each test starts a turn that runs a slow shell command, steers it from another
thread as soon as the command starts, and checks that the final answer reflects
the steer. Skipped unless ``AGENTSHIM_E2E=1`` and the binary is on PATH; use
``AGENTSHIM_E2E_CLAUDE_MODEL=haiku`` and ``AGENTSHIM_E2E_CODEX_MODEL=gpt-6-luna``.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import pytest
from agentshim import (
    Agent,
    ApprovalPolicy,
    NativePermissions,
    SteerConsumed,
    SteerDelivered,
    ToolCall,
    TransportKind,
    TurnRequest,
)
from agentshim.testing import RecordingEventHandler

from tests.e2e.conftest import CLAUDE_MODEL_VAR, CODEX_MODEL_VAR, model_from_env, requires_cli

if TYPE_CHECKING:
    from pathlib import Path

    from agentshim import AgentEvent, Session

pytestmark = pytest.mark.e2e

PROMPT = "Run the shell command `sleep 20 && echo done`, then tell me what it printed."
STEER = "Instead, reply with the word STEERED and stop."


def _steer_during_the_command(provider: str, model_var: str, cwd: Path) -> tuple[str, list[str]]:
    handler = RecordingEventHandler()
    agent = Agent(
        provider,
        transport=TransportKind.STREAM,
        model=model_from_env(model_var),
        permissions=NativePermissions.bypass(),
        approvals=ApprovalPolicy.DENY,
        event_handlers=[handler],
    )
    steers: list[threading.Thread] = []
    errors: list[BaseException] = []

    def steer(session: Session) -> None:
        try:
            session.steer(STEER)
        except BaseException as error:  # noqa: BLE001 - reported by the test thread
            errors.append(error)

    with agent.session(str(cwd)) as session:

        def on_event(event: AgentEvent) -> None:
            if isinstance(event, ToolCall) and not steers:
                thread = threading.Thread(target=steer, args=(session,))
                steers.append(thread)
                thread.start()

        ticket = session.prepare_turn(TurnRequest(prompt=PROMPT, timeout=300))
        turn = session.run(ticket, on_event=_Handler(on_event))
        for thread in steers:
            thread.join()
    assert errors == []
    kinds = [
        type(e).__name__ for e in handler.events if isinstance(e, (SteerDelivered, SteerConsumed))
    ]
    return turn.result.text, kinds


class _Handler:
    def __init__(self, fn: object) -> None:
        self._fn = fn

    def on_event(self, event: AgentEvent) -> None:
        self._fn(event)  # type: ignore[operator]


@requires_cli("claude")
def test_claude_a_steer_changes_the_final_answer(tmp_path: Path) -> None:
    text, kinds = _steer_during_the_command("claude", CLAUDE_MODEL_VAR, tmp_path)
    assert "STEERED" in text.upper()
    assert kinds == ["SteerDelivered", "SteerConsumed"]


@requires_cli("codex")
def test_codex_a_steer_changes_the_final_answer(tmp_path: Path) -> None:
    text, kinds = _steer_during_the_command("codex", CODEX_MODEL_VAR, tmp_path)
    assert "STEERED" in text.upper()
    assert kinds == ["SteerDelivered", "SteerConsumed"]
