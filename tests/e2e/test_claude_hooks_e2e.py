"""``ClaudeProvider(hooks=...)`` against the real Claude Code CLI.

Each test checks the hook *fired*, through a record the hook script writes,
not through what the model says. Each refusal is paired with a positive
control, so a model that declined to run the command cannot pass as a hook
that blocked it.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from agentshim import ClaudeHook, ClaudeProvider, CliAgent, SandboxConfig, interactive_env

from tests.e2e.conftest import CLAUDE_MODEL_VAR, model_from_env, requires_cli

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = [pytest.mark.e2e, requires_cli("claude")]

#: Records every call it sees; denies Bash commands containing the word given.
_HOOK = """
import json, sys
record, deny_word = sys.argv[1], sys.argv[2]
payload = json.load(sys.stdin)
with open(record, "a") as handle:
    handle.write(json.dumps(payload) + "\\n")
command = payload.get("tool_input", {}).get("command", "")
if deny_word and deny_word in command:
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": "blocked by the agentshim e2e hook",
    }}))
"""


@pytest.fixture
def workspace() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as base:
        yield Path(base)


def _hook(workspace: Path, *, deny_word: str) -> tuple[ClaudeHook, Path]:
    script, record = workspace / "hook.py", workspace / "hook-record.jsonl"
    script.write_text(_HOOK)
    hook = ClaudeHook(
        event="PreToolUse",
        matcher="Bash",
        command=[sys.executable, str(script), str(record), deny_word],
        timeout_s=30,
    )
    return hook, record


def _records(record: Path) -> list[dict[str, object]]:
    if not record.exists():
        return []
    return [json.loads(line) for line in record.read_text().splitlines()]


def _touch(path: Path) -> str:
    return (
        f"Use the Bash tool to run exactly this command and nothing else: touch '{path}'\n"
        "Do not retry or work around a failure. Then reply with DONE."
    )


def _agent(provider: ClaudeProvider) -> CliAgent:
    return CliAgent(
        provider,
        model=model_from_env(CLAUDE_MODEL_VAR),
        env={**interactive_env(), **provider.sandbox_env},
    )


def test_a_pre_tool_use_hook_blocks_the_command_the_control_runs(workspace: Path) -> None:
    control = workspace / "control.txt"
    _agent(ClaudeProvider()).run(_touch(control), cwd=str(workspace), timeout=300)
    assert control.exists(), "positive control: the model did not run the command"

    hook, record = _hook(workspace, deny_word="blocked")
    blocked = workspace / "blocked.txt"
    _agent(ClaudeProvider(hooks=[hook])).run(_touch(blocked), cwd=str(workspace), timeout=300)

    assert not blocked.exists()
    calls = _records(record)
    assert calls, "the hook never fired"
    assert all(call["hook_event_name"] == "PreToolUse" for call in calls)
    assert any("blocked.txt" in json.dumps(call["tool_input"]) for call in calls)


def test_user_hooks_and_the_sandbox_confinement_hook_both_run(workspace: Path) -> None:
    """Merging must keep both: the audit hook fires and the sandboxed write lands."""
    hook, record = _hook(workspace, deny_word="")
    provider = ClaudeProvider(
        sandbox=SandboxConfig(confine_native_reads_to=[str(workspace)]), hooks=[hook]
    )
    allowed = workspace / "allowed.txt"
    _agent(provider).run(_touch(allowed), cwd=str(workspace), timeout=300)

    assert allowed.exists(), "positive control: the model did not run the command"
    assert any("allowed.txt" in json.dumps(call["tool_input"]) for call in _records(record))


def test_a_resumed_turn_keeps_its_hooks(workspace: Path) -> None:
    hook, record = _hook(workspace, deny_word="second")
    session = _agent(ClaudeProvider(hooks=[hook])).start_session(cwd=str(workspace))
    session.turn(_touch(workspace / "first.txt"))
    assert (workspace / "first.txt").exists(), "positive control failed"

    second = session.turn(_touch(workspace / "second.txt"))
    assert second.resumed is True
    assert not (workspace / "second.txt").exists()
    assert any("second.txt" in json.dumps(call["tool_input"]) for call in _records(record))
