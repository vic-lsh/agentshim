from pathlib import Path
from unittest.mock import MagicMock, patch

from agentshim.copilot import CopilotCodingAgent

FIXTURE_DIR = Path("tests/fixtures/copilot")


def _read_fixture_lines(name: str) -> list[str]:
    return FIXTURE_DIR.joinpath(name).read_text().splitlines(keepends=True)


def _make_process(lines: list[str]) -> MagicMock:
    process = MagicMock()
    process.pid = 12345
    process.returncode = 0
    process.poll.return_value = 0
    process.wait.return_value = 0
    process.stdin = MagicMock()
    process.stdout = MagicMock()
    process.stdout.readline.side_effect = lines + [""]
    process.stderr = MagicMock()
    process.stderr.readline.side_effect = lambda: ""
    return process


def _build_agent(recorder=None) -> CopilotCodingAgent:
    with patch("shutil.which", return_value="/usr/bin/copilot"):
        with patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")):
            return CopilotCodingAgent(recorder=recorder)


class TestCopilotFixtureReplay:
    def test_generate_replays_real_first_turn_fixture(self):
        process = _make_process(_read_fixture_lines("session_turn_1.jsonl"))

        with patch("subprocess.Popen", return_value=process):
            agent = _build_agent()
            result = agent.generate("ignored", silent=True)

        assert result == "STORED"
        assert agent.last_usage.provider == "copilot"
        assert agent.last_usage.tokens.output_tokens == 5
        assert agent.last_usage.tokens.turns == 1

    def test_session_replays_real_resumed_fixture_and_uses_resume_flag(self):
        process1 = _make_process(_read_fixture_lines("session_turn_1.jsonl"))
        process2 = _make_process(_read_fixture_lines("session_turn_2_resumed.jsonl"))
        captured_cmds: list[list[str]] = []

        def popen_side_effect(cmd, *args, **kwargs):
            captured_cmds.append(cmd)
            return [process1, process2][len(captured_cmds) - 1]

        with patch("subprocess.Popen", side_effect=popen_side_effect):
            agent = _build_agent()
            session = agent.start_session(silent=True)
            first = session.generate("ignored first")
            second = session.generate("ignored second")

        assert first == "STORED"
        assert second == "CEDAR"
        assert session.session_id == "33333333-3333-4333-8333-333333333333"
        assert "--resume" not in captured_cmds[0]
        assert "--resume" in captured_cmds[1]
        resume_idx = captured_cmds[1].index("--resume")
        assert captured_cmds[1][resume_idx + 1] == "33333333-3333-4333-8333-333333333333"

    def test_generate_replays_streaming_fixture_without_duplicate_final_message(self):
        process = _make_process(_read_fixture_lines("streaming_dedup.jsonl"))

        with patch("subprocess.Popen", return_value=process):
            agent = _build_agent()
            result = agent.generate("ignored", silent=True)

        assert result == "Hello"
        assert agent.last_usage.tokens.output_tokens == 5
        assert agent.last_usage.tokens.turns == 1

    def test_generate_replays_tool_and_usage_fixture_and_prefers_usage_event(self):
        recorder = MagicMock()
        process = _make_process(_read_fixture_lines("tool_and_usage.jsonl"))

        with patch("subprocess.Popen", return_value=process):
            agent = _build_agent(recorder=recorder)
            result = agent.generate("ignored", silent=True)

        assert result == "done"
        assert agent.last_usage.provider == "copilot"
        assert agent.last_usage.tokens.input_tokens == 115
        assert agent.last_usage.tokens.output_tokens == 24
        assert agent.last_usage.tokens.cached_input_tokens == 15
        assert agent.last_usage.tokens.turns == 1
        recorder.add_tool_call.assert_called_once()
        call_kwargs = recorder.add_tool_call.call_args.kwargs
        assert call_kwargs["tool"] == "shell"
        assert call_kwargs["args"] == {"command": "printf ok"}
        assert call_kwargs["stdout"] == "ok\nfull"
        assert call_kwargs["exit_code"] == 0
