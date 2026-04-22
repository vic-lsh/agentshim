"""Tests for resume_session_id support across all agent_cli backends."""

from unittest.mock import MagicMock

import pytest

from agentshim.claude import ClaudeCodeCodingAgent, ClaudeGenerationSession
from agentshim.claude.events import ClaudeEvent, SystemEvent
from agentshim.cli_agent import CLIAgentSession, CLICodingAgent
from agentshim.codex import CodexCodingAgent, CodexGenerationSession
from agentshim.gemini import GeminiCodingAgent, GeminiGenerationSession
from agentshim.gemini.events import GeminiEvent, InitEvent
from agentshim.opencode import OpencodeCodingAgent, OpencodeGenerationSession
from agentshim.trajectory import NullTrajectoryRecorder


@pytest.fixture
def mock_binaries(monkeypatch):
    monkeypatch.setattr(
        "agentshim.cli_agent.shutil.which",
        lambda cmd, path=None: f"/usr/local/bin/{cmd}",
    )
    monkeypatch.setattr(CLICodingAgent, "_check_cli", lambda self: None)


# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------


class TestClaudeResume:
    def _session(self):
        return ClaudeGenerationSession(
            binary_name="claude",
            env={},
            log_prefix="[Claude]",
            cmd=["claude", "-p"],
            logger=MagicMock(),
            silent=True,
            recorder=NullTrajectoryRecorder(),
        )

    def test_system_event_parses_session_id(self):
        event = ClaudeEvent.from_dict({"type": "system", "subtype": "init", "session_id": "abc-123"})
        assert isinstance(event, SystemEvent)
        assert event.session_id == "abc-123"

    def test_system_event_session_id_optional(self):
        event = ClaudeEvent.from_dict({"type": "system", "subtype": "init"})
        assert isinstance(event, SystemEvent)
        assert event.session_id is None

    def test_session_captures_session_id(self):
        session = self._session()
        session._process_stdout('{"type":"system","subtype":"init","session_id":"abc-123"}\n')
        session._process_stdout('{"type":"result","result":"done"}\n')
        assert session.session_id == "abc-123"

    def test_get_command_includes_resume_flag(self, mock_binaries):
        agent = ClaudeCodeCodingAgent(model="m")
        cmd = agent._get_command("hi", resume_session_id="abc-123")
        assert "--resume" in cmd
        idx = cmd.index("--resume")
        assert cmd[idx + 1] == "abc-123"
        # Resume flag must precede the prompt positional
        assert idx < cmd.index("hi")

    def test_get_command_omits_resume_flag_by_default(self, mock_binaries):
        agent = ClaudeCodeCodingAgent(model="m")
        cmd = agent._get_command("hi")
        assert "--resume" not in cmd


# ---------------------------------------------------------------------------
# Codex
# ---------------------------------------------------------------------------


class TestCodexResume:
    def _session(self):
        return CodexGenerationSession(
            binary_name="codex",
            env={},
            log_prefix="[Codex]",
            cmd=["codex", "exec", "--json"],
            logger=MagicMock(),
            silent=True,
            recorder=NullTrajectoryRecorder(),
        )

    def test_get_command_includes_json_flag(self, mock_binaries):
        agent = CodexCodingAgent()
        cmd = agent._get_command("hi")
        assert "--json" in cmd

    def test_get_command_uses_resume_subcommand(self, mock_binaries):
        agent = CodexCodingAgent()
        cmd = agent._get_command("hi", resume_session_id="uuid-1")
        assert "exec" in cmd
        assert "resume" in cmd
        # `resume` must come right after `exec`, then the id
        exec_idx = cmd.index("exec")
        assert cmd[exec_idx + 1] == "resume"
        assert cmd[exec_idx + 2] == "uuid-1"
        assert "--json" in cmd

    def test_get_command_no_resume_by_default(self, mock_binaries):
        agent = CodexCodingAgent()
        cmd = agent._get_command("hi")
        assert "resume" not in cmd

    def test_session_captures_thread_id(self):
        session = self._session()
        session._process_stdout('{"type":"thread.started","thread_id":"uuid-1"}\n')
        session._process_stdout('{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"hi"}}\n')
        assert session.session_id == "uuid-1"
        assert "hi" in session.stdout_lines


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


class TestGeminiResume:
    def _session(self):
        return GeminiGenerationSession(
            binary_name="gemini",
            env={},
            log_prefix="[Gemini]",
            cmd=["gemini"],
            logger=MagicMock(),
            silent=True,
            recorder=NullTrajectoryRecorder(),
        )

    def test_init_event_parses_session_id(self):
        event = GeminiEvent.from_dict({"type": "init", "session_id": "uuid-2", "model": "gemini-3"})
        assert isinstance(event, InitEvent)
        assert event.session_id == "uuid-2"

    def test_session_captures_session_id(self):
        session = self._session()
        session._process_stdout('{"type":"init","session_id":"uuid-2","model":"gemini-3"}\n')
        assert session.session_id == "uuid-2"

    def test_get_command_includes_resume_flag(self, mock_binaries):
        agent = GeminiCodingAgent()
        cmd = agent._get_command("hi", resume_session_id="uuid-2")
        assert "--resume" in cmd
        idx = cmd.index("--resume")
        assert cmd[idx + 1] == "uuid-2"

    def test_get_command_omits_resume_flag_by_default(self, mock_binaries):
        agent = GeminiCodingAgent()
        cmd = agent._get_command("hi")
        assert "--resume" not in cmd


# ---------------------------------------------------------------------------
# Opencode
# ---------------------------------------------------------------------------


class TestOpencodeResume:
    def _session(self):
        return OpencodeGenerationSession(
            binary_name="opencode",
            env={},
            log_prefix="[Opencode]",
            cmd=["opencode", "run"],
            logger=MagicMock(),
            silent=True,
            recorder=NullTrajectoryRecorder(),
        )

    def test_session_captures_session_id(self):
        session = self._session()
        session._process_stdout('{"type":"step_start","sessionID":"ses_abc","part":{"type":"step-start"}}\n')
        assert session.session_id == "ses_abc"

    def test_get_command_includes_session_flag(self, mock_binaries):
        agent = OpencodeCodingAgent()
        cmd = agent._get_command("hi", resume_session_id="ses_abc")
        assert "--session" in cmd
        idx = cmd.index("--session")
        assert cmd[idx + 1] == "ses_abc"
        # --session must come after `run`
        assert cmd.index("run") < idx

    def test_get_command_omits_session_flag_by_default(self, mock_binaries):
        agent = OpencodeCodingAgent()
        cmd = agent._get_command("hi")
        assert "--session" not in cmd


# ---------------------------------------------------------------------------
# Session API (start_session → CLIAgentSession.generate)
# ---------------------------------------------------------------------------


class _StubRunSession:
    """Replaces the per-call run-session created by ``_create_session``.

    Records the cmd it was given and emits a fixed ``session_id`` from
    its first ``run`` call so we can verify auto-resume wiring.
    """

    def __init__(self, cmd, cwd, timeout, silent, session_id="sid-1", reply="ok"):
        self.cmd = cmd
        self.cwd = cwd
        self.timeout = timeout
        self.silent = silent
        self.session_id = session_id
        self._reply = reply

    def run(self, prompt):
        return self._reply


class TestSessionAPI:
    def _make_agent(self, mock_binaries, captured):
        agent = ClaudeCodeCodingAgent(model="m")

        def fake_create_session(cmd, cwd=None, timeout=300, silent=False, **_):
            stub = _StubRunSession(cmd, cwd, timeout, silent)
            captured.append(stub)
            return stub

        agent._create_session = fake_create_session  # type: ignore[method-assign]
        return agent

    def test_start_session_returns_session(self, mock_binaries):
        agent = ClaudeCodeCodingAgent()
        sess = agent.start_session(cwd="/tmp", timeout=60)
        assert isinstance(sess, CLIAgentSession)
        assert sess.session_id is None

    def test_first_call_omits_resume_flag(self, mock_binaries):
        captured: list[_StubRunSession] = []
        agent = self._make_agent(mock_binaries, captured)
        sess = agent.start_session()
        sess.generate("hello")
        assert "--resume" not in captured[0].cmd

    def test_second_call_uses_captured_session_id(self, mock_binaries):
        captured: list[_StubRunSession] = []
        agent = self._make_agent(mock_binaries, captured)
        sess = agent.start_session()
        sess.generate("hello")
        # session_id captured from first stub's emitted id
        assert sess.session_id == "sid-1"
        sess.generate("follow up")
        cmd = captured[1].cmd
        assert "--resume" in cmd
        assert cmd[cmd.index("--resume") + 1] == "sid-1"

    def test_session_default_cwd_used(self, mock_binaries):
        captured: list[_StubRunSession] = []
        agent = self._make_agent(mock_binaries, captured)
        sess = agent.start_session(cwd="/sess/default", timeout=120)
        sess.generate("hi")
        assert captured[0].cwd == "/sess/default"
        assert captured[0].timeout == 120

    def test_per_call_overrides_session_defaults(self, mock_binaries):
        captured: list[_StubRunSession] = []
        agent = self._make_agent(mock_binaries, captured)
        sess = agent.start_session(cwd="/sess/default", timeout=120, silent=True)
        sess.generate("hi", cwd="/per/call", timeout=30, silent=False)
        assert captured[0].cwd == "/per/call"
        assert captured[0].timeout == 30
        assert captured[0].silent is False

    def test_agent_generate_is_one_shot(self, mock_binaries):
        """``agent.generate`` does not retain state across calls."""
        captured: list[_StubRunSession] = []
        agent = self._make_agent(mock_binaries, captured)
        agent.generate("first")
        agent.generate("second")
        # Two independent sessions; second call must NOT pass --resume
        assert "--resume" not in captured[0].cmd
        assert "--resume" not in captured[1].cmd
