import json
import subprocess
import sys
from pathlib import Path

import pytest

import agentshim.hooks.confine_reads
from agentshim.sandbox import (
    SandboxConfig,
    build_claude_sandbox_settings,
    resolve_sandbox,
)


class TestResolveSandbox:
    def test_false_returns_none(self):
        assert resolve_sandbox(False) is None

    def test_none_returns_none(self):
        assert resolve_sandbox(None) is None

    def test_true_returns_default_config(self):
        cfg = resolve_sandbox(True)
        assert isinstance(cfg, SandboxConfig)
        assert cfg.fail_if_unavailable is True
        assert cfg.auto_allow_bash is True

    def test_passthrough_config(self):
        cfg = SandboxConfig(fail_if_unavailable=False)
        assert resolve_sandbox(cfg) is cfg

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            resolve_sandbox("yes")  # type: ignore[arg-type]


class TestBuildClaudeSandboxSettings:
    def test_default_enables_sandbox(self):
        payload = build_claude_sandbox_settings(SandboxConfig())
        assert payload == {
            "sandbox": {
                "enabled": True,
                "failIfUnavailable": True,
                "autoAllowBashIfSandboxed": True,
                "allowUnsandboxedCommands": False,
            }
        }

    def test_filesystem_section_populated(self):
        cfg = SandboxConfig(
            allow_write=["/tmp/build"],
            deny_read=["~/.aws/credentials"],
        )
        payload = build_claude_sandbox_settings(cfg)
        fs = payload["sandbox"]["filesystem"]
        assert fs == {"allowWrite": ["/tmp/build"], "denyRead": ["~/.aws/credentials"]}

    def test_network_allowed_domains(self):
        cfg = SandboxConfig(allowed_domains=["github.com", "*.npmjs.org"])
        payload = build_claude_sandbox_settings(cfg)
        assert payload["sandbox"]["network"] == {
            "allowedDomains": ["github.com", "*.npmjs.org"],
        }

    def test_excluded_commands(self):
        cfg = SandboxConfig(excluded_commands=["docker *"])
        payload = build_claude_sandbox_settings(cfg)
        assert payload["sandbox"]["excludedCommands"] == ["docker *"]

    def test_extra_settings_merged(self):
        cfg = SandboxConfig(extra_settings={"enableWeakerNestedSandbox": True})
        payload = build_claude_sandbox_settings(cfg)
        assert payload["sandbox"]["enableWeakerNestedSandbox"] is True


class TestConfineReadsHook:
    """Tests for the PreToolUse confine-reads hook integration."""

    def test_no_hook_when_disabled(self):
        payload = build_claude_sandbox_settings(SandboxConfig())
        assert "hooks" not in payload

    def test_hook_emitted_when_roots_set(self, tmp_path):
        cfg = SandboxConfig(confine_native_reads_to=[str(tmp_path)])
        payload = build_claude_sandbox_settings(cfg)
        assert "hooks" in payload
        entries = payload["hooks"]["PreToolUse"]
        assert len(entries) == 1
        assert entries[0]["matcher"] == "Read|Glob|Grep|Edit|Write|NotebookEdit"
        command = entries[0]["hooks"][0]["command"]
        assert "confine_reads.py" in command
        assert str(tmp_path) in command

    def _run_hook(self, payload: dict, roots: list[str]) -> tuple[int, str]:
        hook_path = Path(agentshim.hooks.confine_reads.__file__).resolve()
        proc = subprocess.run(
            [sys.executable, str(hook_path), *roots],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
        )
        return proc.returncode, proc.stdout

    def test_hook_allows_path_inside_root(self, tmp_path):
        (tmp_path / "foo.txt").write_text("hi")
        rc, out = self._run_hook(
            {"tool_name": "Read", "tool_input": {"file_path": str(tmp_path / "foo.txt")}},
            [str(tmp_path)],
        )
        assert rc == 0
        assert out == ""

    def test_hook_denies_path_outside_root(self, tmp_path):
        rc, out = self._run_hook(
            {"tool_name": "Read", "tool_input": {"file_path": "/etc/passwd"}},
            [str(tmp_path)],
        )
        assert rc == 0
        decision = json.loads(out)
        hso = decision["hookSpecificOutput"]
        assert hso["permissionDecision"] == "deny"
        assert "/etc/passwd" in hso["permissionDecisionReason"]

    def test_hook_handles_glob_path_kwarg(self, tmp_path):
        rc, out = self._run_hook(
            {"tool_name": "Glob", "tool_input": {"pattern": "**/*.py", "path": "/var/log"}},
            [str(tmp_path)],
        )
        assert rc == 0
        decision = json.loads(out)
        assert decision["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_hook_allows_tool_without_path(self, tmp_path):
        rc, out = self._run_hook(
            {"tool_name": "Glob", "tool_input": {"pattern": "**/*.py"}},
            [str(tmp_path)],
        )
        assert rc == 0
        assert out == ""
