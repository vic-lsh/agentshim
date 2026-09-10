"""Claude's settings-file sandbox and its read-confinement hook."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from agentshim.providers.claude import SandboxConfig, build_settings, resolve_sandbox
from agentshim.providers.claude.sandbox import CONFINE_READS_HOOK


class TestResolveSandbox:
    @pytest.mark.parametrize("value", [None, False])
    def test_off_values_yield_none(self, *, value: bool | None) -> None:
        assert resolve_sandbox(value) is None

    def test_true_yields_the_default_config(self) -> None:
        config = resolve_sandbox(value=True)
        assert isinstance(config, SandboxConfig)
        assert config.fail_if_unavailable is True
        assert config.auto_allow_bash is True

    def test_a_config_passes_through_unchanged(self) -> None:
        config = SandboxConfig(fail_if_unavailable=False)
        assert resolve_sandbox(config) is config

    def test_other_types_are_rejected(self) -> None:
        with pytest.raises(TypeError):
            resolve_sandbox("yes")  # type: ignore[arg-type]


class TestSettings:
    def test_defaults(self) -> None:
        assert build_settings(SandboxConfig()) == {
            "sandbox": {
                "enabled": True,
                "failIfUnavailable": True,
                "autoAllowBashIfSandboxed": True,
                "allowUnsandboxedCommands": False,
            }
        }

    def test_filesystem_section(self) -> None:
        config = SandboxConfig(allow_write=["/work/build"], deny_read=["~/.aws/credentials"])
        assert build_settings(config)["sandbox"]["filesystem"] == {
            "allowWrite": ["/work/build"],
            "denyRead": ["~/.aws/credentials"],
        }

    def test_network_section(self) -> None:
        config = SandboxConfig(allowed_domains=["github.com", "*.npmjs.org"])
        assert build_settings(config)["sandbox"]["network"] == {
            "allowedDomains": ["github.com", "*.npmjs.org"]
        }

    def test_excluded_commands(self) -> None:
        config = SandboxConfig(excluded_commands=["docker *"])
        assert build_settings(config)["sandbox"]["excludedCommands"] == ["docker *"]

    def test_extra_settings_are_merged(self) -> None:
        config = SandboxConfig(extra_settings={"enableWeakerNestedSandbox": True})
        assert build_settings(config)["sandbox"]["enableWeakerNestedSandbox"] is True


class TestConfineReadsWiring:
    def test_no_hook_block_by_default(self) -> None:
        assert "hooks" not in build_settings(SandboxConfig())

    def test_hook_block_matches_the_native_file_tools(self, tmp_path: Path) -> None:
        settings = build_settings(SandboxConfig(confine_native_reads_to=[str(tmp_path)]))
        entries = settings["hooks"]["PreToolUse"]
        assert len(entries) == 1
        assert entries[0]["matcher"] == "Read|Glob|Grep|Edit|Write|NotebookEdit"

    def test_the_hook_runs_through_sys_executable(self, tmp_path: Path) -> None:
        """Not ``python3`` off the agent's PATH, and not the file's own +x bit."""
        settings = build_settings(SandboxConfig(confine_native_reads_to=[str(tmp_path)]))
        command = settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
        parts = shlex.split(command)
        assert parts[0] == sys.executable
        assert parts[1] == CONFINE_READS_HOOK
        assert parts[2:] == [str(tmp_path)]

    def test_paths_with_spaces_survive_shell_parsing(self, tmp_path: Path) -> None:
        root = tmp_path / "my project"
        root.mkdir()
        settings = build_settings(SandboxConfig(confine_native_reads_to=[str(root)]))
        command = settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
        assert shlex.split(command)[2:] == [str(root)]


class TestConfineReadsHook:
    def _run(self, payload: dict[str, object], roots: list[str]) -> tuple[int, str]:
        """Invoke the hook exactly the way the settings block does."""
        settings = build_settings(SandboxConfig(confine_native_reads_to=roots))
        command = settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
        proc = subprocess.run(  # noqa: S603 - fixed argv from build_settings, no shell
            shlex.split(command),
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
        )
        return proc.returncode, proc.stdout

    def test_a_path_inside_a_root_is_allowed(self, tmp_path: Path) -> None:
        (tmp_path / "foo.txt").write_text("hi")
        code, out = self._run(
            {"tool_name": "Read", "tool_input": {"file_path": str(tmp_path / "foo.txt")}},
            [str(tmp_path)],
        )
        assert code == 0
        assert out == ""

    def test_a_path_outside_every_root_is_denied(self, tmp_path: Path) -> None:
        code, out = self._run(
            {"tool_name": "Read", "tool_input": {"file_path": "/etc/passwd"}},
            [str(tmp_path)],
        )
        assert code == 0
        decision = json.loads(out)["hookSpecificOutput"]
        assert decision["permissionDecision"] == "deny"
        assert "/etc/passwd" in decision["permissionDecisionReason"]

    def test_the_glob_path_argument_is_checked(self, tmp_path: Path) -> None:
        code, out = self._run(
            {"tool_name": "Glob", "tool_input": {"pattern": "**/*.py", "path": "/var/log"}},
            [str(tmp_path)],
        )
        assert code == 0
        assert json.loads(out)["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_a_tool_call_without_a_path_is_allowed(self, tmp_path: Path) -> None:
        code, out = self._run(
            {"tool_name": "Glob", "tool_input": {"pattern": "**/*.py"}}, [str(tmp_path)]
        )
        assert code == 0
        assert out == ""

    def test_invalid_hook_input_does_not_block_the_tool(self, tmp_path: Path) -> None:
        settings = build_settings(SandboxConfig(confine_native_reads_to=[str(tmp_path)]))
        command = settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
        proc = subprocess.run(  # noqa: S603 - fixed argv from build_settings, no shell
            shlex.split(command), input="not json", text=True, capture_output=True, check=False
        )
        assert proc.returncode == 0
        assert proc.stdout == ""
