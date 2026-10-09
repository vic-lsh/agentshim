"""Claude's settings-file sandbox and its read-confinement hook."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from agentshim.providers.claude import ClaudeHook, SandboxConfig, build_settings, resolve_sandbox
from agentshim.providers.claude.sandbox import CONFINE_READS_HOOK, workspace_write_settings


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

    @pytest.mark.parametrize("name", ["it's here", 'say "hi"', "cost $HOME", "back`tick`"])
    def test_shell_metacharacters_in_a_path_stay_literal(self, tmp_path: Path, name: str) -> None:
        """Claude runs the hook through a shell, so the path has to be quoted."""
        root = tmp_path / name
        root.mkdir()
        settings = build_settings(SandboxConfig(confine_native_reads_to=[str(root)]))
        command = settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]

        echoed = subprocess.run(  # noqa: S602 - a shell is exactly what Claude uses here
            f"printf '%s' {command.split(' ', 2)[2]}",
            shell=True,
            capture_output=True,
            text=True,
            check=True,
        )

        assert shlex.split(command)[2:] == [str(root)]
        assert echoed.stdout == str(root)


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


class TestWorkspaceWriteSettings:
    def _command(self, settings: dict[str, object]) -> list[str]:
        entry = settings["hooks"]["PreToolUse"][0]  # type: ignore[index]
        return shlex.split(entry["hooks"][0]["command"])

    def _run(self, settings: dict[str, object], payload: dict[str, object]) -> str:
        return subprocess.run(  # noqa: S603 - fixed argv from the settings, no shell
            self._command(settings),
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
        ).stdout

    def test_the_sandbox_writes_to_the_roots_and_allows_no_network(self) -> None:
        settings = workspace_write_settings("/work", ["/data/out"])
        sandbox = settings["sandbox"]
        assert sandbox["filesystem"] == {"allowWrite": ["/data/out"]}
        assert sandbox["failIfUnavailable"] is True
        assert sandbox["allowUnsandboxedCommands"] is False
        assert "network" not in sandbox

    def test_only_the_write_tools_are_confined(self) -> None:
        matcher = workspace_write_settings("/work", [])["hooks"]["PreToolUse"][0]["matcher"]
        tools = set(matcher.split("|"))
        assert tools == {"Edit", "Write", "NotebookEdit", "MultiEdit"}

    def test_a_write_outside_the_workspace_and_its_roots_is_denied(self, tmp_path: Path) -> None:
        work, extra = tmp_path / "work", tmp_path / "extra"
        work.mkdir()
        extra.mkdir()
        settings = workspace_write_settings(str(work), [str(extra)])
        denied = self._run(
            settings, {"tool_name": "Write", "tool_input": {"file_path": str(tmp_path / "x")}}
        )
        assert json.loads(denied)["hookSpecificOutput"]["permissionDecision"] == "deny"
        for root in (work, extra):
            allowed = self._run(
                settings, {"tool_name": "Edit", "tool_input": {"file_path": str(root / "f")}}
            )
            assert allowed == ""

    def test_caller_hooks_follow_the_confinement_hook(self) -> None:
        hook = ClaudeHook(event="PreToolUse", command=["/bin/true"], matcher="Bash")
        entries = workspace_write_settings("/work", [], hooks=[hook])["hooks"]["PreToolUse"]
        assert [entry["matcher"] for entry in entries][1:] == ["Bash"]
