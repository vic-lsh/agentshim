"""Only ``AgentShimError`` subclasses escape the public API.

The documented contract is that a caller catches `AgentShimError` and is
done: no bare `OSError`, no `ValueError`, no `subprocess` exception. These
are the paths that used to break it, one test per underlying operation, so a
regression shows up as the wrong exception type rather than as a caller's
crash in production.
"""

from __future__ import annotations

import os
import stat
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from agentshim import (
    AgentShimError,
    CliAgent,
    CliCheckError,
    HostCommandExecutor,
    McpConfigError,
    OutputSchema,
    ProviderCapabilityError,
    StdioMcpServer,
    TransformingExecutor,
    TurnRequest,
)
from agentshim.testing import FakeExecutor, scripted_turn

if TYPE_CHECKING:
    from agentshim import CommandRequest

_ENV = {"PATH": "/usr/bin:/bin", "HOME": "/home/tester"}

_NAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"score": {"type": "number", "minimum": float("nan")}},
    "required": ["score"],
    "additionalProperties": False,
}


def _agent(provider: str = "claude") -> CliAgent:
    return CliAgent(provider, executor=FakeExecutor(scripted_turn(provider, text="ok")), env=_ENV)


class TestMcpInstallCannotWrite:
    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
    def test_a_read_only_workspace_raises_mcp_config_error(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        workspace.chmod(stat.S_IRUSR | stat.S_IXUSR)
        try:
            with pytest.raises(McpConfigError) as excinfo:
                _agent().start_session(cwd=str(workspace)).turn(
                    TurnRequest(
                        prompt="hi",
                        mcp_servers=[StdioMcpServer(name="board", command="python")],
                    )
                )
        finally:
            workspace.chmod(stat.S_IRWXU)

        assert isinstance(excinfo.value, AgentShimError)


class TestSchemaCannotBeSerialized:
    def test_a_nan_in_a_schema_raises_provider_capability_error(self, tmp_path: Path) -> None:
        """Codex takes the schema as a file, so this one reaches ``json.dumps``."""
        with pytest.raises(ProviderCapabilityError) as excinfo:
            _agent("codex").start_session().turn(
                TurnRequest(
                    prompt="hi",
                    output_schema=OutputSchema(schema=_NAN_SCHEMA, host_dir=tmp_path),
                )
            )

        assert isinstance(excinfo.value, AgentShimError)
        assert not list(tmp_path.iterdir())


class TestTransformedCommandCannotRun:
    def test_a_missing_prefix_binary_raises_cli_check_error(self, tmp_path: Path) -> None:
        """The transform usually prefixes ``docker exec`` or a sandbox wrapper."""
        missing = str(tmp_path / "no-such-launcher")

        def prefix(request: CommandRequest) -> CommandRequest:
            return replace(request, argv=[missing, *request.argv])

        executor = TransformingExecutor(
            HostCommandExecutor(),
            prefix,
            find_binary=lambda name, env: f"/opt/{name}",  # noqa: ARG005
        )

        with pytest.raises(CliCheckError) as excinfo:
            CliAgent("claude", executor=executor, env=_ENV)

        assert isinstance(excinfo.value, AgentShimError)
        assert missing in str(excinfo.value)
