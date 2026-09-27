"""``CodexSandboxConfig`` against the real Codex sandbox.

Two tiers:

- ``TestEnforcement`` runs the exact ``--config`` overrides
  ``CodexProvider`` renders through ``codex sandbox``, which applies Codex's
  own sandbox to a plain command. No model, no credentials, deterministic,
  so it runs whenever ``codex`` is installed. Hypothesis draws configs and a
  small oracle predicts what each one must allow; the real sandbox has to
  agree on every probe.
- ``TestTurns`` runs real ``codex exec`` turns (``AGENTSHIM_E2E=1``). It
  covers what only a turn can show: a sandboxed headless turn neither hangs
  on an approval nor escapes, and a resumed turn keeps the sandbox. Every
  refusal is paired with a positive control, so a model that simply declined
  to run the command cannot pass as an enforced sandbox.
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from agentshim import ArgvContext, CliAgent
from agentshim.providers.codex import CodexProvider, CodexSandboxConfig
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.e2e.conftest import CODEX_MODEL_VAR, model_from_env, requires_binary, requires_cli

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.e2e

_SANDBOX_KEYS = ("sandbox_mode", "approval_policy", "sandbox_workspace_write.")

#: Runs inside the sandbox; reports which probes succeeded as JSON.
_PROBE = """
import json, socket, sys
from pathlib import Path

def can_write(directory):
    try:
        (Path(directory) / "probe").write_text("x")
        return True
    except OSError:
        return False

def can_connect(port):
    try:
        socket.create_connection(("127.0.0.1", int(port)), timeout=2).close()
        return True
    except OSError:
        return False

workspace, root, other, port = sys.argv[1:]
print(json.dumps({
    "workspace": can_write(workspace),
    "listed_root": can_write(root),
    "unlisted_dir": can_write(other),
    "network": can_connect(port),
}))
"""


def _sandbox_flags(config: CodexSandboxConfig) -> list[str]:
    """The sandbox ``--config`` pairs, taken from a real rendered argv."""
    argv = CodexProvider(sandbox=config).build_argv(
        ArgvContext("codex", None, {}, None, None, None, None)
    )
    flags: list[str] = []
    for index, arg in enumerate(argv[:-1]):
        if arg == "--config" and argv[index + 1].startswith(_SANDBOX_KEYS):
            flags += ["--config", argv[index + 1]]
    return flags


@contextmanager
def _listener() -> Iterator[int]:
    """A loopback TCP listener that accepts until the block exits."""
    server = socket.create_server(("127.0.0.1", 0))
    server.settimeout(0.2)
    stop = threading.Event()

    def accept_one() -> None:
        try:
            server.accept()[0].close()
        except TimeoutError:
            return

    def accept() -> None:
        while not stop.is_set():
            accept_one()

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    try:
        yield server.getsockname()[1]
    finally:
        stop.set()
        thread.join()
        server.close()


def _expected(config: CodexSandboxConfig, *, root_listed: bool) -> dict[str, bool]:
    """What Codex's documented semantics say this config allows.

    Every probe directory lives under the system temp dir, so ``writable_tmp``
    makes all of them writable in ``workspace-write``.
    """
    if config.mode == "danger-full-access":
        return dict.fromkeys(("workspace", "listed_root", "unlisted_dir", "network"), True)
    if config.mode == "read-only":
        return dict.fromkeys(("workspace", "listed_root", "unlisted_dir", "network"), False)
    return {
        "workspace": True,
        "listed_root": config.writable_tmp or root_listed,
        "unlisted_dir": config.writable_tmp,
        "network": config.network_access,
    }


_modes = st.sampled_from(["read-only", "workspace-write", "danger-full-access"])


@requires_binary("codex")
class TestEnforcement:
    @settings(max_examples=24, suppress_health_check=[HealthCheck.too_slow])
    @given(
        mode=_modes,
        root_listed=st.booleans(),
        network_access=st.booleans(),
        writable_tmp=st.booleans(),
    )
    def test_the_real_sandbox_enforces_exactly_the_rendered_config(
        self, mode: str, *, root_listed: bool, network_access: bool, writable_tmp: bool
    ) -> None:
        with tempfile.TemporaryDirectory() as base, _listener() as port:
            workspace, root, other = (Path(base) / name for name in ("ws", "root", "other"))
            for directory in (workspace, root, other):
                directory.mkdir()
            ww = mode == "workspace-write"
            config = CodexSandboxConfig(
                mode=mode,  # type: ignore[arg-type]
                writable_roots=[str(root)] if ww and root_listed else [],
                network_access=ww and network_access,
                writable_tmp=writable_tmp or not ww,
            )
            completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
                [
                    shutil.which("codex") or "codex",
                    "sandbox",
                    *_sandbox_flags(config),
                    "--",
                    sys.executable,
                    "-c",
                    _PROBE,
                    str(workspace),
                    str(root),
                    str(other),
                    str(port),
                ],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            assert completed.returncode == 0, completed.stderr
            observed = json.loads(completed.stdout.strip().splitlines()[-1])
            assert observed == _expected(config, root_listed=ww and root_listed), config


def _agent(sandbox: CodexSandboxConfig) -> CliAgent:
    return CliAgent(CodexProvider(sandbox=sandbox), model=model_from_env(CODEX_MODEL_VAR))


def _touch(path: Path) -> str:
    return (
        f"Run exactly this shell command and nothing else: touch '{path}'\n"
        "Do not retry or work around a failure. Then reply with DONE."
    )


@pytest.fixture
def dirs() -> Iterator[tuple[Path, Path]]:
    """A workspace and a sibling directory, both outside any writable tmp."""
    with tempfile.TemporaryDirectory() as base:
        workspace, outside = Path(base) / "ws", Path(base) / "outside"
        workspace.mkdir()
        outside.mkdir()
        yield workspace, outside


_CONFINED = CodexSandboxConfig(mode="workspace-write", writable_tmp=False)


@requires_cli("codex")
class TestTurns:
    def test_read_only_blocks_a_write_the_workspace_mode_allows(
        self, dirs: tuple[Path, Path]
    ) -> None:
        workspace, _ = dirs
        control = workspace / "control.txt"
        _agent(_CONFINED).run(_touch(control), cwd=str(workspace), timeout=300)
        assert control.exists(), "positive control: the model did not run the command"

        blocked = workspace / "blocked.txt"
        result = _agent(CodexSandboxConfig(mode="read-only")).run(
            _touch(blocked), cwd=str(workspace), timeout=300
        )
        assert not blocked.exists()
        assert result.exit_code == 0

    def test_workspace_write_blocks_outside_until_the_dir_is_a_writable_root(
        self, dirs: tuple[Path, Path]
    ) -> None:
        workspace, outside = dirs
        denied = outside / "denied.txt"
        _agent(_CONFINED).run(_touch(denied), cwd=str(workspace), timeout=300)
        assert not denied.exists()

        allowed = outside / "allowed.txt"
        widened = CodexSandboxConfig(writable_roots=[str(outside)], writable_tmp=False)
        _agent(widened).run(_touch(allowed), cwd=str(workspace), timeout=300)
        assert allowed.exists(), "positive control: the model did not run the command"

    def test_a_resumed_turn_keeps_the_sandbox(self, dirs: tuple[Path, Path]) -> None:
        workspace, outside = dirs
        session = _agent(_CONFINED).start_session(cwd=str(workspace))
        first = session.turn(_touch(workspace / "first.txt"))
        assert (workspace / "first.txt").exists(), "positive control failed"
        assert first.session_id is not None

        second = session.turn(_touch(outside / "second.txt"))
        assert second.resumed is True
        assert not (outside / "second.txt").exists()


def test_the_probe_oracle_covers_every_mode() -> None:
    """Guard the oracle itself: each mode must be distinguishable."""
    outcomes: dict[str, Any] = {
        mode: _expected(CodexSandboxConfig(mode=mode), root_listed=False)  # type: ignore[arg-type]
        for mode in ("read-only", "workspace-write", "danger-full-access")
    }
    assert len({json.dumps(o, sort_keys=True) for o in outcomes.values()}) == 3
