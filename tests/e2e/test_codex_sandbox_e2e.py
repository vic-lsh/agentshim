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

``excluded_commands`` has both tiers too. ``TestRulesAgainstCodex`` checks
the rendered rules file with ``codex execpolicy check``, Codex's own reader,
and ``TestExemptTurns`` shows in real turns that exactly the exempt command
leaves the sandbox. ``codex sandbox`` cannot show that: it applies the
sandbox to one command and never consults exec-policy rules.
"""

from __future__ import annotations

import json
import os
import shlex
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
from agentshim import ArgvContext, CliAgent, interactive_env
from agentshim.providers.codex import CodexProvider, CodexSandboxConfig, install_rules
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


# -- excluded_commands ------------------------------------------------------

#: Words for rule patterns: quotes, backslashes and ``$`` must reach Codex's
#: Starlark reader as literal characters of one word.
_rule_words = st.text(
    alphabet=st.one_of(
        st.characters(min_codepoint=0x21, max_codepoint=0x7E),
        st.sampled_from(['"', "\\", "'", "$", " ", "é", "\U0001f600"]),
    ),
    min_size=1,
    max_size=8,
).filter(lambda w: not w.startswith("-"))


def _execpolicy(rules: Path, words: list[str]) -> dict[str, Any]:
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [shutil.which("codex") or "codex", "execpolicy", "check", "--rules", str(rules), *words],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


@requires_binary("codex")
class TestRulesAgainstCodex:
    @settings(max_examples=16, suppress_health_check=[HealthCheck.too_slow])
    @given(
        prefix=st.lists(_rule_words, min_size=1, max_size=3),
        extra=st.lists(_rule_words, max_size=2),
    )
    def test_codex_reads_each_command_as_exactly_its_words(
        self, prefix: list[str], extra: list[str]
    ) -> None:
        config = CodexSandboxConfig(excluded_commands=[shlex.join(prefix)])
        with tempfile.TemporaryDirectory() as home:
            rules = install_rules(home, config)
            allowed = _execpolicy(rules, prefix + extra)
            assert allowed["decision"] == "allow"
            assert allowed["matchedRules"][0]["prefixRuleMatch"]["matchedPrefix"] == prefix
            # Refusals, each paired with the positive control above.
            if len(prefix) > 1:
                assert _execpolicy(rules, prefix[:-1])["matchedRules"] == []
            assert _execpolicy(rules, [prefix[0] + "x", *prefix[1:]])["matchedRules"] == []


#: The stand-in for a gateway such as ``sdo detector check``. It always marks
#: that it ran, inside the workspace, so a refused action is told apart from
#: a model that never ran the command; then it tries the two things the
#: sandbox forbids: a unix socket and a file outside the workspace.
_GATEWAY = """#!{python}
import socket, sys
from pathlib import Path

name = Path(sys.argv[0]).name
workspace, outside, sock = Path({workspace!r}), Path({outside!r}), {sock!r}
(workspace / f"{{name}}.ran").write_text("ran")
try:
    client = socket.socket(socket.AF_UNIX)
    client.connect(sock)
    reply = client.recv(16).decode()
except OSError as exc:
    reply = f"denied: {{exc}}"
(workspace / f"{{name}}.socket").write_text(reply)
try:
    (outside / f"{{name}}.out").write_text(reply)
except OSError:
    pass
print(reply)
"""

_EXEMPT, _PLAIN = "agentshim-gateway", "agentshim-gateway-plain"


@contextmanager
def _unix_listener(path: str) -> Iterator[None]:
    """A unix socket that answers ``pong`` until the block exits."""
    server = socket.socket(socket.AF_UNIX)
    server.bind(path)
    server.listen()
    server.settimeout(0.2)
    stop = threading.Event()

    def serve() -> None:
        while not stop.is_set():
            try:
                conn = server.accept()[0]
            except TimeoutError:
                continue
            conn.sendall(b"pong")
            conn.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        server.close()


def _user_codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")


class _Gateway:
    """Where the stand-in gateway leaves its evidence."""

    def __init__(self, workspace: Path, outside: Path, home: Path, env: dict[str, str]) -> None:
        self.workspace, self.outside, self.home, self.env = workspace, outside, home, env

    def ran(self, name: str) -> bool:
        return (self.workspace / f"{name}.ran").exists()

    def socket_reply(self, name: str) -> str:
        return (self.workspace / f"{name}.socket").read_text()

    def wrote_outside(self, name: str) -> bool:
        return (self.outside / f"{name}.out").exists()


@pytest.fixture
def gateway() -> Iterator[_Gateway]:
    """Two identical gateway scripts, a socket, and a dedicated Codex home.

    The home cannot live under the system temp dir: Codex refuses to create
    its sandbox helper there, and every sandboxed command would then fail.
    """
    cache = Path.home() / ".cache"
    cache.mkdir(exist_ok=True)
    with (
        tempfile.TemporaryDirectory() as base,
        tempfile.TemporaryDirectory(prefix="agentshim-e2e-codex-home-", dir=cache) as home,
    ):
        workspace, outside, bin_dir = (Path(base) / n for n in ("ws", "outside", "bin"))
        for directory in (workspace, outside, bin_dir):
            directory.mkdir()
        auth = _user_codex_home() / "auth.json"
        if auth.exists():
            shutil.copy2(auth, Path(home) / "auth.json")
        sock = str(Path(base) / "gw.sock")
        script = _GATEWAY.format(
            python=sys.executable, workspace=str(workspace), outside=str(outside), sock=sock
        )
        for name in (_EXEMPT, _PLAIN):
            (bin_dir / name).write_text(script)
            (bin_dir / name).chmod(0o755)
        env = {
            **interactive_env(),
            "CODEX_HOME": home,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        }
        with _unix_listener(sock):
            yield _Gateway(workspace, outside, Path(home), env)


def _run(command: str) -> str:
    return (
        f"Run exactly this shell command and nothing else: {command}\n"
        "Do not retry or work around a failure. Then reply with DONE."
    )


def _exempting_agent(gw: _Gateway, config: CodexSandboxConfig) -> CliAgent:
    install_rules(gw.home, CodexSandboxConfig(excluded_commands=[f"{_EXEMPT} check"]))
    return CliAgent(
        CodexProvider(sandbox=config), model=model_from_env(CODEX_MODEL_VAR), env=gw.env
    )


_EXEMPTING = CodexSandboxConfig(writable_tmp=False, excluded_commands=[f"{_EXEMPT} check"])


@requires_cli("codex")
class TestExemptTurns:
    def test_only_the_exempt_command_leaves_the_sandbox_also_after_resume(
        self, gateway: _Gateway
    ) -> None:
        session = _exempting_agent(gateway, _EXEMPTING).start_session(cwd=str(gateway.workspace))
        session.turn(_run(f"{_PLAIN} check"))
        assert gateway.ran(_PLAIN), "positive control: the model did not run the command"
        assert gateway.socket_reply(_PLAIN).startswith("denied")
        assert not gateway.wrote_outside(_PLAIN)

        second = session.turn(_run(f"{_EXEMPT} check"))
        assert second.resumed is True
        assert gateway.ran(_EXEMPT)
        assert gateway.socket_reply(_EXEMPT) == "pong"
        assert gateway.wrote_outside(_EXEMPT)

    def test_a_compound_command_line_stays_sandboxed(self, gateway: _Gateway) -> None:
        extra = gateway.outside / "extra.txt"
        _exempting_agent(gateway, _EXEMPTING).run(
            _run(f"{_EXEMPT} check && touch '{extra}'"), cwd=str(gateway.workspace), timeout=300
        )
        assert gateway.ran(_EXEMPT), "positive control: the model did not run the command"
        assert gateway.socket_reply(_EXEMPT).startswith("denied")
        assert not extra.exists()

    def test_without_exemptions_rules_in_the_home_are_ignored(self, gateway: _Gateway) -> None:
        """The same home and rule, but the config exempts nothing: nothing escapes."""
        plain = CodexSandboxConfig(writable_tmp=False)
        _exempting_agent(gateway, plain).run(
            _run(f"{_EXEMPT} check"), cwd=str(gateway.workspace), timeout=300
        )
        assert gateway.ran(_EXEMPT), "positive control: the model did not run the command"
        assert gateway.socket_reply(_EXEMPT).startswith("denied")
        assert not gateway.wrote_outside(_EXEMPT)
