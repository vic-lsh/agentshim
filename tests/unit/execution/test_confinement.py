"""Confinements: the contract, docker argv shape, path mapping, reaping, ``confine``."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest
from agentshim import (
    AgentShimError,
    CliNotFoundError,
    CommandRequest,
    Confinement,
    DockerExecConfinement,
    NullSink,
    ReapError,
    SpawnRequest,
    TransformingExecutor,
    confine,
)
from agentshim.testing import EchoPeer, FakeConfinement, FakeExecutor, FakeRun
from agentshim.testing.contracts import ConfinementContract
from hypothesis import given
from hypothesis import strategies as st

_SEGMENT = st.text(alphabet="abcxyz_-", min_size=1, max_size=4)
_PARTS = st.lists(_SEGMENT, max_size=4)
_ARGV = st.lists(st.text(max_size=8), min_size=1, max_size=6)


def _docker(
    runner: FakeExecutor | None = None,
    *,
    container: str = "ctr1",
    path_map: dict[str, str] | None = None,
    env: dict[str, str] | None = None,
    user: str | None = None,
) -> DockerExecConfinement:
    return DockerExecConfinement(
        lambda: container,
        runner=runner if runner is not None else FakeExecutor(FakeRun()),
        path_map=path_map,
        env=env if env is not None else {},
        user=user,
    )


class TestDockerExecConfinementContract(ConfinementContract):
    def make_confinement(self) -> Confinement:
        return _docker(path_map={"/host/ws": "/ws"}, env={"K": "v"}, user="agent")

    def host_paths(self) -> list[str]:
        return [*super().host_paths(), "/host/ws", "/host/ws/a/b", "/host/wsx"]


class TestFakeConfinementContract(ConfinementContract):
    def make_confinement(self) -> Confinement:
        return FakeConfinement(path_map={"/host/ws": "/ws"}, env={"K": "v"})

    def host_paths(self) -> list[str]:
        return [*super().host_paths(), "/host/ws", "/host/ws/a/b"]


class TestDockerWrap:
    def test_the_full_shape(self) -> None:
        confinement = _docker(path_map={"/h": "/c"}, env={"A": "1", "B": "2"}, user="u")
        assert confinement.wrap(["codex", "app-server"], "/h/w") == [
            "docker",
            "exec",
            "-i",
            "-e",
            "AGENTSHIM_CONFINED=1",
            "-u",
            "u",
            "-w",
            "/c/w",
            "-e",
            "A",
            "-e",
            "B",
            "ctr1",
            "codex",
            "app-server",
        ]

    def test_env_values_never_reach_argv(self) -> None:
        confinement = _docker(env={"TOKEN": "s3cret"})
        assert "s3cret" not in " ".join(confinement.wrap(["x"], None))
        assert confinement.env == {"TOKEN": "s3cret"}

    def test_the_container_is_read_on_every_call(self) -> None:
        current = ["old"]
        confinement = DockerExecConfinement(
            lambda: current[0], runner=FakeExecutor(FakeRun()), env={}
        )
        assert "old" in confinement.wrap(["x"], None)
        current[0] = "new"
        wrapped = confinement.wrap(["x"], None)
        assert "new" in wrapped
        assert "old" not in wrapped

    def test_no_cwd_and_no_user_means_no_flags(self) -> None:
        wrapped = _docker().wrap(["x"], None)
        assert "-w" not in wrapped
        assert "-u" not in wrapped

    @pytest.mark.parametrize("name", ["", "A=B", "A\x00"])
    def test_bad_env_names_are_rejected(self, name: str) -> None:
        with pytest.raises(ValueError, match="environment variable"):
            _docker(env={name: "v"})

    @given(_ARGV, _PARTS)
    def test_argv_tail_is_preserved_and_cwd_is_mapped(
        self, argv: list[str], parts: list[str]
    ) -> None:
        confinement = _docker(path_map={"/h": "/c"})
        cwd = "/".join(["/h", *parts])
        wrapped = confinement.wrap(argv, cwd)
        assert wrapped[len(wrapped) - len(argv) :] == argv
        assert wrapped[wrapped.index("-w") + 1] == "/".join(["/c", *parts])
        assert wrapped[wrapped.index("-w") - 0 :].count("-w") == 1


class TestAgentPath:
    def test_the_longest_prefix_wins(self) -> None:
        confinement = _docker(path_map={"/a": "/x", "/a/b": "/y"})
        assert confinement.agent_path("/a/b/c") == "/y/c"
        assert confinement.agent_path("/a/q") == "/x/q"

    def test_a_prefix_matches_whole_components_only(self) -> None:
        confinement = _docker(path_map={"/a/b": "/y"})
        assert confinement.agent_path("/a/bc") == "/a/bc"
        assert confinement.agent_path("/a/b") == "/y"

    def test_unmapped_paths_are_identity_up_to_normalization(self) -> None:
        confinement = _docker()
        assert confinement.agent_path("/a//b/./c/") == "/a/b/c"
        assert confinement.agent_path("/a/../b") == "/a/../b"
        assert confinement.agent_path(Path("/p/q")) == "/p/q"

    @given(_PARTS, _PARTS)
    def test_a_mapped_path_keeps_its_suffix(self, host: list[str], tail: list[str]) -> None:
        confinement = _docker(path_map={"/h": "/c"})
        path = "/".join(["/h", *host, *tail])
        assert confinement.agent_path(path) == "/".join(["/c", *host, *tail])

    @given(_PARTS)
    def test_unrelated_paths_are_untouched(self, parts: list[str]) -> None:
        confinement = _docker(path_map={"/h": "/c"})
        path = "/".join(["/other", *parts])
        assert confinement.agent_path(path) == PurePosixPath(path).as_posix()

    @given(st.lists(st.sampled_from(["a", "b", ".", "", ".."]), max_size=6))
    def test_normalization_is_idempotent(self, parts: list[str]) -> None:
        confinement = _docker()
        once = confinement.agent_path("/" + "/".join(parts))
        assert confinement.agent_path(once) == once


class TestReap:
    def test_it_runs_one_sh_script_through_docker_exec_in_the_container(self) -> None:
        runner = FakeExecutor(FakeRun())
        _docker(runner, container="ctrX").reap()
        (request,) = runner.requests
        assert list(request.argv[:5]) == ["docker", "exec", "ctrX", "sh", "-c"]
        assert "AGENTSHIM_CONFINED=1" in request.argv[5]

    def test_a_missing_container_is_nothing_to_reap(self) -> None:
        runner = FakeExecutor(FakeRun(stderr=["Error: No such container\n"], returncode=1))
        confinement = _docker(runner)
        confinement.reap()
        confinement.reap()
        assert len(runner.requests) == 2

    def test_a_stopped_container_is_nothing_to_reap(self) -> None:
        run = FakeRun(
            stderr=["Error response from daemon: Container abc is not running\n"], returncode=1
        )
        _docker(FakeExecutor(run)).reap()

    def test_a_missing_docker_client_is_an_error(self) -> None:
        def run(request: CommandRequest) -> FakeRun:
            raise FileNotFoundError(request.argv[0])

        with pytest.raises(ReapError):
            _docker(FakeExecutor(run)).reap()

    def test_a_daemon_error_is_an_error(self) -> None:
        run = FakeRun(stderr=["Cannot connect to the Docker daemon\n"], returncode=1)
        with pytest.raises(ReapError, match="Cannot connect"):
            _docker(FakeExecutor(run)).reap()

    def test_a_timeout_is_an_error(self) -> None:
        with pytest.raises(AgentShimError):
            _docker(FakeExecutor(FakeRun(timeout=True))).reap()

    def test_the_marker_name_cannot_be_overridden_by_env(self) -> None:
        with pytest.raises(ValueError, match="AGENTSHIM_CONFINED"):
            _docker(env={"AGENTSHIM_CONFINED": "0"})

    @pytest.mark.skipif(not Path("/proc/self/environ").exists(), reason="needs /proc")
    def test_the_script_kills_marked_processes_and_spares_the_rest(self) -> None:
        runner = FakeExecutor(FakeRun())
        _docker(runner).reap()
        script = runner.requests[0].argv[5]

        sleeper = [sys.executable, "-c", "import time; time.sleep(600)"]
        marked = subprocess.Popen(sleeper, env={**os.environ, "AGENTSHIM_CONFINED": "1"})  # noqa: S603
        bystander = subprocess.Popen(sleeper, env={**os.environ, "AGENTSHIM_CONFINED": "0"})  # noqa: S603
        try:
            subprocess.run([shutil.which("sh") or "sh", "-c", script], check=True)  # noqa: S603
            assert marked.wait(timeout=30) == -9
            assert bystander.poll() is None
        finally:
            marked.kill()
            bystander.kill()
            marked.wait()
            bystander.wait()


class TestConfine:
    def _confined(self) -> tuple[FakeExecutor, FakeConfinement, TransformingExecutor]:
        inner = FakeExecutor(FakeRun(stdout=["ok\n"]), peers=lambda _r: EchoPeer())
        confinement = FakeConfinement(env={"K": "v"})
        executor = confine(inner, confinement)
        assert isinstance(executor, TransformingExecutor)
        return inner, confinement, executor

    def test_run_goes_through_wrap_with_the_agent_cwd_and_the_confinement_env(self) -> None:
        inner, confinement, executor = self._confined()
        executor.run(
            CommandRequest(argv=["codex"], stdin="s", cwd="/w", env={"PATH": "/bin"}, timeout=3),
            NullSink(),
        )
        (request,) = inner.requests
        assert list(request.argv) == ["fake-confine", "codex"]
        assert confinement.wraps == [(("codex",), "/w")]
        assert request.cwd is None
        assert dict(request.env) == {"PATH": "/bin", "K": "v"}
        assert (request.stdin, request.timeout) == ("s", 3)

    def test_spawn_goes_through_wrap_too(self) -> None:
        inner, confinement, executor = self._confined()
        executor.spawn(SpawnRequest(argv=["codex", "app-server"], cwd="/w", env={"A": "1"}))
        (request,) = inner.spawns
        assert list(request.argv) == ["fake-confine", "codex", "app-server"]
        assert request.cwd is None
        assert dict(request.env) == {"A": "1", "K": "v"}
        assert confinement.wraps == [(("codex", "app-server"), "/w")]

    def test_the_binary_is_trusted_by_name_and_checked_inside(self) -> None:
        inner, confinement, executor = self._confined()
        assert executor.find_binary("codex", {}) == "codex"
        executor.check_binary("codex", {}, timeout=5)
        assert list(inner.requests[0].argv) == ["fake-confine", "codex", "--help"]
        assert confinement.wraps == [(("codex", "--help"), None)]

    def test_a_host_lookup_is_never_made(self) -> None:
        inner = FakeExecutor(FakeRun(), binaries={"other": "/x"})
        executor = confine(inner, FakeConfinement())
        assert executor.find_binary("codex", {}) == "codex"
        with pytest.raises(CliNotFoundError):
            inner.find_binary("codex", {})


class TestTransformingSpawn:
    def test_the_transform_applies_to_spawn(self) -> None:
        inner = FakeExecutor([], peers=lambda _r: EchoPeer())
        executor = TransformingExecutor(
            inner, lambda request: CommandRequest(["env", *request.argv], None, "/c", {}, None)
        )
        executor.spawn(SpawnRequest(argv=["x"], cwd="/h", env={"A": "1"}))
        (request,) = inner.spawns
        assert (list(request.argv), request.cwd, dict(request.env)) == (["env", "x"], "/c", {})


@pytest.mark.skipif(not Path("/proc/self/environ").exists(), reason="needs /proc")
def test_the_script_fails_loudly_when_the_container_lacks_its_tools() -> None:
    runner = FakeExecutor(FakeRun())
    _docker(runner).reap()
    script = runner.requests[0].argv[5]
    result = subprocess.run(  # noqa: S603
        [shutil.which("sh") or "sh", "-c", script],
        env={"PATH": "/nonexistent"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 127
