"""``interactive_env`` capture, cache, and fallbacks."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence

import pytest
from agentshim import interactive_env
from agentshim.core import env as env_module


@pytest.fixture(autouse=True)
def _clear_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_module, "_cache", {})


def _completed(
    stdout: str = "", *, returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    """Build what ``subprocess.run`` returns for the ``bash -i -c env`` probe."""
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class _FakeRun:
    """Stand-in for ``subprocess.run`` as ``env._probe`` calls it.

    Replays *outcomes* one per call, repeating the last once exhausted, and
    records what the probe asked for.
    """

    def __init__(self, *outcomes: subprocess.CompletedProcess[str] | Exception) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0
        self.timeouts: list[float] = []

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        **options: bool,
    ) -> subprocess.CompletedProcess[str]:
        # `options` absorbs capture_output/text/check/start_new_session; only
        # argv-independent behaviour and the timeout matter here.
        del argv, options
        self.calls += 1
        self.timeouts.append(timeout)
        outcome = self._outcomes[min(self.calls - 1, len(self._outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_a_real_shell_capture_has_a_path() -> None:
    env = interactive_env()
    assert "PATH" in env


def test_the_result_is_cached_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _FakeRun(_completed("A=1\n"))
    monkeypatch.setattr(env_module.subprocess, "run", probe)
    assert interactive_env() == {"A": "1"}
    assert interactive_env() == {"A": "1"}
    assert probe.calls == 1


def test_refresh_reprobes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        env_module.subprocess, "run", _FakeRun(_completed("A=1\n"), _completed("A=2\n"))
    )
    assert interactive_env() == {"A": "1"}
    assert interactive_env(refresh=True) == {"A": "2"}


def test_callers_cannot_mutate_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_module.subprocess, "run", _FakeRun(_completed("A=1\n")))
    first = interactive_env()
    first["A"] = "mutated"
    assert interactive_env()["A"] == "1"


def test_the_probe_is_bounded_by_a_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _FakeRun(_completed("A=1\n"))
    monkeypatch.setattr(env_module.subprocess, "run", probe)
    interactive_env()
    assert probe.timeouts == [10.0]


def test_a_hanging_shell_falls_back_to_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _FakeRun(subprocess.TimeoutExpired(cmd="bash", timeout=10.0))
    monkeypatch.setattr(env_module.subprocess, "run", probe)
    assert interactive_env() == os.environ.copy()


def test_a_missing_shell_falls_back_to_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_module.subprocess, "run", _FakeRun(FileNotFoundError("/bin/bash")))
    assert interactive_env() == os.environ.copy()


def test_a_failing_shell_falls_back_to_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _FakeRun(_completed(returncode=1, stderr="boom"))
    monkeypatch.setattr(env_module.subprocess, "run", probe)
    assert interactive_env() == os.environ.copy()


def test_values_containing_equals_are_kept_whole(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _FakeRun(_completed("OPTS=a=b=c\nX=1\n"))
    monkeypatch.setattr(env_module.subprocess, "run", probe)
    assert interactive_env() == {"OPTS": "a=b=c", "X": "1"}
