"""``interactive_env`` capture, cache, and fallbacks."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from types import SimpleNamespace

import pytest
from agentshim import interactive_env
from agentshim.core import env as env_module
from hypothesis import given
from hypothesis import strategies as st


class _Clock:
    """The probe's monotonic clock, which only the shell's time advances."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


@pytest.fixture(autouse=True)
def _clear_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_module, "_cache", {})


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    """A virtual clock in place of the module's ``time``: no test waits for the budget."""
    virtual = _Clock()
    monkeypatch.setattr(env_module, "time", SimpleNamespace(monotonic=virtual.monotonic))
    return virtual


def _completed(
    stdout: str = "", *, returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    """Build what ``subprocess.run`` returns for the ``bash -i -c env`` probe."""
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class _FakeRun:
    """Stand-in for ``subprocess.run`` as ``env._probe`` calls it.

    Replays *outcomes* one per call, repeating the last once exhausted, and
    records what the probe asked for. Each call takes ``elapsed`` seconds of the
    virtual clock; a shell that times out takes the whole timeout it was given.
    """

    def __init__(
        self,
        *outcomes: subprocess.CompletedProcess[str] | Exception,
        clock: _Clock | None = None,
        elapsed: float = 0.0,
    ) -> None:
        self._outcomes = list(outcomes)
        self._clock = clock
        self._elapsed = elapsed
        self.calls = 0
        self.timeouts: list[float] = []
        self.commands: list[str] = []

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        **options: bool,
    ) -> subprocess.CompletedProcess[str]:
        # `options` absorbs capture_output/text/check/start_new_session.
        del options
        self.commands.append(argv[-1])
        self.calls += 1
        self.timeouts.append(timeout)
        outcome = self._outcomes[min(self.calls - 1, len(self._outcomes) - 1)]
        if self._clock is not None:
            hung = isinstance(outcome, subprocess.TimeoutExpired)
            # A real shell is killed at its timeout, so no call outlasts the one it was given.
            self._clock.now += timeout if hung else min(self._elapsed, timeout)
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


def test_a_hanging_shell_falls_back_to_os_environ(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    probe = _FakeRun(subprocess.TimeoutExpired(cmd="bash", timeout=10.0), clock=clock)
    monkeypatch.setattr(env_module.subprocess, "run", probe)
    assert interactive_env() == os.environ.copy()


def test_a_hanging_shell_is_asked_once_not_twice(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    """A shell that used the whole budget leaves nothing for the plain-``env`` retry.

    Regression: the retry had a timeout of its own, so a hung shell held a caller for twice the
    timeout, longer than the limit of a host read that probes the machine.
    """
    probe = _FakeRun(subprocess.TimeoutExpired(cmd="bash", timeout=10.0), clock=clock)
    monkeypatch.setattr(env_module.subprocess, "run", probe)
    interactive_env()
    assert probe.commands == ["env -0"]
    assert clock.now - 1000.0 == 10.0


@given(first=st.floats(min_value=0.0, max_value=30.0), second_hangs=st.booleans())
def test_the_two_attempts_never_take_longer_than_the_budget(
    first: float, *, second_hangs: bool
) -> None:
    """Whatever the first attempt spent, the two together never take longer than the budget.

    The first attempt fails after ``first`` seconds (a shell without ``env -0``); the second is
    granted what is left, and when nothing is left it is not made.
    """
    clock = _Clock()
    probe = _FakeRun(
        _completed(returncode=1),
        subprocess.TimeoutExpired(cmd="bash", timeout=0.0) if second_hangs else _completed("A=1\n"),
        clock=clock,
        elapsed=first,
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(env_module, "_cache", {})
        patch.setattr(env_module, "time", SimpleNamespace(monotonic=clock.monotonic))
        patch.setattr(env_module.subprocess, "run", probe)
        interactive_env()
    assert clock.now - 1000.0 <= 10.0
    assert probe.timeouts[0] == 10.0
    assert (len(probe.timeouts) == 2) == (first < 10.0)
    if len(probe.timeouts) == 2:
        assert probe.timeouts[1] == pytest.approx(10.0 - first)


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


def test_the_probe_asks_for_nul_separated_output(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _FakeRun(_completed("A=1\0"))
    monkeypatch.setattr(env_module.subprocess, "run", probe)
    assert interactive_env() == {"A": "1"}
    assert probe.commands == ["env -0"]


def test_a_multi_line_value_is_kept_whole(monkeypatch: pytest.MonkeyPatch) -> None:
    """Splitting on newlines truncated a key and turned its tail into junk keys."""
    key = "-----BEGIN KEY-----\nAAAA\n-----END KEY-----"
    probe = _FakeRun(_completed(f"SSH_KEY={key}\0X=1\0"))
    monkeypatch.setattr(env_module.subprocess, "run", probe)
    assert interactive_env() == {"SSH_KEY": key, "X": "1"}


def test_a_shell_without_env_zero_falls_back_to_plain_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _FakeRun(
        _completed(returncode=1, stderr="env: illegal option -- 0"),
        _completed("A=1\nB=2\n"),
    )
    monkeypatch.setattr(env_module.subprocess, "run", probe)
    assert interactive_env() == {"A": "1", "B": "2"}
    assert probe.commands == ["env -0", "env"]
