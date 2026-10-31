"""``interactive_env`` capture, cache, and fallbacks."""

from __future__ import annotations

import math
import os
import subprocess
from collections.abc import Sequence

import pytest
from agentshim import interactive_env
from agentshim.core import env as env_module
from hypothesis import given
from hypothesis import strategies as st

ATTEMPT_S = 10.0
"""The time one attempt of the capture has, as the module declares it."""


class _Clock:
    """A virtual clock only the shell's time advances: no test waits for a timeout."""

    def __init__(self) -> None:
        self.now = 0.0


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
    records what the probe asked for. With a *clock*, call ``n`` takes
    ``seconds[n]`` of the virtual clock (zero when *seconds* is empty), and a shell
    that would take longer than the timeout it was given is killed at it, as a
    real one is: the clock advances by the timeout and the call raises
    ``TimeoutExpired``. A stalled shell is one that takes ``math.inf``.
    """

    def __init__(
        self,
        *outcomes: subprocess.CompletedProcess[str] | Exception,
        clock: _Clock | None = None,
        seconds: Sequence[float] = (),
    ) -> None:
        self._outcomes = list(outcomes)
        self._clock = clock
        self._seconds = list(seconds)
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
        takes = self._seconds[min(self.calls - 1, len(self._seconds) - 1)] if self._seconds else 0.0
        if self._clock is not None:
            self._clock.now += min(takes, timeout)
            if takes > timeout:
                raise subprocess.TimeoutExpired(cmd="bash", timeout=timeout)
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


def test_each_attempt_is_bounded_by_its_own_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _FakeRun(_completed(returncode=1), _completed("A=1\n"))
    monkeypatch.setattr(env_module.subprocess, "run", probe)
    interactive_env()
    assert probe.timeouts == [ATTEMPT_S, ATTEMPT_S]


def test_a_hanging_shell_falls_back_to_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock()
    probe = _FakeRun(_completed("A=1\n"), clock=clock, seconds=[math.inf])
    monkeypatch.setattr(env_module.subprocess, "run", probe)
    assert interactive_env() == os.environ.copy()
    assert clock.now == 2 * ATTEMPT_S


def test_a_stalled_first_attempt_leaves_the_retry_its_own_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``env -0`` stalls past its time and plain ``env`` answers in 3 seconds: captured.

    Regression (agentshim 0.24.3): both attempts shared one 10 second budget, so the stall
    left the retry nothing and a shell that was only slow to start was reported as having no
    CLIs on its PATH.
    """
    clock = _Clock()
    probe = _FakeRun(
        _completed("never read\n"), _completed("A=1\n"), clock=clock, seconds=[math.inf, 3.0]
    )
    monkeypatch.setattr(env_module.subprocess, "run", probe)
    assert interactive_env() == {"A": "1"}
    assert probe.commands == ["env -0", "env"]
    assert probe.timeouts == [ATTEMPT_S, ATTEMPT_S]
    assert clock.now == ATTEMPT_S + 3.0


def test_a_late_failure_of_env_zero_leaves_the_retry_its_own_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``env -0`` is unsupported and says so after 6 seconds; plain ``env`` takes 6: captured.

    Regression (agentshim 0.24.3): the retry got the 4 seconds the first attempt left.
    """
    clock = _Clock()
    probe = _FakeRun(
        _completed(returncode=1, stderr="env: illegal option -- 0"),
        _completed("A=1\n"),
        clock=clock,
        seconds=[6.0, 6.0],
    )
    monkeypatch.setattr(env_module.subprocess, "run", probe)
    assert interactive_env() == {"A": "1"}
    assert probe.timeouts == [ATTEMPT_S, ATTEMPT_S]
    assert clock.now == 12.0


@given(
    first=st.floats(min_value=0.0, max_value=3 * ATTEMPT_S) | st.just(math.inf),
    second=st.floats(min_value=0.0, max_value=3 * ATTEMPT_S) | st.just(math.inf),
)
def test_the_retry_is_captured_exactly_when_it_answers_within_its_own_timeout(
    first: float, second: float
) -> None:
    """Whatever the first attempt spent or how it failed, the second is judged on its time alone.

    The first attempt fails (a shell without ``env -0``, or one that stalled); the second answers
    after ``second`` seconds. It is captured when that fits its timeout, however long the first
    took, and the two together never take longer than two attempts' timeouts.
    """
    clock = _Clock()
    probe = _FakeRun(
        _completed(returncode=1), _completed("A=1\n"), clock=clock, seconds=[first, second]
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(env_module, "_cache", {})
        patch.setattr(env_module.subprocess, "run", probe)
        captured = interactive_env()
    assert (captured == {"A": "1"}) == (second <= ATTEMPT_S)
    assert probe.timeouts == [ATTEMPT_S, ATTEMPT_S]
    assert clock.now <= 2 * ATTEMPT_S


def test_a_failed_capture_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """A capture that failed once is asked for again, not kept for the life of the process.

    Regression (agentshim 0.24.3): the ``os.environ`` fallback was cached like an answer, so one
    stall in a loaded start-up reported no CLIs until the process restarted.
    """
    probe = _FakeRun(
        _completed(returncode=1),
        _completed(returncode=1),
        _completed("PATH=/opt/cli\n"),
    )
    monkeypatch.setattr(env_module.subprocess, "run", probe)
    assert interactive_env() == os.environ.copy()
    assert interactive_env() == {"PATH": "/opt/cli"}
    assert interactive_env() == {"PATH": "/opt/cli"}
    assert probe.calls == 3


@given(failures=st.integers(min_value=1, max_value=5))
def test_a_capture_is_retried_after_any_number_of_failures(failures: int) -> None:
    """However many calls fail first, the first call that can capture does, and is then cached."""
    probe = _FakeRun(*[_completed(returncode=1)] * (2 * failures), _completed("A=1\n"))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(env_module, "_cache", {})
        patch.setattr(env_module.subprocess, "run", probe)
        failed = [interactive_env() for _ in range(failures)]
        captured = [interactive_env() for _ in range(2)]
    assert failed == [os.environ.copy()] * failures
    assert captured == [{"A": "1"}] * 2
    assert probe.calls == 2 * failures + 1


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
