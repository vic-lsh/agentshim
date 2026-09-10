"""``interactive_env`` capture, cache, and fallbacks."""

from __future__ import annotations

import os
import subprocess
from typing import Any

import pytest

from agentshim import interactive_env
from agentshim.core import env as env_module


@pytest.fixture(autouse=True)
def _clear_cache() -> Any:
    env_module._cached = None  # pyright: ignore[reportPrivateUsage]
    yield
    env_module._cached = None  # pyright: ignore[reportPrivateUsage]


def test_a_real_shell_capture_has_a_path() -> None:
    env = interactive_env()
    assert "PATH" in env


def test_the_result_is_cached_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def probe(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="A=1\n", stderr="")

    monkeypatch.setattr(env_module.subprocess, "run", probe)
    assert interactive_env() == {"A": "1"}
    assert interactive_env() == {"A": "1"}
    assert len(calls) == 1


def test_refresh_reprobes(monkeypatch: pytest.MonkeyPatch) -> None:
    values = iter(["A=1\n", "A=2\n"])

    def probe(*args: Any, **kwargs: Any) -> Any:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=next(values), stderr="")

    monkeypatch.setattr(env_module.subprocess, "run", probe)
    assert interactive_env() == {"A": "1"}
    assert interactive_env(refresh=True) == {"A": "2"}


def test_callers_cannot_mutate_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        env_module.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(args=[], returncode=0, stdout="A=1\n", stderr=""),
    )
    first = interactive_env()
    first["A"] = "mutated"
    assert interactive_env()["A"] == "1"


def test_the_probe_is_bounded_by_a_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[float] = []

    def probe(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs["timeout"])
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="A=1\n", stderr="")

    monkeypatch.setattr(env_module.subprocess, "run", probe)
    interactive_env()
    assert seen == [10.0]


def test_a_hanging_shell_falls_back_to_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    def probe(*args: Any, **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd="bash", timeout=10.0)

    monkeypatch.setattr(env_module.subprocess, "run", probe)
    assert interactive_env() == os.environ.copy()


def test_a_missing_shell_falls_back_to_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    def probe(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError("/bin/bash")

    monkeypatch.setattr(env_module.subprocess, "run", probe)
    assert interactive_env() == os.environ.copy()


def test_a_failing_shell_falls_back_to_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        env_module.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="boom"),
    )
    assert interactive_env() == os.environ.copy()


def test_values_containing_equals_are_kept_whole(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        env_module.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(args=[], returncode=0, stdout="OPTS=a=b=c\nX=1\n", stderr=""),
    )
    assert interactive_env() == {"OPTS": "a=b=c", "X": "1"}
