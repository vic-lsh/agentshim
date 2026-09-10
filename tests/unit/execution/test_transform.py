"""``TransformingExecutor`` rewrites every request, health check included."""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from agentshim import (
    CliCheckError,
    CliNotFoundError,
    CommandRequest,
    HostCommandExecutor,
    NullSink,
    TransformingExecutor,
)
from agentshim.testing import FakeExecutor, FakeRun


def _wrap(request: CommandRequest) -> CommandRequest:
    return replace(request, argv=["/usr/bin/env", "AGENTSHIM_WRAPPED=1", *request.argv])


class TestRun:
    def test_argv_is_rewritten_before_the_inner_executor_sees_it(self) -> None:
        inner = FakeExecutor(FakeRun(stdout=["line\n"]))
        executor = TransformingExecutor(inner, _wrap)

        executor.run(
            CommandRequest(argv=["claude", "-p"], stdin=None, cwd=None, env={}, timeout=None),
            NullSink(),
        )

        assert list(inner.requests[0].argv) == ["/usr/bin/env", "AGENTSHIM_WRAPPED=1", "claude", "-p"]

    def test_the_transform_can_change_any_field(self) -> None:
        inner = FakeExecutor(FakeRun())
        executor = TransformingExecutor(inner, lambda request: replace(request, cwd="/workspace"))

        executor.run(
            CommandRequest(argv=["claude"], stdin="hi", cwd=None, env={}, timeout=None),
            NullSink(),
        )

        assert inner.requests[0].cwd == "/workspace"
        assert inner.requests[0].stdin == "hi"


class TestFindBinary:
    def test_lookup_delegates_to_the_inner_executor_by_default(self) -> None:
        inner = FakeExecutor(FakeRun(), binaries={"claude": "/inner/claude"})
        assert TransformingExecutor(inner, _wrap).find_binary("claude", {}) == "/inner/claude"

    def test_a_custom_lookup_wins(self) -> None:
        inner = FakeExecutor(FakeRun(), binaries={"claude": "/inner/claude"})
        executor = TransformingExecutor(inner, _wrap, find_binary=lambda name, env: f"/container/bin/{name}")
        assert executor.find_binary("claude", {}) == "/container/bin/claude"

    def test_inner_lookup_failures_propagate(self) -> None:
        inner = FakeExecutor(FakeRun(), binaries={"other": "/inner/other"})
        with pytest.raises(CliNotFoundError):
            TransformingExecutor(inner, _wrap).find_binary("claude", {})


class TestCheckBinary:
    def test_the_health_check_goes_through_the_transform(self) -> None:
        inner = FakeExecutor(FakeRun())
        TransformingExecutor(inner, _wrap).check_binary("/bin/claude", {}, timeout=5)

        assert list(inner.requests[0].argv) == ["/usr/bin/env", "AGENTSHIM_WRAPPED=1", "/bin/claude", "--help"]
        assert inner.requests[0].timeout == 5

    def test_a_failing_health_check_raises(self) -> None:
        inner = FakeExecutor(FakeRun(returncode=2, stderr=["nope\n"]))
        with pytest.raises(CliCheckError, match="exited with code 2"):
            TransformingExecutor(inner, _wrap).check_binary("/bin/claude", {}, timeout=5)

    def test_a_timing_out_health_check_raises(self) -> None:
        inner = FakeExecutor(FakeRun(timeout=True))
        with pytest.raises(CliCheckError, match="did not respond"):
            TransformingExecutor(inner, _wrap).check_binary("/bin/claude", {}, timeout=5)

    def test_a_transform_makes_an_otherwise_unrunnable_binary_checkable(self, tmp_path: Path) -> None:
        """The real point: the binary only exists behind the transform."""
        target = tmp_path / "claude"
        target.write_text("#!/bin/sh\nexit 0\n")
        target.chmod(0o755)

        def redirect(request: CommandRequest) -> CommandRequest:
            return replace(request, argv=[str(target), *list(request.argv)[1:]])

        executor = TransformingExecutor(HostCommandExecutor(), redirect)
        executor.check_binary("/does/not/exist/claude", os.environ.copy(), timeout=10)

    def test_transform_applies_when_running_a_real_command(self, tmp_path: Path) -> None:
        marker = tmp_path / "marker"

        def redirect(request: CommandRequest) -> CommandRequest:
            code = f"open({str(marker)!r}, 'w').write('ran')"
            return replace(request, argv=[sys.executable, "-c", code])

        executor = TransformingExecutor(HostCommandExecutor(), redirect)
        result = executor.run(
            CommandRequest(argv=["claude"], stdin=None, cwd=None, env=os.environ.copy(), timeout=30),
            NullSink(),
        )

        assert result.returncode == 0
        assert marker.read_text() == "ran"
