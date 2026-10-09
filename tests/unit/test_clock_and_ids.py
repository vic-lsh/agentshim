"""``Clock`` and ``IdAllocator`` implementations."""

from __future__ import annotations

from agentshim import Clock, RandomIds, StopSignal, SystemClock
from agentshim.testing import FakeClock, SequentialIds
from agentshim.testing.contracts import ClockContract
from hypothesis import given
from hypothesis import strategies as st


class TestSystemClock(ClockContract):
    def make_clock(self) -> Clock:
        return SystemClock()


class TestFakeClock(ClockContract):
    def make_clock(self) -> Clock:
        return FakeClock()


@given(st.lists(st.integers(min_value=0, max_value=10**6), max_size=20))
def test_fake_clock_time_is_the_sum_of_its_waits(waits: list[int]) -> None:
    clock = FakeClock()
    for seconds in waits:
        assert clock.wait(float(seconds)) is False
    assert clock.monotonic() == float(sum(waits))
    assert clock.waits == [float(w) for w in waits]


def test_fake_clock_starts_at_zero_and_a_set_stop_does_not_advance_it() -> None:
    clock = FakeClock()
    stop = StopSignal()
    stop.set()
    assert clock.monotonic() == 0.0
    assert clock.wait(30.0, stop) is True
    assert clock.monotonic() == 0.0
    assert clock.waits == [30.0]


def test_stop_signal_round_trips() -> None:
    stop = StopSignal()
    assert not stop.is_set()
    stop.set()
    stop.set()
    assert stop.is_set()


@given(st.lists(st.sampled_from(["turn", "req", "x"]), max_size=30))
def test_sequential_ids_count_per_prefix(prefixes: list[str]) -> None:
    ids = SequentialIds()
    seen: dict[str, int] = {}
    for prefix in prefixes:
        seen[prefix] = seen.get(prefix, 0) + 1
        assert ids.new_id(prefix) == f"{prefix}-{seen[prefix]}"


def test_random_ids_are_unique_and_keep_their_prefix() -> None:
    ids = RandomIds()
    made = {ids.new_id("turn") for _ in range(100)}
    assert len(made) == 100
    assert all(value.startswith("turn-") for value in made)
