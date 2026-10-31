"""Recorded cost does not move: what agentshim charges is pinned to a checked-in golden.

``tests/fixtures/recorded_costs.json`` was recorded with agentshim 0.23.1. A
turn's cost feeds replay and budgets in callers, so changing how the pinned
table is produced, or adding any other source of rates, must leave every number
below exactly as it was. A deliberate price change regenerates the file with
``python scripts/record_cost_golden.py``; its diff is the list of costs that moved.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "record_cost_golden.py"
GOLDEN = REPO / "tests" / "fixtures" / "recorded_costs.json"


@pytest.fixture(scope="module")
def recorder() -> ModuleType:
    """The recorder script, imported by path (``scripts/`` is not a package)."""
    name = "record_cost_golden"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


@pytest.fixture(scope="module")
def current(recorder: ModuleType) -> dict[str, list[dict[str, object]]]:
    return json.loads(recorder.render(recorder.build()))


@pytest.fixture(scope="module")
def golden() -> dict[str, list[dict[str, object]]]:
    return json.loads(GOLDEN.read_text())


@pytest.mark.parametrize("section", ["usages", "rates", "costs", "streams"])
def test_section_equals_the_recorded_one(
    section: str,
    current: dict[str, list[dict[str, object]]],
    golden: dict[str, list[dict[str, object]]],
) -> None:
    assert current[section] == golden[section]


def test_the_golden_has_something_to_hold() -> None:
    document = json.loads(GOLDEN.read_text())
    assert len(document["rates"]) >= 29
    assert len(document["streams"]) >= 6
    assert any(row["costs"] is not None for row in document["costs"])
    # The recorded streams include a two-model Claude turn and a one-hour cache write.
    assert any(stream["tokens"]["cache_write_1h_input_tokens"] for stream in document["streams"])


def test_the_file_is_what_the_script_writes(recorder: ModuleType) -> None:
    assert GOLDEN.read_text() == recorder.render(recorder.build())
