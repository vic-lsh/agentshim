"""Hypothesis profiles shared by every suite.

``default`` keeps a normal ``pytest`` run fast. ``fuzz`` is for hunting: it
runs far more examples per property and has no deadline, so a slow shrink is
not reported as a failure. Select it with ``HYPOTHESIS_PROFILE=fuzz``.
"""

from __future__ import annotations

import os

from hypothesis import HealthCheck, settings

settings.register_profile("default", max_examples=200, deadline=None)
settings.register_profile(
    "fuzz",
    max_examples=20_000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))
