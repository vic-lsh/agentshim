"""Hypothesis profiles shared by every suite.

``default`` keeps a normal ``pytest`` run fast. ``fuzz`` is for hunting: it
runs far more examples per property and has no deadline, so a slow shrink is
not reported as a failure. Select it with ``HYPOTHESIS_PROFILE=fuzz``. Its
example count keeps a full unit run under five minutes; raise it with
``HYPOTHESIS_MAX_EXAMPLES`` only for an explicitly requested long run.
"""

from __future__ import annotations

import os

from hypothesis import HealthCheck, settings

settings.register_profile("default", max_examples=200, deadline=None)
settings.register_profile(
    "fuzz",
    max_examples=int(os.environ.get("HYPOTHESIS_MAX_EXAMPLES", "1000")),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))
