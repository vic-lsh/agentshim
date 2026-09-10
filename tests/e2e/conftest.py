"""End-to-end tests run the real provider CLIs.

They are skipped unless ``AGENTSHIM_E2E=1`` and the binary is on PATH, so a
normal ``pytest`` run needs no credentials and makes no network calls.
"""

from __future__ import annotations

import os
import shutil

import pytest


def requires_cli(binary: str) -> pytest.MarkDecorator:
    """Skip the test unless e2e is enabled and *binary* is installed."""
    enabled = os.environ.get("AGENTSHIM_E2E") == "1"
    reason = "set AGENTSHIM_E2E=1" if not enabled else f"{binary} is not on PATH"
    return pytest.mark.skipif(not enabled or shutil.which(binary) is None, reason=reason)
