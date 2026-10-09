"""Provider names are data owned by ``agentshim/providers/``.

Outside that package (and the test doubles) no string constant may equal a
provider name: code that branches on one is a provider-specific special case
that belongs behind the provider's profile or protocol.
"""

from __future__ import annotations

import ast
from pathlib import Path

import agentshim
from agentshim import provider_names

_ROOT = Path(agentshim.__file__).parent
_EXEMPT_DIRS = ("providers", "testing")

#: ``(path relative to agentshim/, literal)`` of every existing violation, each
#: with why it exists. Keyed by literal rather than line so an unrelated edit
#: above the site does not break the test. Remove an entry when the site stops branching on the name;
#: never add one without a justification.
ALLOWLIST: dict[tuple[str, str], str] = {
    ("core/pricing.py", "codex"): "PROVIDER_VENDORS maps provider name to billing vendor",
    ("core/pricing.py", "claude"): "PROVIDER_VENDORS maps provider name to billing vendor",
}


def _violations() -> set[tuple[str, str]]:
    names = set(provider_names())
    found: set[tuple[str, str]] = set()
    for path in sorted(_ROOT.rglob("*.py")):
        relative = path.relative_to(_ROOT)
        if relative.parts[0] in _EXEMPT_DIRS:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in names
            ):
                found.add((relative.as_posix(), node.value))
    return found


def test_no_provider_name_literal_outside_the_providers_package() -> None:
    unexpected = sorted(_violations() - ALLOWLIST.keys())
    assert unexpected == []


def test_every_allowlisted_site_still_violates() -> None:
    stale = [site for site in ALLOWLIST if site not in _violations()]
    assert stale == []
