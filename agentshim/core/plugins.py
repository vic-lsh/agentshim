"""Explicit plugins and the caller-owned state of an isolated session."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class PluginRef:
    """One named plugin from a local marketplace checkout.

    Remote sources are deliberately absent: the caller supplies the checkout
    whose marketplace manifest is to be installed by the provider CLI.
    """

    marketplace: Path
    name: str

    def __post_init__(self) -> None:
        """Reject ambiguous paths and plugin selectors."""
        if not self.marketplace.is_absolute():
            msg = "plugin marketplace must be an absolute path"
            raise ValueError(msg)
        if (
            not self.name
            or not self.name[0].isalnum()
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in self.name)
        ):
            msg = f"invalid plugin name: {self.name!r}"
            raise ValueError(msg)


@dataclass(frozen=True)
class SessionPlugins:
    """Exactly these plugins, plus CLI built-ins, in a dedicated session home.

    Passing this option excludes the operator's plugins, skills and config,
    regardless of the session's scopes. An empty ``refs`` selects no plugins.
    Omitting the option preserves the provider's normal scope behavior.
    ``home`` must be fresh on first use, belongs to the caller, and must be
    retained unchanged for resume. Never share it between sessions.
    """

    home: Path
    refs: tuple[PluginRef, ...] = ()

    def __post_init__(self) -> None:
        """Require a relocatable, immutable session specification."""
        if not self.home.is_absolute():
            msg = "plugin session home must be an absolute path"
            raise ValueError(msg)
        object.__setattr__(self, "refs", tuple(self.refs))
