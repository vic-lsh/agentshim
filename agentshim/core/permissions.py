"""What the agent's own sandbox may do, and what to do when it asks.

These are the *native* permissions: the agent CLI's built-in sandbox and
approval prompts. They are independent of any outside ``Confinement``.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import cast


class NativeMode(Enum):
    """Which sandbox the agent applies to the commands it runs."""

    #: The agent's own sandbox and approvals are off.
    BYPASS = "bypass"
    #: Commands may read but not write.
    READ_ONLY = "read_only"
    #: Commands may write the working directory and ``writable_roots``.
    WORKSPACE_WRITE = "workspace_write"


class ApprovalPolicy(Enum):
    """What a transport does when the agent asks for permission or user input.

    A long unattended run has no human to answer, so a transport never waits
    for one: it either answers at once or ends the turn.
    """

    #: Answer the request with a refusal and keep the turn going.
    DENY = "deny"
    #: End the turn with a failure.
    FAIL_TURN = "fail_turn"


@dataclass(frozen=True)
class NativePermissions:
    """The native sandbox a session asks of its agent.

    ``BYPASS`` disables the agent's own sandbox and approvals. It is only safe
    when an outside ``Confinement`` bounds the agent; on its own it gives the
    agent the full authority of the user running it.

    ``writable_roots`` (absolute paths) and ``network`` widen
    ``WORKSPACE_WRITE`` and are rejected with any other mode, so a value is
    never silently ignored.
    """

    mode: NativeMode
    writable_roots: tuple[str, ...] = ()
    network: bool = False

    def __post_init__(self) -> None:
        """Reject a combination the agent would ignore or misread."""
        if not isinstance(self.mode, NativeMode):  # pyright: ignore[reportUnnecessaryIsInstance]
            msg = f"mode must be a NativeMode, got {type(self.mode).__name__}"
            raise TypeError(msg)
        object.__setattr__(self, "writable_roots", _as_roots(self.writable_roots))
        if not isinstance(self.network, bool):  # pyright: ignore[reportUnnecessaryIsInstance]
            msg = f"network must be a bool, got {type(self.network).__name__}"
            raise TypeError(msg)
        if self.mode is not NativeMode.WORKSPACE_WRITE and (self.writable_roots or self.network):
            msg = (
                "writable_roots and network only apply to workspace_write; "
                f"{self.mode.value} would ignore them"
            )
            raise ValueError(msg)

    @classmethod
    def bypass(cls) -> NativePermissions:
        """No native sandbox. Only safe inside an outside ``Confinement``."""
        return cls(NativeMode.BYPASS)

    @classmethod
    def read_only(cls) -> NativePermissions:
        """Commands may read but not write."""
        return cls(NativeMode.READ_ONLY)

    @classmethod
    def workspace_write(
        cls, writable_roots: Sequence[str] = (), *, network: bool = False
    ) -> NativePermissions:
        """Commands may write the working directory and *writable_roots*."""
        return cls(NativeMode.WORKSPACE_WRITE, tuple(_as_roots(writable_roots)), network)


def _as_roots(value: object) -> tuple[str, ...]:
    """Validate ``writable_roots`` and freeze it into a tuple.

    A bare string is rejected rather than iterated, which would turn
    ``"/data"`` into the roots ``"/"``, ``"d"``, ``"a"``, ... A set is
    rejected too: its order is arbitrary.
    """
    if isinstance(value, str) or not isinstance(value, Sequence):
        msg = f"writable_roots must be a sequence of paths, not {type(value).__name__}"
        raise TypeError(msg)
    roots = tuple(cast("Sequence[object]", value))
    for root in roots:
        if not isinstance(root, str):
            msg = f"writable_roots entries must be str, got {type(root).__name__}"
            raise TypeError(msg)
        if "\x00" in root:
            msg = f"writable_roots entry contains a NUL byte: {root!r}"
            raise ValueError(msg)
        if not os.path.isabs(root):  # noqa: PTH117 - a str contract, not a Path
            msg = f"writable_roots entries must be absolute; got {root!r}"
            raise ValueError(msg)
    return cast("tuple[str, ...]", roots)
