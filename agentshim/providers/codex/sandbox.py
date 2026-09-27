"""Codex's OS sandbox for the commands the model runs.

Codex confines model-generated shell commands itself (Landlock and seccomp on
Linux, Seatbelt on macOS). By default agentshim turns that off with
``--dangerously-bypass-approvals-and-sandbox``, which suits a caller that
already isolates the whole process. ``CodexSandboxConfig`` keeps the CLI's own
sandbox on instead.

Every setting is rendered as a ``--config`` override rather than ``--sandbox``,
because ``codex exec resume`` accepts ``--config`` but not ``--sandbox``: one
rendering serves fresh and resumed turns alike. For ``workspace-write`` every
key is emitted explicitly, even at its default, so a user's
``~/.codex/config.toml`` cannot widen the sandbox a caller asked for.

See https://developers.openai.com/codex/security.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, cast, get_args

from ._toml import toml_array, toml_bool, toml_str

SandboxMode = Literal["read-only", "workspace-write", "danger-full-access"]

#: Every mode Codex accepts, in its own spelling.
SANDBOX_MODES: tuple[SandboxMode, ...] = get_args(SandboxMode)

_WORKSPACE_WRITE: SandboxMode = "workspace-write"


def _roots() -> tuple[str, ...]:
    return ()


@dataclass(frozen=True)
class CodexSandboxConfig:
    """How Codex sandboxes the commands the model runs.

    Attributes:
        mode: ``read-only`` lets commands read but not write anywhere and
            blocks the network. ``workspace-write`` also lets them write the
            working directory, ``writable_roots`` and (by default) the temp
            directories. ``danger-full-access`` removes the sandbox but, unlike
            the bypass flag, keeps Codex's approval policy in force.
        writable_roots: Extra absolute directories commands may write.
            ``workspace-write`` only. Any sequence of strings is accepted and
            stored as a tuple, so equal configs compare and hash equal.
        network_access: Let commands open network sockets, including local
            ones such as the Docker socket. ``workspace-write`` only.
        writable_tmp: Keep ``/tmp`` and ``$TMPDIR`` writable, which is Codex's
            own default. ``workspace-write`` only.
    """

    mode: SandboxMode = _WORKSPACE_WRITE
    writable_roots: Sequence[str] = field(default_factory=_roots)
    network_access: bool = False
    writable_tmp: bool = True

    def __post_init__(self) -> None:
        """Reject a config Codex would reject or silently ignore."""
        if self.mode not in SANDBOX_MODES:
            msg = f"mode must be one of {', '.join(SANDBOX_MODES)}; got {self.mode!r}"
            raise ValueError(msg)
        object.__setattr__(self, "writable_roots", _as_roots(self.writable_roots))
        for name in ("network_access", "writable_tmp"):
            value: object = getattr(self, name)
            if not isinstance(value, bool):
                msg = f"{name} must be a bool, got {type(value).__name__}"
                raise TypeError(msg)
        if self.mode != _WORKSPACE_WRITE and self._widens_workspace_write():
            msg = (
                "writable_roots, network_access and writable_tmp only apply to "
                f"workspace-write; {self.mode} would ignore them"
            )
            raise ValueError(msg)

    def _widens_workspace_write(self) -> bool:
        return bool(self.writable_roots) or self.network_access or not self.writable_tmp


def _as_roots(value: object) -> tuple[str, ...]:
    """Validate ``writable_roots`` and freeze it into a tuple.

    A bare string is rejected rather than iterated, which would turn
    ``"/data"`` into the roots ``"/"``, ``"d"``, ``"a"``, ... A set is
    rejected too: its order is arbitrary, so the rendered argv would be.
    """
    if isinstance(value, str) or not isinstance(value, Sequence):
        msg = f"writable_roots must be a sequence of paths, not {type(value).__name__}"
        raise TypeError(msg)
    roots = tuple(cast("Sequence[object]", value))
    for root in roots:
        _check_root(root)
    return cast("tuple[str, ...]", roots)


def _check_root(root: object) -> None:
    if not isinstance(root, str):
        msg = f"writable_roots entries must be str, got {type(root).__name__}"
        raise TypeError(msg)
    if "\x00" in root:
        msg = f"writable_roots entry contains a NUL byte: {root!r}"
        raise ValueError(msg)
    if not _is_utf8(root):
        msg = f"writable_roots entry is not valid UTF-8 text: {root!r}"
        raise ValueError(msg)
    if not os.path.isabs(root):  # noqa: PTH117 - a str contract, not a Path
        msg = f"writable_roots entries must be absolute; got {root!r}"
        raise ValueError(msg)


def _is_utf8(text: str) -> bool:
    """TOML and argv both need text that encodes; a lone surrogate does not."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def resolve_sandbox(value: object) -> CodexSandboxConfig | None:
    """Normalize the ``sandbox`` provider option to a config or ``None``.

    ``None`` keeps agentshim's historical behaviour: Codex's sandbox and
    approvals are bypassed. There is deliberately no ``True`` shorthand,
    because no one mode is an obvious default for every caller.
    """
    if value is None or isinstance(value, CodexSandboxConfig):
        return value
    msg = f"sandbox must be a CodexSandboxConfig or None, got {type(value).__name__}"
    raise TypeError(msg)


def sandbox_overrides(config: CodexSandboxConfig) -> list[tuple[str, str]]:
    """Return the ``(key, TOML value)`` pairs that impose *config*.

    ``approval_policy = "never"`` rides along because a headless turn has no
    one to approve anything: without it a command the sandbox refuses would
    wait for an answer that never comes.
    """
    pairs = [
        ("sandbox_mode", toml_str(config.mode)),
        ("approval_policy", toml_str("never")),
    ]
    if config.mode == _WORKSPACE_WRITE:
        prefix = "sandbox_workspace_write"
        excluded = toml_bool(not config.writable_tmp)
        pairs += [
            (f"{prefix}.writable_roots", toml_array(config.writable_roots)),
            (f"{prefix}.network_access", toml_bool(config.network_access)),
            (f"{prefix}.exclude_slash_tmp", excluded),
            (f"{prefix}.exclude_tmpdir_env_var", excluded),
        ]
    return pairs
