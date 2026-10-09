"""How ``NativePermissions`` become the sandbox and approval settings Codex takes.

Codex inherits ``sandbox_mode`` and ``approval_policy`` from the user's
``config.toml`` unless a request names them, so the transport sends all of them
on ``thread/start``, ``thread/resume`` and every ``turn/start``, and checks what
the server says it applied.

- ``BYPASS``: sandbox ``danger-full-access``, approval ``never``, policy
  ``dangerFullAccess``.
- ``READ_ONLY``: sandbox ``read-only``, approval ``on-request``, policy
  ``readOnly`` (network off).
- ``WORKSPACE_WRITE``: sandbox ``workspace-write``, approval ``on-request``,
  policy ``workspaceWrite`` carrying the writable roots and the network flag.
  Only ``turn/start`` can carry those two, so every turn sends them.

A sandboxed mode asks with ``on-request`` rather than ``never``: the sandbox is
what stops a write, and the one thing the agent can still do is ask to go
beyond it, which is the request the caller's ``ApprovalPolicy`` answers. With
``never`` that request is never made, so the policy would have nothing to
decide and a refusal would be invisible. ``untrusted`` is the other
alternative; it asks before every command that is not on Codex's short list of
known-safe reads, so under ``DENY`` it would forbid ordinary work the sandbox
already makes safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from agentshim.core.errors import ProviderCapabilityError
from agentshim.core.permissions import NativeMode

from .protocol import (
    AskForApproval,
    AskForApprovalKind,
    SandboxMode,
    SandboxPolicy,
    SandboxPolicyDangerFullAccess,
    SandboxPolicyReadOnly,
    SandboxPolicyWorkspaceWrite,
)

if TYPE_CHECKING:
    from agentshim.core.permissions import NativePermissions


@dataclass(frozen=True)
class CodexPermissions:
    """The Codex settings one ``NativePermissions`` value stands for."""

    #: Coarse sandbox mode, the only form ``thread/start`` and ``thread/resume`` take.
    sandbox: SandboxMode
    approval: AskForApprovalKind
    #: The detailed policy ``turn/start`` takes; carries writable roots and network.
    policy: SandboxPolicy


def codex_permissions(permissions: NativePermissions) -> CodexPermissions:
    """Translate *permissions* into Codex's sandbox mode, approval policy and sandbox policy."""
    if permissions.mode is NativeMode.BYPASS:
        return CodexPermissions(
            SandboxMode.DANGER_FULL_ACCESS,
            AskForApprovalKind.NEVER,
            SandboxPolicyDangerFullAccess(),
        )
    if permissions.mode is NativeMode.READ_ONLY:
        return CodexPermissions(
            SandboxMode.READ_ONLY,
            AskForApprovalKind.ON_REQUEST,
            SandboxPolicyReadOnly(network_access=False),
        )
    return CodexPermissions(
        SandboxMode.WORKSPACE_WRITE,
        AskForApprovalKind.ON_REQUEST,
        SandboxPolicyWorkspaceWrite(
            writable_roots=permissions.writable_roots, network_access=permissions.network
        ),
    )


def check_applied(
    expected: CodexPermissions, sandbox: SandboxPolicy, approval: AskForApproval
) -> None:
    """Raise ``ProviderCapabilityError`` unless the server applied what was asked.

    Compares the policy type, the approval policy and, for ``readOnly``, that
    the network stayed off. The thread-level echo of ``workspaceWrite`` shows
    only the default roots (the coarse mode cannot carry them), so its roots
    and network flag are not compared.
    """
    problems: list[str] = []
    if type(sandbox) is not type(expected.policy):
        problems.append(f"sandbox {_describe(sandbox)} instead of {_describe(expected.policy)}")
    elif isinstance(sandbox, SandboxPolicyReadOnly) and sandbox.network_access:
        problems.append("read-only sandbox with network access")
    if approval != expected.approval:
        problems.append(f"approval policy {approval!r} instead of {expected.approval.value!r}")
    if problems:
        msg = "codex did not apply the requested permissions: " + "; ".join(problems)
        raise ProviderCapabilityError(msg)


def _describe(policy: SandboxPolicy) -> str:
    return type(policy).__name__.removeprefix("SandboxPolicy")
