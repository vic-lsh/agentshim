"""``NativePermissions`` validation and ``ProviderProfile`` mode declarations."""

from __future__ import annotations

import pytest
from agentshim import ApprovalPolicy, NativeMode, NativePermissions, get_provider, provider_names
from hypothesis import given
from hypothesis import strategies as st

_SEGMENT = st.text(alphabet="abcdefg_-.", min_size=1, max_size=6)
_ABS = st.lists(_SEGMENT, min_size=1, max_size=4).map(lambda parts: "/" + "/".join(parts))


def test_constructors_build_each_mode() -> None:
    assert NativePermissions.bypass() == NativePermissions(NativeMode.BYPASS)
    assert NativePermissions.read_only().mode is NativeMode.READ_ONLY
    wrote = NativePermissions.workspace_write(["/data"], network=True)
    assert (wrote.mode, wrote.writable_roots, wrote.network) == (
        NativeMode.WORKSPACE_WRITE,
        ("/data",),
        True,
    )


@given(roots=st.lists(_ABS, max_size=5), network=st.booleans())
def test_workspace_write_accepts_any_absolute_roots_and_freezes_them(
    roots: list[str], *, network: bool
) -> None:
    permissions = NativePermissions.workspace_write(roots, network=network)
    assert permissions.writable_roots == tuple(roots)
    assert isinstance(permissions.writable_roots, tuple)
    assert hash(permissions) == hash(NativePermissions.workspace_write(roots, network=network))


@given(st.lists(_ABS, min_size=1, max_size=3))
def test_roots_are_rejected_outside_workspace_write(roots: list[str]) -> None:
    for mode in (NativeMode.BYPASS, NativeMode.READ_ONLY):
        with pytest.raises(ValueError, match="workspace_write"):
            NativePermissions(mode, writable_roots=tuple(roots))


@pytest.mark.parametrize("mode", [NativeMode.BYPASS, NativeMode.READ_ONLY])
def test_network_is_rejected_outside_workspace_write(mode: NativeMode) -> None:
    with pytest.raises(ValueError, match="network"):
        NativePermissions(mode, network=True)


@given(st.text(max_size=8).filter(lambda text: "\x00" not in text and not text.startswith("/")))
def test_relative_roots_are_rejected(root: str) -> None:
    with pytest.raises(ValueError, match="absolute"):
        NativePermissions.workspace_write([root])


def test_a_nul_byte_in_a_root_is_rejected() -> None:
    with pytest.raises(ValueError, match="NUL"):
        NativePermissions.workspace_write(["/a\x00b"])


@pytest.mark.parametrize("bad", ["/data", {"/data"}, 5, None])
def test_roots_must_be_a_sequence_not_a_bare_string_or_set(bad: object) -> None:
    with pytest.raises(TypeError, match="writable_roots"):
        NativePermissions(NativeMode.WORKSPACE_WRITE, writable_roots=bad)  # type: ignore[arg-type]


def test_non_string_root_entries_are_rejected() -> None:
    with pytest.raises(TypeError, match="str"):
        NativePermissions.workspace_write([5])  # type: ignore[list-item]


def test_mode_must_be_a_native_mode() -> None:
    with pytest.raises(TypeError, match="NativeMode"):
        NativePermissions("bypass")  # type: ignore[arg-type]


def test_approval_policy_has_the_two_documented_values() -> None:
    assert {policy.name for policy in ApprovalPolicy} == {"DENY", "FAIL_TURN"}


@pytest.mark.parametrize("name", provider_names())
def test_every_provider_declares_at_least_bypass(name: str) -> None:
    modes = get_provider(name).profile.native_permission_modes
    assert NativeMode.BYPASS in modes
