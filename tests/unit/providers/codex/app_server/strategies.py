"""Hypothesis strategies derived from the generated dataclasses' own type hints."""

from __future__ import annotations

import dataclasses
import enum
import types
import typing
from functools import cache
from typing import Any

from agentshim.providers.codex.app_server import protocol as p
from hypothesis import strategies as st

MAX_DEPTH = 6

json_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**53), max_value=2**53),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.text(max_size=8),
)
json_values: st.SearchStrategy[Any] = st.recursive(
    json_scalars,
    lambda inner: st.one_of(
        st.lists(inner, max_size=3), st.dictionaries(st.text(max_size=6), inner, max_size=3)
    ),
    max_leaves=8,
)
json_objects = st.dictionaries(st.text(max_size=6), json_values, max_size=4)
request_ids = st.one_of(st.text(min_size=1, max_size=8), st.integers(min_value=0, max_value=2**31))


def protocol_dataclasses() -> list[type]:
    """Every dataclass the generator emitted, without the envelopes and unknown variants."""
    skip = {
        "Notification",
        "ServerRequest",
        "ClientRequest",
        "ClientNotification",
        "UnknownParams",
    }
    found = [
        obj
        for name, obj in vars(p).items()
        if isinstance(obj, type)
        and dataclasses.is_dataclass(obj)
        and obj.__module__ == p.__name__
        and name not in skip
        and not name.startswith("Unknown")
    ]
    return sorted(found, key=lambda c: c.__name__)


@cache
def hints(cls: type) -> dict[str, Any]:
    return typing.get_type_hints(cls, dict(vars(p)))


def for_type(tp: Any, depth: int) -> st.SearchStrategy[Any]:  # noqa: C901, PLR0911
    """A strategy for values of the annotation ``tp``."""
    if isinstance(tp, typing.ForwardRef):
        return json_values
    if tp is type(None):
        return st.none()
    for scalar, strategy in (
        (bool, st.booleans()),
        (int, st.integers(min_value=-(2**53), max_value=2**53)),
        (float, st.floats(allow_nan=False, allow_infinity=False, width=32)),
        (str, st.text(max_size=8)),
    ):
        if tp is scalar:
            return strategy
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin is tuple:
        item = for_type(args[0], depth - 1)
        return st.lists(item, max_size=2 if depth > 0 else 0).map(tuple)
    if origin is dict:
        return st.dictionaries(
            st.text(max_size=4), for_type(args[1], depth - 1), max_size=2 if depth > 0 else 0
        )
    if origin is list:
        return st.lists(for_type(args[0], depth - 1), max_size=2 if depth > 0 else 0)
    if origin in (typing.Union, types.UnionType):
        members = [a for a in args if not getattr(a, "__name__", "").startswith("Unknown")]
        if depth <= 0:
            members = [
                m for m in members if m is type(None) or m in (str, int, bool, float)
            ] or members
        return st.one_of(*[for_type(m, depth) for m in members])
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        return st.sampled_from(list(tp))
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        return for_dataclass(tp, depth - 1)
    msg = f"no strategy for {tp!r}"
    raise TypeError(msg)


def for_dataclass(cls: type, depth: int = MAX_DEPTH) -> st.SearchStrategy[Any]:
    """A strategy for instances of the generated dataclass ``cls``."""
    if depth < -MAX_DEPTH:
        msg = f"{cls.__name__} cannot be built without unbounded recursion"
        raise RecursionError(msg)
    kwargs = {f.name: for_type(hints(cls)[f.name], depth) for f in dataclasses.fields(cls)}
    return st.builds(cls, **kwargs)
