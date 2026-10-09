"""Decoding and encoding helpers shared by the generated Codex protocol module.

The generated ``protocol.py`` keeps each type's ``from_wire``/``to_wire`` short
by delegating the mechanics to this module: a decoder is any callable
``(value, path) -> T`` that returns the typed value or raises
``CodexProtocolError`` naming the offending path. Decoders are tolerant by
construction: they read only the keys they know, so extra keys never fail.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from typing import TYPE_CHECKING, TypeVar, Union, cast

from agentshim.core.errors import AgentShimError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

from collections.abc import Callable

#: Any value ``json.loads`` can return.
JsonValue = Union[str, int, float, bool, None, list["JsonValue"], dict[str, "JsonValue"]]  # noqa: UP007
#: A JSON object.
JsonObject = dict[str, JsonValue]

T = TypeVar("T")
E = TypeVar("E", bound=Enum)

#: A decoder turns a wire value at ``path`` into a typed value.
Decoder = Callable[[object, str], T]


class CodexProtocolError(AgentShimError):
    """A Codex app-server message did not have the shape the protocol promises."""

    def __init__(self, path: str, problem: str) -> None:
        """Record where in the message the problem was found.

        Args:
            path: Dotted location of the offending value, rooted at the type name.
            problem: What was wrong there.
        """
        self.path = path
        self.problem = problem
        super().__init__(f"{path}: {problem}")


_JSON_TYPE_NAMES: tuple[tuple[type | tuple[type, ...], str], ...] = (
    (bool, "boolean"),
    ((int, float), "number"),
    (str, "string"),
    (list, "array"),
    (dict, "object"),
)


def describe(value: object) -> str:
    """Name the JSON type of ``value`` for an error message."""
    if value is None:
        return "null"
    for kind, name in _JSON_TYPE_NAMES:
        if isinstance(value, kind):
            return name
    return type(value).__name__


def wrong_type(path: str, expected: str, value: object) -> CodexProtocolError:
    """Build the error for a value whose JSON type is not ``expected``."""
    return CodexProtocolError(path, f"expected {expected}, got {describe(value)}")


def as_object(value: object, path: str) -> dict[str, object]:
    """Return ``value`` as a JSON object or raise."""
    if not isinstance(value, dict):
        raise wrong_type(path, "object", value)
    return {str(k): v for k, v in cast("dict[object, object]", value).items()}


def decode_str(value: object, path: str) -> str:
    """Decode a JSON string."""
    if not isinstance(value, str):
        raise wrong_type(path, "string", value)
    return value


def decode_bool(value: object, path: str) -> bool:
    """Decode a JSON boolean."""
    if not isinstance(value, bool):
        raise wrong_type(path, "boolean", value)
    return value


def decode_int(value: object, path: str) -> int:
    """Decode a JSON integer; ``true`` and ``1.5`` are not integers."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise wrong_type(path, "integer", value)
    return value


def decode_float(value: object, path: str) -> float:
    """Decode a JSON number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise wrong_type(path, "number", value)
    return float(value)


def decode_json(value: object, path: str) -> JsonValue:
    """Accept any JSON value, checking only that it is JSON."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, list):
        items = cast("list[object]", value)
        return [decode_json(v, f"{path}[{i}]") for i, v in enumerate(items)]
    if isinstance(value, dict):
        members = cast("dict[object, object]", value)
        return {str(k): decode_json(v, f"{path}.{k}") for k, v in members.items()}
    raise wrong_type(path, "JSON value", value)


def decode_json_object(value: object, path: str) -> JsonObject:
    """Accept any JSON object."""
    if not isinstance(value, dict):
        raise wrong_type(path, "object", value)
    members = cast("dict[object, object]", value)
    return {str(k): decode_json(v, f"{path}.{k}") for k, v in members.items()}


def nullable(decode: Decoder[T]) -> Decoder[T | None]:
    """Wrap ``decode`` so JSON ``null`` becomes ``None``."""

    def run(value: object, path: str) -> T | None:
        return None if value is None else decode(value, path)

    return run


def tuple_of(decode: Decoder[T]) -> Decoder[tuple[T, ...]]:
    """Decode a JSON array into a tuple of ``decode``d items."""

    def run(value: object, path: str) -> tuple[T, ...]:
        if not isinstance(value, list):
            raise wrong_type(path, "array", value)
        items = cast("list[object]", value)
        return tuple(decode(v, f"{path}[{i}]") for i, v in enumerate(items))

    return run


def dict_of(decode: Decoder[T]) -> Decoder[dict[str, T]]:
    """Decode a JSON object into a dict of ``decode``d values."""

    def run(value: object, path: str) -> dict[str, T]:
        if not isinstance(value, dict):
            raise wrong_type(path, "object", value)
        members = cast("dict[object, object]", value)
        return {str(k): decode(v, f"{path}.{k}") for k, v in members.items()}

    return run


def _attempt(decode: Decoder[T], value: object, path: str) -> T | CodexProtocolError:
    """Run ``decode``, returning its error instead of raising it."""
    try:
        return decode(value, path)
    except CodexProtocolError as exc:
        return exc


def first_of(*decoders: Decoder[T]) -> Decoder[T]:
    """Try each decoder in turn; fail with the last error if none accepts."""

    def run(value: object, path: str) -> T:
        error: CodexProtocolError | None = None
        for decode in decoders:
            result = _attempt(decode, value, path)
            if not isinstance(result, CodexProtocolError):
                return result
            error = result
        raise error or CodexProtocolError(path, "no alternative accepts the value")

    return run


def required(obj: Mapping[str, object], key: str, path: str, decode: Decoder[T]) -> T:
    """Decode ``obj[key]``, which the schema requires to be present."""
    if key not in obj:
        raise CodexProtocolError(path, f"missing required key {key!r}")
    return decode(obj[key], f"{path}.{key}")


def optional(obj: Mapping[str, object], key: str, path: str, decode: Decoder[T]) -> T | None:
    """Decode ``obj[key]``; an absent or ``null`` value is ``None``."""
    value = obj.get(key)
    return None if value is None else decode(value, f"{path}.{key}")


def with_default(
    obj: Mapping[str, object], key: str, path: str, decode: Decoder[T], *, default: T
) -> T:
    """Decode ``obj[key]``; an absent or ``null`` value is ``default``."""
    value = obj.get(key)
    return default if value is None else decode(value, f"{path}.{key}")


def not_listed(path: str, text: str, enum: type[Enum]) -> CodexProtocolError:
    """Build the error for a string the enum does not list."""
    allowed = ", ".join(repr(m.value) for m in enum)
    return CodexProtocolError(path, f"{text!r} is not one of {allowed}")


def decode_enum(enum: type[E], value: object, path: str) -> E:
    """Decode a string into ``enum``, rejecting a value the enum does not list.

    Used for types that only travel client to server, where the schema's values
    are the only valid ones.
    """
    text = decode_str(value, path)
    try:
        return enum(text)
    except ValueError:
        raise not_listed(path, text, enum) from None


def decode_enum_tolerant(enum: type[E], value: object, path: str) -> E | str:
    """Decode a string into ``enum``, keeping a value the enum does not list.

    Used for types that travel server to client: the server may add values
    between CLI versions, and the raw string must survive for the caller.
    """
    text = decode_str(value, path)
    try:
        return enum(text)
    except ValueError:
        return text


def enum_or_none(enum: type[E], value: str) -> E | None:
    """Return the member of ``enum`` for ``value``, or ``None`` if unlisted."""
    try:
        return enum(value)
    except ValueError:
        return None


def decode_tagged(
    value: object,
    path: str,
    *,
    key: str,
    variants: Mapping[str, Decoder[T]],
    unknown: Callable[[JsonObject], T] | None,
) -> T:
    """Decode an object whose ``key`` member selects the variant.

    ``unknown`` builds the catch-all variant for a tag no variant claims; when
    it is ``None`` an unclaimed tag is an error.
    """
    obj = as_object(value, path)
    tag = obj.get(key)
    if isinstance(tag, str) and tag in variants:
        return variants[tag](obj, path)
    if unknown is None:
        raise CodexProtocolError(path, f"unknown {key} {tag!r}")
    return unknown(decode_json_object(obj, path))


def _variant_for(value: object, variants: Mapping[str, Decoder[T]]) -> Decoder[T] | None:
    """The decoder whose key is present in the object ``value``, if any."""
    if not isinstance(value, dict):
        return None
    members = cast("dict[object, object]", value)
    return next((decode for key, decode in variants.items() if key in members), None)


def decode_external(
    value: object,
    path: str,
    *,
    units: Callable[[str], T | None],
    variants: Mapping[str, Decoder[T]],
    unknown: Callable[[JsonValue], T] | None,
) -> T:
    """Decode a union of bare strings and single-key objects.

    ``units`` maps a bare string to its member (or ``None`` if unlisted);
    ``variants`` is keyed by the single object key that names the variant.
    """
    if isinstance(value, str):
        unit = units(value)
        if unit is not None:
            return unit
    else:
        decode = _variant_for(value, variants)
        if decode is not None:
            return decode(value, path)
    if unknown is None:
        raise CodexProtocolError(path, f"unrecognized value of type {describe(value)}")
    return unknown(decode_json(value, path))


def decode_params(
    obj: Mapping[str, object],
    path: str,
    method: str,
    table: Mapping[str, Decoder[T]],
    fallback: Callable[[JsonValue], T],
) -> T:
    """Decode a message's ``params`` by its ``method``.

    A method missing from ``table`` is not an error: its params are kept
    verbatim through ``fallback`` so newer servers do not break older clients.
    """
    decode = table.get(method)
    if decode is None:
        return fallback(decode_json(obj.get("params"), f"{path}.params"))
    return required(obj, "params", path, decode)


def unencodable(value: object) -> str:
    """Describe why ``value`` cannot be encoded."""
    if is_dataclass(value):
        names = [f.name for f in fields(value)]
        return f"{type(value).__name__} has no to_wire (fields: {names})"
    return f"cannot encode {type(value).__name__} as JSON"


def fields_to_wire(
    required: Mapping[str, object], optional: Mapping[str, object]
) -> dict[str, JsonValue]:
    """Encode an object: every ``required`` member (``None`` as null), then the non-``None`` ``optional`` ones."""
    out = {key: encode(value) for key, value in required.items()}
    out.update({key: encode(value) for key, value in optional.items() if value is not None})
    return out


def encode(value: object) -> JsonValue:
    """Encode a generated value (dataclass, enum, tuple, dict, scalar) as JSON."""
    if isinstance(value, Enum):
        return encode(value.value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in cast("Sequence[object]", value)]
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in cast("dict[object, object]", value).items()}
    to_wire = getattr(value, "to_wire", None)
    if callable(to_wire):
        return encode(to_wire())
    raise TypeError(unencodable(value))
