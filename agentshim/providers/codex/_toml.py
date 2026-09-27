"""TOML literals for Codex's ``--config key=value`` overrides.

Codex parses the value of each override as TOML and, when that fails, uses
the raw text as a string instead. A malformed literal therefore does not
error: it silently changes the value's type. Everything this provider puts in
an override goes through these helpers, and ``unescape_toml`` inverts
``toml_str`` so the argv parsers can read back exactly what was rendered.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

#: TOML's short escapes. Every other control character becomes ``\\uXXXX``.
_SHORT_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}
_UNESCAPES = {escape[1]: char for char, escape in _SHORT_ESCAPES.items()}
_UNICODE_ESCAPE_WIDTHS = {"u": 4, "U": 8}
_DEL = 0x7F
_FIRST_PRINTABLE = 0x20


def toml_str(value: str) -> str:
    """Quote *value* as a TOML basic string that parses back to *value*.

    TOML forbids raw control characters in a basic string, and Codex treats
    an override that fails to parse as TOML as a raw literal instead, so an
    unescaped newline would silently change the value's type.
    """
    parts: list[str] = []
    for char in value:
        if char in _SHORT_ESCAPES:
            parts.append(_SHORT_ESCAPES[char])
        elif ord(char) < _FIRST_PRINTABLE or ord(char) == _DEL:
            parts.append(f"\\u{ord(char):04X}")
        else:
            parts.append(char)
    return '"' + "".join(parts) + '"'


def toml_array(values: Sequence[str]) -> str:
    """Render strings as a TOML inline array."""
    return "[" + ",".join(toml_str(value) for value in values) + "]"


def toml_bool(value: bool) -> str:  # noqa: FBT001 - a value, not a mode switch
    """Render a TOML boolean."""
    return "true" if value else "false"


def unescape_toml(body: str) -> str:
    """Invert ``toml_str`` on the text between the quotes.

    Raises:
        ValueError: *body* holds an escape ``toml_str`` never writes.
    """
    result: list[str] = []
    index = 0
    while index < len(body):
        char = body[index]
        if char != "\\":
            result.append(char)
            index += 1
            continue
        code = body[index + 1 : index + 2]
        if code in _UNESCAPES:
            result.append(_UNESCAPES[code])
            index += 2
        elif code in _UNICODE_ESCAPE_WIDTHS:
            width = _UNICODE_ESCAPE_WIDTHS[code]
            digits = body[index + 2 : index + 2 + width]
            if len(digits) != width:
                msg = f"truncated \\{code} escape in {body!r}"
                raise ValueError(msg)
            result.append(chr(int(digits, 16)))
            index += 2 + width
        else:
            msg = f"invalid TOML escape \\{code} in {body!r}"
            raise ValueError(msg)
    return "".join(result)
