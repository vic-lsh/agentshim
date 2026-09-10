"""Output-schema dialect checks and materialization.

Providers disagree about which JSON Schema subset their structured-output
flag accepts, and the failure mode is an expensive agent turn that ends in a
CLI parse error. These checks run before the process starts.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import TYPE_CHECKING, Any, cast

from ._files import atomic_write
from .profile import SchemaDialect

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

# Keywords that need schema-evaluation features outside the subset the
# provider CLIs implement. Field names are never inspected as keywords:
# ``properties``, ``$defs`` and ``definitions`` are traversed as maps of
# subschemas, so a property literally named ``if`` is fine.
UNSUPPORTED_KEYWORDS = frozenset(
    {
        "$anchor",
        "$dynamicAnchor",
        "$dynamicRef",
        "$id",
        "$schema",
        "allOf",
        "contains",
        "dependentRequired",
        "dependentSchemas",
        "else",
        "if",
        "maxContains",
        "minContains",
        "not",
        "oneOf",
        "patternProperties",
        "prefixItems",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)

_SUBSCHEMA_MAPS = frozenset({"properties", "$defs", "definitions"})


def dialect_problems(schema: Mapping[str, Any], dialect: SchemaDialect) -> list[str]:
    """Return every reason *schema* is not expressible in *dialect*.

    An empty list means the provider will accept it. The list is returned
    rather than raised so a caller can decide between failing and falling
    back to a prompt-level schema instruction.
    """
    problems: list[str] = []
    if schema.get("type") != "object":
        problems.append("# root must be a schema with type 'object'")
    _visit_for_problems(dict(schema), "#", dialect, problems)
    return problems


def _visit_for_problems(
    node: object,
    location: str,
    dialect: SchemaDialect,
    problems: list[str],
) -> None:
    if isinstance(node, list):
        for index, value in enumerate(cast("list[object]", node)):
            _visit_for_problems(value, f"{location}/{index}", dialect, problems)
        return
    if not isinstance(node, dict):
        return
    mapping = cast("dict[str, Any]", node)

    reference = mapping.get("$ref")
    if reference is not None:
        if not isinstance(reference, str) or not reference.startswith("#/"):
            problems.append(f"{location} uses a non-local $ref")
        return

    properties = mapping.get("properties")
    if properties is not None and not isinstance(properties, dict):
        problems.append(f"{location}/properties must be an object")
        return
    if (properties is not None or mapping.get("type") == "object") and dialect is SchemaDialect.STRICT:
        additional = mapping.get("additionalProperties")
        if additional not in (None, False):
            problems.append(f"{location} allows arbitrary object keys")

    for key, value in mapping.items():
        if key in _SUBSCHEMA_MAPS:
            if not isinstance(value, dict):
                problems.append(f"{location}/{key} must be an object")
                continue
            for name, subschema in cast("dict[str, Any]", value).items():
                _visit_for_problems(subschema, f"{location}/{key}/{name}", dialect, problems)
            continue
        if key in UNSUPPORTED_KEYWORDS:
            problems.append(f"{location} uses unsupported keyword {key!r}")
            continue
        _visit_for_problems(value, f"{location}/{key}", dialect, problems)


def normalize(schema: Mapping[str, Any], dialect: SchemaDialect) -> dict[str, Any]:
    """Return a copy of *schema* rewritten into the provider's house style.

    Generators such as Pydantic omit defaulted properties from ``required``
    and leave ``additionalProperties`` unset, which the CLIs read as "any
    subset of these keys, plus anything else". This closes every object,
    requires every declared property, drops ``default``, and strips the
    annotation siblings of a ``$ref`` that the strict subset forbids.
    Callers that want the schema passed through untouched skip this.
    """
    return cast("dict[str, Any]", _normalized(copy.deepcopy(dict(schema)), dialect))


def _normalized(node: object, dialect: SchemaDialect) -> object:
    if isinstance(node, list):
        return [_normalized(value, dialect) for value in cast("list[object]", node)]
    if not isinstance(node, dict):
        return node
    mapping = cast("dict[str, Any]", node)

    reference = mapping.get("$ref")
    if isinstance(reference, str):
        return {"$ref": reference}

    mapping.pop("default", None)
    properties = mapping.get("properties")
    if isinstance(properties, dict):
        _close_object(mapping, dialect)
        mapping["required"] = list(cast("dict[str, Any]", properties))
    elif mapping.get("type") == "object":
        _close_object(mapping, dialect)
        mapping["required"] = []

    for key, value in list(mapping.items()):
        if key in _SUBSCHEMA_MAPS and isinstance(value, dict):
            mapping[key] = {
                name: _normalized(subschema, dialect) for name, subschema in cast("dict[str, Any]", value).items()
            }
            continue
        mapping[key] = _normalized(value, dialect)
    return mapping


def _close_object(mapping: dict[str, Any], dialect: SchemaDialect) -> None:
    additional = mapping.get("additionalProperties")
    if additional in (None, False):
        mapping["additionalProperties"] = False
        return
    if dialect is SchemaDialect.STRICT:
        # dialect_problems already reported this; normalize does not guess.
        mapping["additionalProperties"] = False


def compact_json(schema: Mapping[str, Any]) -> str:
    """Serialize *schema* for an inline CLI flag, keeping argv small."""
    return json.dumps(dict(schema), separators=(",", ":"))


def materialize(schema: Mapping[str, Any], host_dir: Path) -> Path:
    """Write *schema* under *host_dir* as ``<sha>.json`` and return the path.

    The name is content-addressed so repeated turns with the same schema
    reuse one file, and the write is atomic so a CLI reading the path
    concurrently never sees a partial document.
    """
    encoded = (json.dumps(dict(schema), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    target = host_dir / f"{digest}.json"
    atomic_write(target, encoded)
    return target
