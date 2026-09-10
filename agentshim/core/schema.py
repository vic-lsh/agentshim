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

# Annotations that describe a schema rather than constrain a value. They are
# what a generator such as Pydantic emits alongside the real constraints, and
# what the strictest CLI subset refuses to read, so ``normalize`` removes
# them. Field names are never matched against this set: ``properties`` is
# traversed as a map of subschemas, so a property named ``title`` survives.
_METADATA_KEYWORDS = frozenset({"$schema", "$id", "title", "description", "examples"})

# ``$schema`` and ``$id`` identify the dialect and the document; a CLI that
# accepts open-ended schemas ignores them rather than failing on them. Codex's
# ``--output-schema`` subset does not, so ``STRICT`` keeps reporting them.
_OPEN_DIALECT_TOLERATES = frozenset({"$schema", "$id"})


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
    """Walk one schema node, appending a problem for every unsupported construct.

    Traversal is by JSON pointer so each problem names the offending node.
    A ``$ref`` node terminates the walk: its siblings are annotations the
    strict subset ignores, and the target is checked where it is defined.
    """
    if isinstance(node, list):
        for index, value in enumerate(cast("list[object]", node)):
            _visit_for_problems(value, f"{location}/{index}", dialect, problems)
        return
    if not isinstance(node, dict):
        return
    mapping = cast("dict[str, Any]", node)

    reference = mapping.get("$ref")
    if reference is not None:
        _check_local_ref(reference, location, problems)
        return

    if not _check_properties_shape(mapping, location, problems):
        return
    _check_closed_object(mapping, location, dialect, problems)
    _visit_members(mapping, location, dialect, problems)


def _check_local_ref(reference: object, location: str, problems: list[str]) -> None:
    """Reject a ``$ref`` the CLI would have to fetch or resolve externally."""
    if not isinstance(reference, str) or not reference.startswith("#/"):
        problems.append(f"{location} uses a non-local $ref")


def _check_properties_shape(mapping: dict[str, Any], location: str, problems: list[str]) -> bool:
    """Check that ``properties``, if present, is an object.

    Returns ``False`` when it is not, in which case the caller stops: nothing
    below a malformed ``properties`` can be read as a schema, and descending
    would only report the same defect once per child.
    """
    properties = mapping.get("properties")
    if properties is not None and not isinstance(properties, dict):
        problems.append(f"{location}/properties must be an object")
        return False
    return True


def _check_closed_object(
    mapping: dict[str, Any],
    location: str,
    dialect: SchemaDialect,
    problems: list[str],
) -> None:
    """Require an object node to forbid undeclared keys, in the strict dialect.

    A node counts as an object if it declares ``properties`` or says so with
    ``type``. Looser dialects accept an open object, so nothing is reported.
    """
    if dialect is not SchemaDialect.STRICT:
        return
    if mapping.get("properties") is None and mapping.get("type") != "object":
        return
    additional = mapping.get("additionalProperties")
    if additional not in (None, False):
        problems.append(f"{location} allows arbitrary object keys")


def _visit_members(
    mapping: dict[str, Any],
    location: str,
    dialect: SchemaDialect,
    problems: list[str],
) -> None:
    """Check every entry of a schema node, keyword or subschema.

    Only keys reached here are matched against ``UNSUPPORTED_KEYWORDS``, which
    is what keeps a property named ``if`` from being read as the ``if`` keyword.
    """
    for key, value in mapping.items():
        if key in _SUBSCHEMA_MAPS:
            _visit_subschema_map(value, f"{location}/{key}", dialect, problems)
        elif key in UNSUPPORTED_KEYWORDS:
            if dialect is not SchemaDialect.STRICT and key in _OPEN_DIALECT_TOLERATES:
                continue
            problems.append(f"{location} uses unsupported keyword {key!r}")
        elif key in _METADATA_KEYWORDS:
            # Annotations, not constraints: nothing below them is a schema.
            continue
        else:
            _visit_for_problems(value, f"{location}/{key}", dialect, problems)


def _visit_subschema_map(
    value: object,
    location: str,
    dialect: SchemaDialect,
    problems: list[str],
) -> None:
    """Walk a map of named subschemas such as ``properties`` or ``$defs``.

    The names are user-chosen field names, so they are traversed as data and
    never inspected as schema keywords.
    """
    if not isinstance(value, dict):
        problems.append(f"{location} must be an object")
        return
    for name, subschema in cast("dict[str, Any]", value).items():
        _visit_for_problems(subschema, f"{location}/{name}", dialect, problems)


def normalize(schema: Mapping[str, Any], dialect: SchemaDialect) -> dict[str, Any]:
    """Return a copy of *schema* rewritten into the provider's house style.

    Generators such as Pydantic omit defaulted properties from ``required``
    and leave ``additionalProperties`` unset, which the CLIs read as "any
    subset of these keys, plus anything else". This closes every object,
    requires every declared property, drops ``default``, strips the annotation
    siblings of a ``$ref`` that the strict subset forbids, and removes the
    document metadata (``$schema``, ``$id``, ``title``, ``description``,
    ``examples``) that ``dialect_problems`` reports under ``STRICT``. What
    comes back is therefore a schema the strict subset accepts. Callers that
    want the schema passed through untouched skip this.
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
    for annotation in _METADATA_KEYWORDS:
        mapping.pop(annotation, None)
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
                name: _normalized(subschema, dialect)
                for name, subschema in cast("dict[str, Any]", value).items()
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
