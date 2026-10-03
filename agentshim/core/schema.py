"""Output-schema dialect checks and materialization.

Providers disagree about which JSON Schema subset their structured-output
flag accepts, and the failure mode is an expensive agent turn that ends in a
CLI parse error. These checks run before the process starts.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from ._files import atomic_write
from .errors import ProviderCapabilityError
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

# Keywords whose values are data, never subschemas. ``required`` is a list of
# field names and ``enum``/``const`` hold instance values, so a name such as
# ``"$ref"`` inside them must not be read as a keyword.
_NON_SCHEMA_KEYWORDS = frozenset({"required", "enum", "const"})


def dialect_problems(schema: Mapping[str, Any], dialect: SchemaDialect) -> list[str]:
    """Return every reason *schema* is not expressible in *dialect*.

    An empty list means the provider will accept it. The list is returned
    rather than raised so a caller can decide between failing and falling
    back to a prompt-level schema instruction. Each problem starts with the
    JSON pointer of the offending node.

    ``STRICT`` is OpenAI's strict structured-output subset, which Codex sends
    with ``strict: true``: every object declares ``properties``, sets
    ``additionalProperties: false`` and lists exactly its ``properties`` in
    ``required`` (an optional value is
    expressed as nullable instead), the root is not an ``anyOf``, and every
    ``$ref`` resolves inside the document.
    """
    root = dict(schema)
    walk = _Walk(root=root, dialect=dialect, problems=[])
    if schema.get("type") != "object":
        walk.problems.append("# root must be a schema with type 'object'")
    if dialect is SchemaDialect.STRICT and "anyOf" in schema:
        walk.problems.append("# root must not use 'anyOf'")
    walk.visit(root, "#")
    return walk.problems


def _pointer_token(name: str) -> str:
    """Escape a field name as one JSON pointer reference token (RFC 6901)."""
    return name.replace("~", "~0").replace("/", "~1")


def _is_object_node(mapping: dict[str, Any]) -> bool:
    """Say whether a node describes an object, nullable or not."""
    if mapping.get("properties") is not None:
        return True
    declared = mapping.get("type")
    if isinstance(declared, list):
        return "object" in cast("list[object]", declared)
    return declared == "object"


@dataclass
class _Walk:
    """One pass of ``dialect_problems`` over a schema document.

    Traversal is by JSON pointer so each problem names the offending node.
    """

    root: dict[str, Any]
    dialect: SchemaDialect
    problems: list[str]

    def visit(self, node: object, location: str) -> None:
        """Walk one schema node, appending a problem for every unsupported construct.

        A ``$ref`` node terminates the walk: its siblings are annotations the
        strict subset ignores, and the target is checked where it is defined,
        since every local target is inside the document this walk covers.
        """
        if isinstance(node, list):
            for index, value in enumerate(cast("list[object]", node)):
                self.visit(value, f"{location}/{index}")
            return
        if not isinstance(node, dict):
            return
        mapping = cast("dict[str, Any]", node)

        reference = mapping.get("$ref")
        if reference is not None:
            self._check_ref(reference, location)
            return

        if not self._check_properties_shape(mapping, location):
            return
        if self.dialect is SchemaDialect.STRICT and _is_object_node(mapping):
            self._check_closed_object(mapping, location)
            self._check_all_required(mapping, location)
        self._visit_members(mapping, location)

    def _check_ref(self, reference: object, location: str) -> None:
        """Reject a ``$ref`` the CLI would have to fetch, or one that dangles."""
        if not isinstance(reference, str) or not (reference == "#" or reference.startswith("#/")):
            self.problems.append(f"{location} uses a non-local $ref")
            return
        if self.dialect is SchemaDialect.STRICT and not _resolves(self.root, reference):
            self.problems.append(f"{location} has a $ref {reference!r} that resolves to nothing")

    def _check_properties_shape(self, mapping: dict[str, Any], location: str) -> bool:
        """Check that ``properties``, if present, is an object.

        Returns ``False`` when it is not, in which case the caller stops:
        nothing below a malformed ``properties`` can be read as a schema, and
        descending would only report the same defect once per child.
        """
        properties = mapping.get("properties")
        if properties is not None and not isinstance(properties, dict):
            self.problems.append(f"{location}/properties must be an object")
            return False
        return True

    def _check_closed_object(self, mapping: dict[str, Any], location: str) -> None:
        """Require an object node to declare ``properties`` and close itself.

        Strict mode needs ``additionalProperties`` present, not merely not
        ``true``: an absent one is an open object in JSON Schema. It also
        refuses an object with no ``properties`` at all, even an empty one.
        """
        if "properties" not in mapping:
            self.problems.append(f"{location} must declare properties (use {{}} for none)")
        if "additionalProperties" not in mapping:
            self.problems.append(f"{location} must set additionalProperties: false")
        elif mapping["additionalProperties"] is not False:
            self.problems.append(f"{location} allows arbitrary object keys")

    def _check_all_required(self, mapping: dict[str, Any], location: str) -> None:
        """Require ``required`` to list exactly the declared properties.

        Strict mode has no optional properties: a value that may be absent is
        declared required and nullable, e.g. ``{"type": ["string", "null"]}``.
        """
        properties = cast("dict[str, Any]", mapping.get("properties") or {})
        required = mapping.get("required", [])
        if not isinstance(required, list) or not all(
            isinstance(name, str) for name in cast("list[object]", required)
        ):
            self.problems.append(f"{location}/required must be an array of property names")
            return
        listed = set(cast("list[str]", required))
        for name in properties:
            if name not in listed:
                self.problems.append(
                    f"{location}/properties/{_pointer_token(name)} is optional; strict mode "
                    "requires every property in 'required' (make it nullable instead)"
                )
        for name in cast("list[str]", required):
            if name not in properties:
                self.problems.append(
                    f"{location}/required names {name!r}, which is not in 'properties'"
                )

    def _visit_members(self, mapping: dict[str, Any], location: str) -> None:
        """Check every entry of a schema node, keyword or subschema.

        Only keys reached here are matched against ``UNSUPPORTED_KEYWORDS``,
        which is what keeps a property named ``if`` from being read as the
        ``if`` keyword.
        """
        for key, value in mapping.items():
            if key in _SUBSCHEMA_MAPS:
                self._visit_subschema_map(value, f"{location}/{key}")
            elif key in UNSUPPORTED_KEYWORDS:
                if self.dialect is not SchemaDialect.STRICT and key in _OPEN_DIALECT_TOLERATES:
                    continue
                self.problems.append(f"{location} uses unsupported keyword {key!r}")
            elif key in _METADATA_KEYWORDS or key in _NON_SCHEMA_KEYWORDS:
                # Annotations or plain values: nothing below them is a schema.
                continue
            else:
                self.visit(value, f"{location}/{key}")

    def _visit_subschema_map(self, value: object, location: str) -> None:
        """Walk a map of named subschemas such as ``properties`` or ``$defs``.

        The names are user-chosen field names, so they are traversed as data
        and never inspected as schema keywords.
        """
        if not isinstance(value, dict):
            self.problems.append(f"{location} must be an object")
            return
        for name, subschema in cast("dict[str, Any]", value).items():
            self.visit(subschema, f"{location}/{_pointer_token(name)}")


def _resolves(root: dict[str, Any], reference: str) -> bool:
    """Say whether a local ``$ref`` names a schema inside *root*."""
    node: object = root
    for raw in reference[2:].split("/") if reference != "#" else []:
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict):
            mapping = cast("dict[str, object]", node)
            if token not in mapping:
                return False
            node = mapping[token]
        elif isinstance(node, list):
            items = cast("list[object]", node)
            if not token.isdigit() or int(token) >= len(items):
                return False
            node = items[int(token)]
        else:
            return False
    return isinstance(node, dict)


def normalize(schema: Mapping[str, Any], dialect: SchemaDialect) -> dict[str, Any]:
    """Return a copy of *schema* rewritten into the provider's house style.

    Generators such as Pydantic omit defaulted properties from ``required``
    and leave ``additionalProperties`` unset, which the CLIs read as "any
    subset of these keys, plus anything else". This closes every object,
    requires every declared property, drops ``default``, strips the annotation
    siblings of a ``$ref`` that the strict subset forbids, and removes the
    document metadata (``$schema``, ``$id``, ``title``, ``description``,
    ``examples``) that ``dialect_problems`` reports under ``STRICT``. An open
    map (``additionalProperties`` a schema or ``true``) is kept, since closing
    it would change what the schema accepts; so ``dialect_problems`` on the
    normalized schema reports exactly what the dialect cannot express.
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
    for annotation in _METADATA_KEYWORDS:
        mapping.pop(annotation, None)
    properties = mapping.get("properties")
    if isinstance(properties, dict):
        _close_object(mapping, dialect)
        mapping["required"] = list(cast("dict[str, Any]", properties))
    elif _is_object_node(mapping):
        _close_object(mapping, dialect)
        mapping["properties"] = {}
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
    """Close an object whose extra keys are merely unspecified.

    An open map (``additionalProperties`` a schema or ``true``) is left open
    in every dialect: closing it would silently turn ``dict[str, float]`` into
    an object that can only be empty. Under ``STRICT`` it stays a problem for
    ``dialect_problems`` to report on the normalized schema.
    """
    del dialect  # an open map is left open in every dialect
    if mapping.get("additionalProperties") in (None, False):
        mapping["additionalProperties"] = False


def compact_json(schema: Mapping[str, Any]) -> str:
    """Serialize *schema* for an inline CLI flag, keeping argv small."""
    return json.dumps(dict(schema), separators=(",", ":"))


def materialize(schema: Mapping[str, Any], host_dir: Path) -> Path:
    """Write *schema* under *host_dir* as ``<sha>.json`` and return the path.

    The name is content-addressed so repeated turns with the same schema
    reuse one file, and the write is atomic so a CLI reading the path
    concurrently never sees a partial document.

    A schema carrying a value JSON cannot express, a ``NaN`` or an infinity,
    raises ``ProviderCapabilityError``: the provider cannot be given this
    schema, which is the same answer a dialect problem gets, and it is what
    keeps a bare ``ValueError`` out of ``turn()``.
    """
    try:
        serialized = json.dumps(dict(schema), indent=2, sort_keys=True, allow_nan=False)
    except ValueError as exc:
        msg = f"output schema cannot be serialized as JSON: {exc}"
        raise ProviderCapabilityError(msg) from exc
    encoded = (serialized + "\n").encode()
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    target = host_dir / f"{digest}.json"
    atomic_write(target, encoded)
    return target
