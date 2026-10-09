"""Generate the typed Codex app-server protocol module from the CLI's own schema.

Three steps, each usable on its own:

1. ``--from-cli`` (or ``--from-export DIR``) reads the JSON Schema exported by
   ``codex app-server generate-json-schema``, keeps only the transitive closure
   of the types named in ``scripts/codex_protocol/allowlist.json``, and writes
   that pruned schema to ``scripts/codex_protocol/schema.json`` together with
   the codex-cli version it came from.
2. Without those flags the checked-in pruned schema is the input, so the
   output is reproducible without the CLI installed.
3. The result is written to
   ``agentshim/providers/codex/app_server/protocol.py``; ``--check`` compares
   instead of writing and exits non-zero on drift.

The generator uses only the standard library. The shape of the emitted code
(frozen dataclasses, tolerant inbound decoding, strict outbound types) is
described in ``docs/architecture.md``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "scripts" / "codex_protocol"
ALLOWLIST_PATH = DATA_DIR / "allowlist.json"
SCHEMA_PATH = DATA_DIR / "schema.json"
OUTPUT_PATH = ROOT / "agentshim" / "providers" / "codex" / "app_server" / "protocol.py"
EXPORT_BUNDLE = "codex_app_server_protocol.schemas.json"

#: Envelope class names the generator emits itself; schema types may not use them.
ENVELOPE_NAMES = frozenset(
    {
        "Notification",
        "ServerRequest",
        "Response",
        "ErrorResponse",
        "RpcError",
        "ClientRequest",
        "ClientNotification",
        "UnknownParams",
    }
)

#: Schema keys that carry prose or validation hints the generated code does not use.
DROPPED_KEYS = frozenset({"description", "title", "$schema", "minimum", "minLength", "format"})
#: Schema keys whose value is data, not a sub-schema, so it is copied untouched.
DATA_KEYS = frozenset({"default", "enum", "const", "required", "type"})


class GeneratorError(Exception):
    """The schema uses a construct the generator does not support, or is inconsistent."""


# --------------------------------------------------------------------------
# Step 1: pruning the exported schema
# --------------------------------------------------------------------------


def normalize(node: Any) -> Any:  # noqa: ANN401 - JSON in, JSON out
    """Rewrite refs to the flat ``#/definitions/Name`` form and drop prose keys."""
    if isinstance(node, list):
        return [normalize(item) for item in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "$ref":
            out[key] = "#/definitions/" + value.rsplit("/", 1)[1]
        elif key in DROPPED_KEYS:
            continue
        elif key == "properties":
            out[key] = {name: normalize(sub) for name, sub in value.items()}
        elif key in DATA_KEYS:
            out[key] = value
        else:
            out[key] = normalize(value)
    return out


def flatten_export(bundle: dict[str, Any]) -> dict[str, Any]:
    """Merge the bundle's ``v2`` namespace into one flat definitions table."""
    defs = bundle["definitions"]
    flat: dict[str, Any] = {k: v for k, v in defs.items() if k != "v2"}
    for key, value in defs.get("v2", {}).items():
        flat.setdefault(key, value)
    return flat


def collect_refs(node: Any, out: set[str]) -> None:  # noqa: ANN401 - JSON walk
    """Add every definition name referenced anywhere under ``node`` to ``out``."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref":
                out.add(value.rsplit("/", 1)[1])
            elif key in ("enum", "const", "default"):
                continue
            else:
                collect_refs(value, out)
    elif isinstance(node, list):
        for item in node:
            collect_refs(item, out)


def allowlist_roots(allow: dict[str, Any]) -> list[str]:
    """Every schema type name the allowlist names, in a stable order."""
    names: list[str] = []
    for req in allow["client_requests"]:
        names += [req["params"], req["result"]]
    names += [n["params"] for n in allow["server_notifications"]]
    for req in allow["server_requests"]:
        names += [req["params"], req["response"]]
    return names


def check_messages(bundle: dict[str, Any], allow: dict[str, Any]) -> None:
    """Fail if the export's method table disagrees with the allowlist."""
    tables = {
        "client_requests": "ClientRequest",
        "server_notifications": "ServerNotification",
        "server_requests": "ServerRequest",
    }
    problems: list[str] = []
    for section, union in tables.items():
        exported = {
            v["properties"]["method"]["enum"][0]: v["properties"].get("params", {}).get("$ref")
            for v in bundle["definitions"][union]["oneOf"]
        }
        for entry in allow[section]:
            ref = exported.get(entry["method"], "<absent>")
            want = entry["params"]
            if ref is None or ref == "<absent>" or ref.rsplit("/", 1)[1] != want:
                problems.append(f"{section} {entry['method']}: export has {ref}, allowlist {want}")
    if problems:
        raise GeneratorError("allowlist disagrees with the export:\n  " + "\n  ".join(problems))


def prune(export_dir: Path, allow: dict[str, Any], codex_version: str) -> dict[str, Any]:
    """Build the pruned schema document from a ``generate-json-schema`` export."""
    bundle = json.loads((export_dir / EXPORT_BUNDLE).read_text())
    check_messages(bundle, allow)
    flat = flatten_export(bundle)
    missing = [n for n in allowlist_roots(allow) if n not in flat]
    if missing:
        raise GeneratorError(f"allowlisted types missing from the export: {missing}")
    opaque = set(allow.get("opaque", []))
    unknown_opaque = sorted(opaque - set(flat))
    if unknown_opaque:
        raise GeneratorError(f"opaque types missing from the export: {unknown_opaque}")
    seen: set[str] = set()
    stack = list(allowlist_roots(allow))
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        refs: set[str] = set()
        if name not in opaque:
            collect_refs(flat[name], refs)
        stack.extend(refs)
    return {
        "codex_version": codex_version,
        "messages": allow,
        "definitions": {n: {} if n in opaque else normalize(flat[n]) for n in sorted(seen)},
    }


def dump_schema(schema: dict[str, Any]) -> str:
    """Serialize the pruned schema deterministically."""
    return json.dumps(schema, indent=1, sort_keys=True) + "\n"


def export_from_cli(out_dir: Path) -> str:
    """Run the installed CLI's schema export into ``out_dir``; return its version."""
    version = subprocess.run(
        ["codex", "--version"],  # noqa: S607 - the CLI is looked up on PATH on purpose
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["codex", "app-server", "generate-json-schema", "--out", str(out_dir)],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"\d+\.\d+\.\d+\S*", version)
    return match.group(0) if match else version


# --------------------------------------------------------------------------
# Step 2: schema -> intermediate representation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Prim:
    """A JSON scalar."""

    name: str  # str | int | float | bool


@dataclass(frozen=True)
class JsonAny:
    """Any JSON value (an unconstrained schema)."""


@dataclass(frozen=True)
class Ref:
    """A named definition."""

    name: str


@dataclass(frozen=True)
class Seq:
    """A JSON array."""

    item: Ty


@dataclass(frozen=True)
class Dict:
    """A JSON object used as a string-keyed map."""

    item: Ty


@dataclass(frozen=True)
class Opt:
    """A value that may be JSON ``null``."""

    inner: Ty


@dataclass(frozen=True)
class Either:
    """A scalar that may be one of several JSON types."""

    members: tuple[Ty, ...]


Ty = Prim | JsonAny | Ref | Seq | Dict | Opt | Either


@dataclass
class Field:
    """One property of an object definition."""

    wire: str
    py: str
    ty: Ty
    mode: str  # required | optional | default
    default: str | None = None  # Python expression when mode == default


@dataclass
class ObjectDef:
    """A JSON object, emitted as a frozen dataclass."""

    name: str
    fields: list[Field]
    tag: tuple[str, str] | None = None  # (key, value) of an internally tagged variant
    wrap: str | None = None  # key of an externally tagged variant
    scalar: bool = False  # the wrapped payload is one non-object value, held in ``value``


@dataclass
class EnumDef:
    """A closed set of strings."""

    name: str
    members: list[tuple[str, str]]  # (python name, wire value)


@dataclass
class AliasDef:
    """A name for another type."""

    name: str
    ty: Ty


@dataclass
class TaggedDef:
    """A union of objects selected by a const member."""

    name: str
    key: str
    variants: dict[str, str]  # tag value -> variant ObjectDef name


@dataclass
class ExternalDef:
    """A union of bare strings and single-key objects."""

    name: str
    units: str | None  # name of the EnumDef holding the bare strings
    variants: dict[str, str]  # object key -> variant ObjectDef name


Def = ObjectDef | EnumDef | AliasDef | TaggedDef | ExternalDef

PY_KEYWORDS = frozenset(
    [
        "False",
        "None",
        "True",
        "and",
        "as",
        "assert",
        "async",
        "await",
        "break",
        "class",
        "continue",
        "def",
        "del",
        "elif",
        "else",
        "except",
        "finally",
        "for",
        "from",
        "global",
        "if",
        "import",
        "in",
        "is",
        "lambda",
        "nonlocal",
        "not",
        "or",
        "pass",
        "raise",
        "return",
        "try",
        "while",
        "with",
        "yield",
    ]
)
PY_RESERVED_ATTRS = frozenset({"from_wire", "to_wire", "parse_result", "METHOD", "TAG"})
#: A JSON Schema node: an object, or the boolean schema ``true``.
Schema = Union[dict[str, Any], bool]  # noqa: UP007 - evaluated at runtime on 3.10 too
SCALARS = {"string": "str", "integer": "int", "number": "float", "boolean": "bool"}


def words(text: str) -> list[str]:
    """Split ``camelCase``, ``snake_case`` and ``kebab-case`` text into lowercase words."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", spaced)
    return [w.lower() for w in re.split(r"[^A-Za-z0-9]+", spaced) if w]


def snake(text: str) -> str:
    """``camelCase`` -> ``snake_case``, avoiding Python keywords and our own attributes."""
    name = "_".join(words(text)) or "value"
    if name[0].isdigit():
        name = "v_" + name
    if name in PY_KEYWORDS or name in PY_RESERVED_ATTRS:
        name += "_"
    return name


def pascal(text: str) -> str:
    """``camelCase``/``a/b`` -> ``PascalCase``."""
    return "".join(w[:1].upper() + w[1:] for w in words(text)) or "Value"


def member_name(value: str) -> str:
    """Enum member name for a wire string."""
    name = "_".join(words(value)).upper() or "EMPTY"
    return "V_" + name if name[0].isdigit() else name


def is_null(node: dict[str, Any]) -> bool:
    """True for the ``{"type": "null"}`` schema."""
    return node.get("type") == "null"


def string_enum_values(node: dict[str, Any]) -> list[str] | None:
    """The values of a ``{"type": "string", "enum": [...]}`` schema, else ``None``."""
    if node.get("type") == "string" and "enum" in node and "properties" not in node:
        return list(node["enum"])
    return None


class Builder:
    """Turns the pruned definitions into the intermediate representation."""

    def __init__(self, definitions: dict[str, Any]) -> None:
        """Start with the pruned definition table; call ``build`` to fill ``defs``."""
        self.raw = definitions
        self.defs: dict[str, Def] = {}

    def build(self) -> dict[str, Def]:
        """Classify every definition, returning the registry (variants included)."""
        for name, node in self.raw.items():
            if name in ENVELOPE_NAMES:
                raise GeneratorError(f"schema type {name} collides with a generated envelope")
            self.define(name, node)
        return self.defs

    def add(self, definition: Def) -> None:
        """Register ``definition``, rejecting a duplicate name."""
        if definition.name in self.defs:
            raise GeneratorError(f"duplicate generated name {definition.name}")
        self.defs[definition.name] = definition

    # -- definitions ------------------------------------------------------

    def define(self, name: str, node: dict[str, Any]) -> None:
        """Register the definition ``name`` (plus any anonymous helpers it needs)."""
        values = string_enum_values(node)
        if values is not None:
            self.add(self.make_enum(name, values))
        elif "oneOf" in node and "properties" not in node:
            self.define_one_of(name, node["oneOf"])
        elif "properties" in node or self.is_empty_object(node):
            self.add(ObjectDef(name, self.fields_of(name, node)))
        else:
            self.add(AliasDef(name, self.type_of(node, name)))

    @staticmethod
    def is_empty_object(node: dict[str, Any]) -> bool:
        """True for ``{"type": "object"}``: a message with no members, such as ``{}``."""
        return node.get("type") == "object" and "additionalProperties" not in node

    def make_enum(self, name: str, values: list[str]) -> EnumDef:
        """An ``EnumDef`` with unique member names."""
        members = [(member_name(v), v) for v in values]
        if len({m for m, _ in members}) != len(members):
            raise GeneratorError(f"enum {name} has colliding member names")
        return EnumDef(name, members)

    def define_one_of(self, name: str, alts: list[dict[str, Any]]) -> None:
        """Classify a ``oneOf`` as enum, internally tagged or externally tagged union."""
        strings = [string_enum_values(a) for a in alts]
        if all(s is not None for s in strings):
            merged = [v for s in strings for v in (s or [])]
            self.add(self.make_enum(name, merged))
            return
        objects = [a for a, s in zip(alts, strings, strict=True) if s is None]
        tag = self.tag_key(objects) if len(objects) == len(alts) else None
        if tag is not None:
            self.define_tagged(name, tag, objects)
        else:
            self.define_external(name, [v for s in strings if s for v in s], objects)

    @staticmethod
    def tag_key(objects: list[dict[str, Any]]) -> str | None:
        """The property every alternative fixes to one const value, if any."""
        first = objects[0].get("properties", {})
        for key in sorted(first, key=lambda k: (k != "type", k)):
            if all(
                key in a.get("properties", {})
                and len(a["properties"][key].get("enum", [])) == 1
                and key in a.get("required", [])
                for a in objects
            ):
                return key
        return None

    def define_tagged(self, name: str, key: str, objects: list[dict[str, Any]]) -> None:
        """Register an internally tagged union and one object per variant."""
        variants: dict[str, str] = {}
        for alt in objects:
            tag = alt["properties"][key]["enum"][0]
            vname = name + pascal(tag)
            props = {k: v for k, v in alt["properties"].items() if k != key}
            node = {**alt, "properties": props}
            self.add(ObjectDef(vname, self.fields_of(vname, node), tag=(key, tag)))
            variants[tag] = vname
        self.add(TaggedDef(name, key, variants))

    def define_external(self, name: str, units: list[str], objects: list[dict[str, Any]]) -> None:
        """Register a union of bare strings and ``{key: payload}`` objects."""
        variants: dict[str, str] = {}
        for alt in objects:
            props = alt.get("properties", {})
            if len(props) != 1 or alt.get("required") != list(props):
                raise GeneratorError(f"{name}: unsupported oneOf alternative {sorted(props)}")
            ((key, payload),) = props.items()
            vname = name + pascal(key)
            if "properties" in payload:
                self.add(ObjectDef(vname, self.fields_of(vname, payload), wrap=key))
            else:
                value = Field("value", "value", self.type_of(payload, vname + "Value"), "required")
                self.add(ObjectDef(vname, [value], wrap=key, scalar=True))
            variants[key] = vname
        unit_name = None
        if units:
            unit_name = name + "Kind"
            self.add(self.make_enum(unit_name, units))
        self.add(ExternalDef(name, unit_name, variants))

    # -- fields -----------------------------------------------------------

    def fields_of(self, owner: str, node: dict[str, Any]) -> list[Field]:
        """The dataclass fields for an object schema."""
        required = set(node.get("required", []))
        out: list[Field] = []
        seen: set[str] = set()
        for wire, sub in node.get("properties", {}).items():
            fld = self.make_field(owner, wire, sub, wire in required)
            if fld.py in seen:
                raise GeneratorError(f"{owner}: field names collide on {fld.py}")
            seen.add(fld.py)
            out.append(fld)
        return out

    def make_field(
        self,
        owner: str,
        wire: str,
        node: Schema,
        required: bool,  # noqa: FBT001
    ) -> Field:
        """One field, with its presence mode and default."""
        ty = self.type_of(node, owner + pascal(wire))
        py = snake(wire)
        if required:
            return Field(wire, py, ty, "required")
        inner = ty.inner if isinstance(ty, Opt) else ty
        default = self.default_expr(
            inner, (node.get("default") if isinstance(node, dict) else None)
        )
        if default is None:
            return Field(wire, py, inner, "optional")
        return Field(wire, py, inner, "default", default)

    def default_expr(self, ty: Ty, value: Any) -> str | None:  # noqa: ANN401 - JSON value
        """Python source for a non-null schema default, or ``None`` if there is none."""
        if value is None:
            return None
        if isinstance(value, list) and not value and isinstance(ty, Seq):
            return "()"
        if isinstance(ty, Ref) and isinstance(value, str):
            target = self.defs.get(ty.name)
            if target is None:
                target = self.peek_enum(ty.name)
            if isinstance(target, EnumDef):
                member = next((m for m, v in target.members if v == value), None)
                if member is None:
                    raise GeneratorError(f"default {value!r} is not a member of {ty.name}")
                return f"{ty.name}.{member}"
        if isinstance(ty, Prim) and isinstance(value, (bool, int, float, str)):
            return repr(value)
        raise GeneratorError(f"unsupported default {value!r} for {ty}")

    def peek_enum(self, name: str) -> Def | None:
        """Define ``name`` now if it is an enum not yet registered (for defaults)."""
        node = self.raw.get(name)
        if node is None:
            return None
        values = string_enum_values(node)
        if values is None and "oneOf" in node:
            strings = [string_enum_values(a) for a in node["oneOf"]]
            if all(s is not None for s in strings):
                values = [v for s in strings for v in (s or [])]
        if values is None:
            return None
        return self.make_enum(name, values)

    # -- types ------------------------------------------------------------

    def type_of(self, node: Schema, hint: str) -> Ty:
        """The type a property schema describes; ``hint`` names anonymous helpers."""
        if isinstance(node, bool):
            return JsonAny()
        if "$ref" in node:
            return Ref(node["$ref"].rsplit("/", 1)[1])
        if "allOf" in node:
            if len(node["allOf"]) != 1:
                raise GeneratorError(f"{hint}: allOf with several members")
            return self.type_of(node["allOf"][0], hint)
        if "anyOf" in node:
            alts = node["anyOf"]
            rest = [a for a in alts if not is_null(a)]
            tys = tuple(self.type_of(a, hint) for a in rest)
            ty: Ty = tys[0] if len(tys) == 1 else Either(tys)
            return Opt(ty) if len(rest) < len(alts) else ty
        if "oneOf" in node:
            self.define(hint, node)
            return Ref(hint)
        if "enum" in node:
            self.add(self.make_enum(hint, list(node["enum"])))
            return Ref(hint)
        kind = node.get("type")
        if isinstance(kind, list):
            rest_kinds = [k for k in kind if k != "null"]
            tys = tuple(self.type_of({**node, "type": k}, hint) for k in rest_kinds)
            ty = tys[0] if len(tys) == 1 else Either(tys)
            return Opt(ty) if len(rest_kinds) < len(kind) else ty
        if kind in SCALARS:
            return Prim(SCALARS[kind])
        if kind == "array":
            return Seq(self.type_of(node["items"], hint + "Item") if "items" in node else JsonAny())
        if kind == "object" or "properties" in node:
            return self.object_type(node, hint)
        if kind == "null" or not node:
            return JsonAny()
        raise GeneratorError(f"{hint}: unsupported schema {json.dumps(node)[:120]}")

    def object_type(self, node: dict[str, Any], hint: str) -> Ty:
        """An inline object: a named helper dataclass, a map, or free-form JSON."""
        if "properties" in node:
            self.add(ObjectDef(hint, self.fields_of(hint, node)))
            return Ref(hint)
        extra = node.get("additionalProperties")
        if isinstance(extra, dict):
            return Dict(self.type_of(extra, hint + "Value"))
        return JsonAny()


# --------------------------------------------------------------------------
# Step 3: intermediate representation -> Python source
# --------------------------------------------------------------------------


def type_refs(ty: Ty) -> list[str]:
    """Names of the definitions referenced by ``ty``."""
    if isinstance(ty, Ref):
        return [ty.name]
    if isinstance(ty, (Seq, Dict)):
        return type_refs(ty.item)
    if isinstance(ty, Opt):
        return type_refs(ty.inner)
    if isinstance(ty, Either):
        return [n for m in ty.members for n in type_refs(m)]
    return []


def def_deps(definition: Def) -> list[str]:
    """Names this definition's emitted code needs to exist first."""
    if isinstance(definition, ObjectDef):
        return [n for f in definition.fields for n in type_refs(f.ty)]
    if isinstance(definition, AliasDef):
        return type_refs(definition.ty)
    if isinstance(definition, TaggedDef):
        return list(definition.variants.values())
    if isinstance(definition, ExternalDef):
        units = [definition.units] if definition.units else []
        return units + list(definition.variants.values())
    return []


def reachable(defs: dict[str, Def], roots: list[str]) -> set[str]:
    """Definitions reachable from ``roots`` through field types and union members."""
    seen: set[str] = set()
    stack = list(roots)
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        stack.extend(def_deps(defs[name]))
    return seen


def emission_order(defs: dict[str, Def]) -> list[str]:
    """Dependencies first; a cycle is broken where it closes (annotations are lazy)."""
    order: list[str] = []
    state: dict[str, int] = {}

    def visit(name: str) -> None:
        if name in state:
            return
        state[name] = 1
        for dep in def_deps(defs[name]):
            visit(dep)
        order.append(name)
        state[name] = 2

    for name in sorted(defs):
        visit(name)
    return order


@dataclass
class Emitter:
    """Renders the registry as the protocol module."""

    defs: dict[str, Def]
    schema: dict[str, Any]
    inbound: set[str]
    outbound: set[str]
    lines: list[str] = field(default_factory=list)

    # -- type rendering ---------------------------------------------------

    def tolerant(self, name: str) -> bool:
        """True when ``name`` can arrive from the server, so unknown values must survive."""
        return name in self.inbound

    def render(self, ty: Ty) -> str:
        """Annotation text for ``ty``."""
        if isinstance(ty, Prim):
            return ty.name
        if isinstance(ty, JsonAny):
            return "JsonValue"
        if isinstance(ty, Ref):
            target = self.defs[ty.name]
            if isinstance(target, EnumDef) and self.tolerant(ty.name):
                return f"{ty.name} | str"
            return ty.name
        if isinstance(ty, Seq):
            return f"tuple[{self.render(ty.item)}, ...]"
        if isinstance(ty, Dict):
            return f"dict[str, {self.render(ty.item)}]"
        if isinstance(ty, Opt):
            return f"{self.render(ty.inner)} | None"
        return " | ".join(self.render(m) for m in ty.members)

    def decoder(self, ty: Ty) -> str:
        """Source of a ``(value, path) -> T`` decoder for ``ty``."""
        if isinstance(ty, Prim):
            return f"decode_{ty.name}"
        if isinstance(ty, JsonAny):
            return "decode_json"
        if isinstance(ty, Ref):
            target = self.defs[ty.name]
            if isinstance(target, AliasDef):
                return self.decoder(target.ty)
            if isinstance(target, (TaggedDef, ExternalDef)):
                return f"{snake(ty.name)}_from_wire"
            return f"{ty.name}.from_wire"
        if isinstance(ty, Seq):
            return f"tuple_of({self.decoder(ty.item)})"
        if isinstance(ty, Dict):
            return f"dict_of({self.decoder(ty.item)})"
        if isinstance(ty, Opt):
            return f"nullable({self.decoder(ty.inner)})"
        return "first_of(" + ", ".join(self.decoder(m) for m in ty.members) + ")"

    def put(self, text: str = "") -> None:
        """Append a line of output."""
        self.lines.append(text)

    # -- per-definition emission -----------------------------------------

    def emit_def(self, name: str) -> None:
        """Emit one definition (variants of a union are emitted as their own definitions)."""
        definition = self.defs[name]
        if isinstance(definition, EnumDef):
            self.emit_enum(definition)
        elif isinstance(definition, AliasDef):
            self.put(f"{name} = {self.render(definition.ty)}")
            self.put()
            self.put()
        elif isinstance(definition, ObjectDef):
            self.emit_object(definition)
        elif isinstance(definition, TaggedDef):
            self.emit_tagged(definition)
        else:
            self.emit_external(definition)

    def emit_enum(self, definition: EnumDef) -> None:
        """A ``str`` enum; inbound enums keep unknown strings as plain ``str``."""
        name = definition.name
        decode = "decode_enum_tolerant" if self.tolerant(name) else "decode_enum"
        result = f"{name} | str" if self.tolerant(name) else name
        self.put(f"class {name}(str, Enum):")
        self.put(f'    """Codex wire enum ``{name}``."""')
        self.put()
        for member, value in definition.members:
            self.put(f"    {member} = {value!r}")
        self.put()
        self.put("    @classmethod")
        self.put(f'    def from_wire(cls, value: object, path: str = "{name}") -> {result}:')
        self.put('        """Decode a wire string."""')
        self.put(f"        return {decode}(cls, value, path)")
        self.put()
        self.put()

    def request_info(self, name: str) -> dict[str, str] | None:
        """The allowlist entry whose params type is ``name``, if it is a client request."""
        for entry in self.schema["messages"]["client_requests"]:
            if entry["params"] == name:
                return entry
        return None

    def emit_object(self, definition: ObjectDef) -> None:
        """A frozen dataclass with ``from_wire`` and ``to_wire``."""
        name = definition.name
        self.put("@dataclass(frozen=True, kw_only=True)")
        self.put(f"class {name}:")
        self.put(f'    """Codex wire type ``{name}``."""')
        self.put()
        info = self.request_info(name)
        if info:
            self.put(f"    METHOD: ClassVar[str] = {info['method']!r}")
            self.put()
        for fld in definition.fields:
            self.put(self.field_declaration(fld))
        if definition.fields or info:
            self.put()
        self.emit_from_wire(definition)
        self.put()
        self.emit_to_wire(definition)
        if info:
            self.put()
            self.put("    @staticmethod")
            self.put(
                f"    def parse_result(result: object) -> {info['result']}:\n"
                f'        """Decode the response to this request."""\n'
                f"        return {info['result']}.from_wire(result)"
            )
        self.put()
        self.put()

    def field_declaration(self, fld: Field) -> str:
        """``name: type [= default]`` for a dataclass field."""
        annotation = self.render(fld.ty)
        if fld.mode == "required":
            return f"    {fld.py}: {annotation}"
        if fld.mode == "optional":
            return f"    {fld.py}: {annotation} | None = None"
        return f"    {fld.py}: {annotation} = {fld.default}"

    def field_decode(self, fld: Field) -> str:
        """The keyword argument that decodes ``fld`` from the object ``obj``."""
        dec = self.decoder(fld.ty)
        args = f'obj, "{fld.wire}", path'
        if fld.mode == "required":
            return f"{fld.py}=required({args}, {dec})"
        if fld.mode == "optional":
            return f"{fld.py}=optional({args}, {dec})"
        return f"{fld.py}=with_default({args}, {dec}, default={fld.default})"

    def emit_from_wire(self, definition: ObjectDef) -> None:
        """The tolerant decoder classmethod."""
        name = definition.name
        self.put("    @classmethod")
        self.put(f'    def from_wire(cls, value: object, path: str = "{name}") -> {name}:')
        self.put(
            '        """Decode the wire object; extra keys are ignored, a missing or'
            ' ill-typed field raises."""'
        )
        if definition.scalar:
            self.put("        obj = as_object(value, path)")
            self.put(
                f"        return cls(value=required(obj, {definition.wrap!r}, path, {self.decoder(definition.fields[0].ty)}))"
            )
            return
        if definition.wrap:
            self.put("        outer = as_object(value, path)")
            self.put(f'        obj = required(outer, "{definition.wrap}", path, as_object)')
            self.put(f'        path = f"{{path}}.{definition.wrap}"')
        elif definition.fields:
            self.put("        obj = as_object(value, path)")
        else:
            self.put("        as_object(value, path)")
        if not definition.fields:
            self.put("        return cls()")
            return
        self.put("        return cls(")
        for fld in definition.fields:
            self.put(f"            {self.field_decode(fld)},")
        self.put("        )")

    def emit_to_wire(self, definition: ObjectDef) -> None:
        """The encoder: camelCase keys, ``None`` omitted unless the schema requires null."""
        self.put("    def to_wire(self) -> dict[str, JsonValue]:")
        self.put('        """Encode as the JSON object the server expects."""')
        if definition.scalar:
            self.put(f"        return {{{definition.wrap!r}: encode(self.value)}}")
            return
        required = [(f.wire, f.py) for f in definition.fields if f.mode != "optional"]
        optional = [(f.wire, f.py) for f in definition.fields if f.mode == "optional"]
        head = [(definition.tag[0], repr(definition.tag[1]))] if definition.tag else []
        body = ", ".join(
            [f"{k!r}: {v}" for k, v in head] + [f"{w!r}: self.{p}" for w, p in required]
        )
        extra = ", ".join(f"{w!r}: self.{p}" for w, p in optional)
        call = f"fields_to_wire({{{body}}}, {{{extra}}})"
        if definition.wrap:
            call = f"{{{definition.wrap!r}: {call}}}"
        self.put(f"        return {call}")

    def emit_tagged(self, definition: TaggedDef) -> None:
        """Union alias, unknown-variant class and decoder for a tagged union."""
        name = definition.name
        snake_name = snake(name)
        members = list(definition.variants.values())
        if self.tolerant(name):
            members.append(f"Unknown{name}")
            self.put("@dataclass(frozen=True)")
            self.put(f"class Unknown{name}:")
            self.put(f'    """A ``{name}`` variant this client version does not know."""')
            self.put()
            self.put("    raw: JsonObject")
            self.put()
            self.put("    def to_wire(self) -> JsonValue:")
            self.put('        """Encode the preserved wire object unchanged."""')
            self.put("        return self.raw")
            self.put()
            self.put()
        self.put(f"{name} = {' | '.join(members)}")
        self.put()
        self.put()
        table = f"_{snake_name.upper()}_VARIANTS"
        self.put(f"{table}: dict[str, Decoder[{name}]] = {{")
        for tag, vname in definition.variants.items():
            self.put(f"    {tag!r}: {vname}.from_wire,")
        self.put("}")
        self.put()
        self.put()
        unknown = f"Unknown{name}" if self.tolerant(name) else "None"
        self.put(f'def {snake_name}_from_wire(value: object, path: str = "{name}") -> {name}:')
        self.put(f'    """Decode a ``{name}`` by its ``{definition.key}`` tag."""')
        self.put(
            f"    return decode_tagged(\n"
            f"        value, path, key={definition.key!r}, variants={table}, unknown={unknown}\n"
            f"    )"
        )
        self.put()
        self.put()

    def emit_external(self, definition: ExternalDef) -> None:
        """Union alias, unknown-variant class and decoder for bare-string/single-key unions."""
        name = definition.name
        snake_name = snake(name)
        members = ([definition.units] if definition.units else []) + list(
            definition.variants.values()
        )
        if self.tolerant(name):
            members.append(f"Unknown{name}")
            self.put("@dataclass(frozen=True)")
            self.put(f"class Unknown{name}:")
            self.put(f'    """A ``{name}`` value this client version does not know."""')
            self.put()
            self.put("    raw: JsonValue")
            self.put()
            self.put("    def to_wire(self) -> JsonValue:")
            self.put('        """Encode the preserved wire value unchanged."""')
            self.put("        return self.raw")
            self.put()
            self.put()
        self.put(f"{name} = {' | '.join(members)}")
        self.put()
        self.put()
        table = f"_{snake_name.upper()}_VARIANTS"
        self.put(f"{table}: dict[str, Decoder[{name}]] = {{")
        for key, vname in definition.variants.items():
            self.put(f"    {key!r}: {vname}.from_wire,")
        self.put("}")
        self.put()
        self.put()
        units = (
            f"lambda text: enum_or_none({definition.units}, text)"
            if definition.units
            else "lambda _text: None"
        )
        unknown = f"Unknown{name}" if self.tolerant(name) else "None"
        self.put(f'def {snake_name}_from_wire(value: object, path: str = "{name}") -> {name}:')
        self.put(f'    """Decode a ``{name}``: a bare string or a single-key object."""')
        self.put(
            f"    return decode_external(\n"
            f"        value, path, units={units}, variants={table}, unknown={unknown}\n"
            f"    )"
        )
        self.put()
        self.put()

    # -- the whole module ------------------------------------------------

    def emit_module(self) -> str:
        """The complete source text of ``protocol.py``."""
        version = self.schema["codex_version"]
        for name in emission_order(self.defs):
            self.emit_def(name)
        self.emit_envelopes()
        body = "\n".join(self.lines).rstrip("\n") + "\n"
        used = [n for n in WIRE_IMPORTS if re.search(rf"\b{n}\b", body)]
        header = [
            '"""Typed Codex app-server protocol messages (generated)."""',
            "",
            f"# Generated by scripts/generate_codex_protocol.py from codex-cli {version}.",
            "# Do not edit; regenerate with `python scripts/generate_codex_protocol.py`.",
            "",
            "from __future__ import annotations",
            "",
            "from dataclasses import dataclass",
            "from enum import Enum",
            "from typing import ClassVar",
            "",
            "from ._wire import (",
            *[f"    {n}," for n in used],
            ")",
            "",
            f"CODEX_VERSION = {version!r}",
            '"""The codex-cli release whose schema these types were generated from."""',
            "",
            "",
        ]
        return "\n".join(header) + body

    def emit_envelopes(self) -> None:
        """Message envelopes, method tables and the two dispatchers."""
        messages = self.schema["messages"]
        subs = {
            "notification_params": [n["params"] for n in messages["server_notifications"]],
            "request_params": [n["params"] for n in messages["server_requests"]],
            "request_results": [n["response"] for n in messages["server_requests"]],
            "client_params": [n["params"] for n in messages["client_requests"]],
        }
        union = {k: " | ".join(v) for k, v in subs.items()}
        notif_table = "".join(
            f"    {n['method']!r}: {n['params']}.from_wire,\n"
            for n in messages["server_notifications"]
        )
        request_table = "".join(
            f"    {n['method']!r}: {n['params']}.from_wire,\n" for n in messages["server_requests"]
        )
        client_table = "".join(
            f"    {n['method']!r}: {n['params']}.from_wire,\n" for n in messages["client_requests"]
        )
        client_notes = "".join(f"    {n['method']!r},\n" for n in messages["client_notifications"])
        text = ENVELOPE_TEMPLATE.format(
            notification_params=union["notification_params"],
            request_params=union["request_params"],
            request_results=union["request_results"],
            client_params=union["client_params"],
            notif_table=notif_table,
            request_table=request_table,
            client_table=client_table,
            client_notes=client_notes,
        )
        self.lines.extend(text.rstrip("\n").split("\n"))


WIRE_IMPORTS = [
    "CodexProtocolError",
    "Decoder",
    "JsonObject",
    "JsonValue",
    "as_object",
    "decode_bool",
    "decode_enum",
    "decode_enum_tolerant",
    "decode_external",
    "decode_float",
    "decode_int",
    "decode_json",
    "decode_json_object",
    "decode_params",
    "decode_str",
    "decode_tagged",
    "dict_of",
    "encode",
    "enum_or_none",
    "fields_to_wire",
    "first_of",
    "nullable",
    "optional",
    "required",
    "tuple_of",
    "with_default",
]

ENVELOPE_TEMPLATE = '''\
@dataclass(frozen=True)
class UnknownParams:
    """Params of a method this client version does not know, kept as received."""

    raw: JsonValue

    def to_wire(self) -> JsonValue:
        """Encode the preserved params unchanged."""
        return self.raw


NotificationParams = {notification_params} | UnknownParams
ServerRequestParams = {request_params} | UnknownParams
ServerRequestResult = {request_results}
ClientRequestParams = {client_params}

_NOTIFICATION_PARAMS: dict[str, Decoder[NotificationParams]] = {{
{notif_table}}}

_SERVER_REQUEST_PARAMS: dict[str, Decoder[ServerRequestParams]] = {{
{request_table}}}

_CLIENT_REQUEST_PARAMS: dict[str, Decoder[ClientRequestParams]] = {{
{client_table}}}

CLIENT_NOTIFICATION_METHODS: frozenset[str] = frozenset(
    {{
{client_notes}    }}
)
"""Methods the client may send without an id and without params."""


@dataclass(frozen=True, kw_only=True)
class Notification:
    """A server-to-client notification: no id, never answered."""

    method: str
    params: NotificationParams
    emitted_at_ms: int | None = None

    @classmethod
    def from_wire(cls, value: object, path: str = "Notification") -> Notification:
        """Decode a notification; an unlisted method keeps its params as `UnknownParams`."""
        obj = as_object(value, path)
        method = required(obj, "method", path, decode_str)
        return cls(
            method=method,
            params=decode_params(obj, path, method, _NOTIFICATION_PARAMS, UnknownParams),
            emitted_at_ms=optional(obj, "emittedAtMs", path, decode_int),
        )

    def to_wire(self) -> dict[str, JsonValue]:
        """Encode as the JSON object the server sends."""
        out: dict[str, JsonValue] = {{"method": self.method, "params": encode(self.params)}}
        if self.emitted_at_ms is not None:
            out["emittedAtMs"] = self.emitted_at_ms
        return out


@dataclass(frozen=True, kw_only=True)
class ServerRequest:
    """A server-to-client request: the client must answer every one, even unknown methods."""

    id: RequestId
    method: str
    params: ServerRequestParams

    @classmethod
    def from_wire(cls, value: object, path: str = "ServerRequest") -> ServerRequest:
        """Decode a request; an unlisted method keeps its params as `UnknownParams`."""
        obj = as_object(value, path)
        method = required(obj, "method", path, decode_str)
        return cls(
            id=required(obj, "id", path, first_of(decode_str, decode_int)),
            method=method,
            params=decode_params(obj, path, method, _SERVER_REQUEST_PARAMS, UnknownParams),
        )

    def to_wire(self) -> dict[str, JsonValue]:
        """Encode as the JSON object the server sends."""
        return {{"id": self.id, "method": self.method, "params": encode(self.params)}}


@dataclass(frozen=True, kw_only=True)
class Response:
    """A successful reply to a request. ``result`` is decoded by the request's own type."""

    id: RequestId
    result: JsonValue

    @classmethod
    def from_wire(cls, value: object, path: str = "Response") -> Response:
        """Decode ``{{"id", "result"}}``."""
        obj = as_object(value, path)
        return cls(
            id=required(obj, "id", path, first_of(decode_str, decode_int)),
            result=required(obj, "result", path, decode_json),
        )

    def to_wire(self) -> dict[str, JsonValue]:
        """Encode as the JSON object on the wire."""
        return {{"id": self.id, "result": self.result}}


@dataclass(frozen=True, kw_only=True)
class RpcError:
    """The ``error`` member of a failed reply."""

    code: int
    message: str
    data: JsonValue = None

    @classmethod
    def from_wire(cls, value: object, path: str = "RpcError") -> RpcError:
        """Decode ``{{"code", "message", "data"?}}``."""
        obj = as_object(value, path)
        return cls(
            code=required(obj, "code", path, decode_int),
            message=required(obj, "message", path, decode_str),
            data=optional(obj, "data", path, decode_json),
        )

    def to_wire(self) -> dict[str, JsonValue]:
        """Encode as the JSON object on the wire."""
        out: dict[str, JsonValue] = {{"code": self.code, "message": self.message}}
        if self.data is not None:
            out["data"] = self.data
        return out


@dataclass(frozen=True, kw_only=True)
class ErrorResponse:
    """A failed reply to a request (``id`` is ``None`` if the peer could not tell)."""

    id: RequestId | None
    error: RpcError

    @classmethod
    def from_wire(cls, value: object, path: str = "ErrorResponse") -> ErrorResponse:
        """Decode ``{{"id", "error"}}``."""
        obj = as_object(value, path)
        return cls(
            id=optional(obj, "id", path, first_of(decode_str, decode_int)),
            error=required(obj, "error", path, RpcError.from_wire),
        )

    def to_wire(self) -> dict[str, JsonValue]:
        """Encode as the JSON object on the wire."""
        return {{"id": self.id, "error": self.error.to_wire()}}


@dataclass(frozen=True, kw_only=True)
class ClientRequest:
    """A client-to-server request; the method comes from the params type."""

    id: RequestId
    params: ClientRequestParams

    @classmethod
    def from_wire(cls, value: object, path: str = "ClientRequest") -> ClientRequest:
        """Decode a request whose method is one the allowlist covers."""
        obj = as_object(value, path)
        method = required(obj, "method", path, decode_str)
        decode = _CLIENT_REQUEST_PARAMS.get(method)
        if decode is None:
            raise CodexProtocolError(path, f"unsupported client method {{method!r}}")
        return cls(
            id=required(obj, "id", path, first_of(decode_str, decode_int)),
            params=required(obj, "params", path, decode),
        )

    def to_wire(self) -> dict[str, JsonValue]:
        """Encode as one line of the client's JSON stream."""
        return {{"id": self.id, "method": self.params.METHOD, "params": self.params.to_wire()}}


@dataclass(frozen=True, kw_only=True)
class ClientNotification:
    """A client-to-server notification without params (``initialized``)."""

    method: str

    @classmethod
    def from_wire(cls, value: object, path: str = "ClientNotification") -> ClientNotification:
        """Decode a notification whose method is one the allowlist covers."""
        obj = as_object(value, path)
        method = required(obj, "method", path, decode_str)
        if method not in CLIENT_NOTIFICATION_METHODS:
            raise CodexProtocolError(path, f"unsupported client notification {{method!r}}")
        return cls(method=method)

    def to_wire(self) -> dict[str, JsonValue]:
        """Encode as one line of the client's JSON stream."""
        return {{"method": self.method}}


INITIALIZED = ClientNotification(method="initialized")
"""The notification that ends the handshake, sent after the ``initialize`` response."""

ServerMessage = Notification | ServerRequest | Response | ErrorResponse
ClientMessage = ClientRequest | ClientNotification | Response | ErrorResponse


def _classify(obj: dict[str, object]) -> str:
    """Name the envelope kind from which of ``method``/``id``/``result``/``error`` appear."""
    if "method" in obj:
        return "request" if "id" in obj else "notification"
    if "error" in obj:
        return "error"
    if "result" in obj and "id" in obj:
        return "response"
    raise CodexProtocolError(
        "message", "has none of method, result or error: " + ", ".join(sorted(obj))
    )


def parse_server_message(value: object) -> ServerMessage:
    """Parse one JSON object read from the server's stdout.

    The wire has no ``jsonrpc`` member: a ``method`` with an ``id`` is a server
    request, a ``method`` alone a notification, ``result`` a response and
    ``error`` a failed response. Unknown methods and extra keys are kept or
    ignored rather than rejected; a malformed envelope raises
    ``CodexProtocolError``.
    """
    obj = as_object(value, "message")
    kind = _classify(obj)
    if kind == "request":
        return ServerRequest.from_wire(obj)
    if kind == "notification":
        return Notification.from_wire(obj)
    if kind == "error":
        return ErrorResponse.from_wire(obj)
    return Response.from_wire(obj)


def parse_client_message(value: object) -> ClientMessage:
    """Parse one JSON object read from the client's stdout (for fake servers and tests)."""
    obj = as_object(value, "message")
    kind = _classify(obj)
    if kind == "request":
        return ClientRequest.from_wire(obj)
    if kind == "notification":
        return ClientNotification.from_wire(obj)
    if kind == "error":
        return ErrorResponse.from_wire(obj)
    return Response.from_wire(obj)


def reply(request_id: RequestId, result: ServerRequestResult) -> Response:
    """Build the reply to a server request from its typed result."""
    return Response(id=request_id, result=encode(result))
'''


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def directions(defs: dict[str, Def], messages: dict[str, Any]) -> tuple[set[str], set[str]]:
    """Definitions that travel server-to-client and client-to-server."""
    inbound_roots = [r["result"] for r in messages["client_requests"]]
    inbound_roots += [n["params"] for n in messages["server_notifications"]]
    inbound_roots += [r["params"] for r in messages["server_requests"]]
    outbound_roots = [r["params"] for r in messages["client_requests"]]
    outbound_roots += [r["response"] for r in messages["server_requests"]]
    return reachable(defs, inbound_roots), reachable(defs, outbound_roots)


def generate(schema: dict[str, Any]) -> str:
    """The text of ``protocol.py`` for a pruned schema document."""
    for section in ("client_requests", "server_notifications", "server_requests"):
        if not schema["messages"][section]:
            raise GeneratorError(f"allowlist section {section} is empty")
    defs = Builder(schema["definitions"]).build()
    inbound, outbound = directions(defs, schema["messages"])
    return Emitter(defs, schema, inbound, outbound).emit_module()


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--from-cli", action="store_true", help="re-export the schema from the installed codex"
    )
    source.add_argument("--from-export", type=Path, help="prune an existing export directory")
    parser.add_argument("--codex-version", help="version to record with --from-export")
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH, help="pruned schema file")
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH, help="generated module")
    parser.add_argument(
        "--check", action="store_true", help="exit 1 instead of writing if files would change"
    )
    return parser.parse_args(argv)


def refresh_schema(args: argparse.Namespace) -> str | None:
    """Prune a fresh export into the schema text, or ``None`` to keep the checked-in file."""
    allow = json.loads(ALLOWLIST_PATH.read_text())
    if args.from_export:
        if not args.codex_version:
            raise GeneratorError("--from-export needs --codex-version")
        return dump_schema(prune(args.from_export, allow, args.codex_version))
    if args.from_cli:
        with tempfile.TemporaryDirectory() as tmp:
            version = export_from_cli(Path(tmp))
            return dump_schema(prune(Path(tmp), allow, version))
    return None


def write_or_check(path: Path, text: str, *, check: bool) -> bool:
    """Write ``text`` to ``path`` (or compare when ``check``); True if it was already current."""
    current = path.read_text() if path.exists() else None
    if current == text:
        return True
    if not check:
        path.write_text(text)
    return False


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        fresh = refresh_schema(args)
        schema_text = fresh if fresh is not None else args.schema.read_text()
        module = generate(json.loads(schema_text))
    except GeneratorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    current = True
    if fresh is not None:
        current &= write_or_check(args.schema, fresh, check=args.check)
    current &= write_or_check(args.out, module, check=args.check)
    if args.check and not current:
        print(
            "generated files are out of date; run scripts/generate_codex_protocol.py",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
