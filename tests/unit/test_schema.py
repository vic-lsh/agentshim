"""Output-schema dialect checks and materialization."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, cast

import pytest
from agentshim import SchemaDialect, compact_json, dialect_problems, materialize, normalize
from hypothesis import given
from hypothesis import strategies as st

_SIMPLE = {
    "type": "object",
    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    "required": ["a", "b"],
    "additionalProperties": False,
}

# What a ``dict[str, float]`` field produces: arbitrary keys, typed values.
_MAPPING = {
    "type": "object",
    "properties": {"metrics": {"type": "object", "additionalProperties": {"type": "number"}}},
    "required": ["metrics"],
    "additionalProperties": False,
}


class TestDialectProblems:
    def test_closed_schema_is_accepted_by_both_dialects(self) -> None:
        assert dialect_problems(_SIMPLE, SchemaDialect.STRICT) == []
        assert dialect_problems(_SIMPLE, SchemaDialect.OPEN) == []

    def test_arbitrary_keys_are_rejected_by_strict_only(self) -> None:
        problems = dialect_problems(_MAPPING, SchemaDialect.STRICT)
        assert "#/properties/metrics allows arbitrary object keys" in problems
        assert dialect_problems(_MAPPING, SchemaDialect.OPEN) == []

    def test_true_additional_properties_is_arbitrary_keys(self) -> None:
        schema = {"type": "object", "properties": {}, "additionalProperties": True}
        assert dialect_problems(schema, SchemaDialect.STRICT) != []
        assert dialect_problems(schema, SchemaDialect.OPEN) == []

    def test_non_object_root_is_a_problem(self) -> None:
        assert dialect_problems({"type": "array"}, SchemaDialect.OPEN) != []

    @pytest.mark.parametrize("keyword", ["allOf", "oneOf", "not", "if", "patternProperties"])
    def test_unsupported_keywords_are_reported(self, keyword: str) -> None:
        schema = {"type": "object", "properties": {"a": {"type": "string"}}, keyword: {}}
        problems = dialect_problems(schema, SchemaDialect.OPEN)
        assert any(keyword in problem for problem in problems)

    @pytest.mark.parametrize("keyword", ["$schema", "$id"])
    def test_document_metadata_is_strict_only(self, keyword: str) -> None:
        """Codex refuses these; an open-dialect CLI just ignores them."""
        schema = {**_SIMPLE, keyword: "https://example.invalid/thing"}
        assert dialect_problems(schema, SchemaDialect.OPEN) == []
        assert any(keyword in problem for problem in dialect_problems(schema, SchemaDialect.STRICT))

    @pytest.mark.parametrize("keyword", ["title", "description", "examples"])
    def test_annotations_are_accepted_by_both_dialects(self, keyword: str) -> None:
        schema = {**_SIMPLE, keyword: "note"}
        assert dialect_problems(schema, SchemaDialect.OPEN) == []
        assert dialect_problems(schema, SchemaDialect.STRICT) == []

    def test_a_property_named_like_a_keyword_is_fine(self) -> None:
        schema = {
            "type": "object",
            "properties": {"if": {"type": "string"}},
            "additionalProperties": False,
        }
        assert dialect_problems(schema, SchemaDialect.OPEN) == []

    def test_non_local_ref_is_reported(self) -> None:
        schema = {"type": "object", "properties": {"a": {"$ref": "https://example.com/x.json"}}}
        problems = dialect_problems(schema, SchemaDialect.OPEN)
        assert any("$ref" in problem for problem in problems)

    def test_local_ref_is_accepted(self) -> None:
        schema = {
            "type": "object",
            "properties": {"a": {"$ref": "#/$defs/Inner"}},
            "additionalProperties": False,
            "$defs": {"Inner": {"type": "object", "properties": {}, "additionalProperties": False}},
        }
        assert dialect_problems(schema, SchemaDialect.OPEN) == []

    def test_defs_are_traversed(self) -> None:
        schema = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
            "$defs": {"Inner": {"type": "object", "additionalProperties": {"type": "string"}}},
        }
        assert dialect_problems(schema, SchemaDialect.STRICT) != []

    def test_check_does_not_mutate_the_input(self) -> None:
        schema = {"type": "object", "properties": {"a": {"type": "string", "default": "x"}}}
        before = json.dumps(schema, sort_keys=True)
        dialect_problems(schema, SchemaDialect.STRICT)
        assert json.dumps(schema, sort_keys=True) == before


def _closed(properties: dict[str, Any], **extra: object) -> dict[str, Any]:
    """A strict-valid object node over *properties*, every one required."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
        **extra,
    }


# The schema that failed in production: ``no_change_reason`` is optional.
_OPTIONAL_REASON = {
    "type": "object",
    "properties": {
        "changed": {"type": "boolean"},
        "no_change_reason": {"type": "string"},
    },
    "required": ["changed"],
    "additionalProperties": False,
}


class TestStrictRequired:
    """OpenAI strict structured outputs: every property is required."""

    def test_an_optional_property_is_rejected_by_strict_only(self) -> None:
        problems = dialect_problems(_OPTIONAL_REASON, SchemaDialect.STRICT)
        assert len(problems) == 1
        assert problems[0].startswith("#/properties/no_change_reason ")
        assert "nullable" in problems[0]
        assert dialect_problems(_OPTIONAL_REASON, SchemaDialect.OPEN) == []

    def test_a_missing_required_array_makes_every_property_optional(self) -> None:
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        schema["additionalProperties"] = False
        problems = dialect_problems(schema, SchemaDialect.STRICT)
        assert [p.split()[0] for p in problems] == ["#/properties/a"]

    @pytest.mark.parametrize(
        "nullable",
        [
            {"type": ["string", "null"]},
            {"anyOf": [{"type": "string"}, {"type": "null"}]},
        ],
    )
    def test_an_optional_value_made_nullable_is_accepted(self, nullable: dict[str, Any]) -> None:
        schema = _closed({"changed": {"type": "boolean"}, "no_change_reason": nullable})
        assert dialect_problems(schema, SchemaDialect.STRICT) == []

    def test_a_required_name_outside_properties_is_rejected(self) -> None:
        schema = _closed({"a": {"type": "string"}})
        schema["required"] = ["a", "ghost"]
        problems = dialect_problems(schema, SchemaDialect.STRICT)
        assert problems == ["#/required names 'ghost', which is not in 'properties'"]

    def test_required_must_be_a_list_of_names(self) -> None:
        schema = _closed({"a": {"type": "string"}})
        schema["required"] = "a"
        problems = dialect_problems(schema, SchemaDialect.STRICT)
        assert problems == ["#/required must be an array of property names"]

    def test_an_object_with_empty_properties_is_accepted(self) -> None:
        schema = _closed({"empty": _closed({})})
        assert dialect_problems(schema, SchemaDialect.STRICT) == []

    def test_an_object_without_properties_is_rejected_by_strict_only(self) -> None:
        """The API answers "object schema missing properties", even for an empty object."""
        schema = _closed({"empty": {"type": "object", "additionalProperties": False}})
        problems = dialect_problems(schema, SchemaDialect.STRICT)
        assert problems == ["#/properties/empty must declare properties (use {} for none)"]
        assert dialect_problems(schema, SchemaDialect.OPEN) == []

    def test_a_nested_optional_property_is_reported_by_pointer(self) -> None:
        inner = _closed({"x": {"type": "integer"}, "y": {"type": "integer"}})
        inner["required"] = ["x"]
        schema = _closed({"outer": _closed({"inner": inner})})
        problems = dialect_problems(schema, SchemaDialect.STRICT)
        assert [p.split()[0] for p in problems] == [
            "#/properties/outer/properties/inner/properties/y"
        ]

    def test_an_optional_property_under_items_is_reported(self) -> None:
        item = _closed({"x": {"type": "integer"}, "y": {"type": "integer"}})
        item["required"] = ["y"]
        schema = _closed({"rows": {"type": "array", "items": item}})
        problems = dialect_problems(schema, SchemaDialect.STRICT)
        assert [p.split()[0] for p in problems] == ["#/properties/rows/items/properties/x"]

    def test_an_optional_property_in_an_any_of_branch_is_reported(self) -> None:
        branch = _closed({"x": {"type": "integer"}})
        branch["required"] = []
        schema = _closed({"choice": {"anyOf": [{"type": "null"}, branch]}})
        problems = dialect_problems(schema, SchemaDialect.STRICT)
        assert [p.split()[0] for p in problems] == ["#/properties/choice/anyOf/1/properties/x"]

    @pytest.mark.parametrize("defs_key", ["$defs", "definitions"])
    def test_an_optional_property_in_a_ref_target_is_reported(self, defs_key: str) -> None:
        target = _closed({"x": {"type": "integer"}})
        target["required"] = []
        schema = _closed({"a": {"$ref": f"#/{defs_key}/Inner"}}, **{defs_key: {"Inner": target}})
        problems = dialect_problems(schema, SchemaDialect.STRICT)
        assert [p.split()[0] for p in problems] == [f"#/{defs_key}/Inner/properties/x"]

    def test_a_pointer_escapes_slash_and_tilde(self) -> None:
        schema = _closed({"a/b": {"type": "string"}, "c~d": {"type": "string"}})
        schema["required"] = []
        problems = dialect_problems(schema, SchemaDialect.STRICT)
        assert [p.split()[0] for p in problems] == ["#/properties/a~1b", "#/properties/c~0d"]


class TestStrictObjects:
    def test_a_missing_additional_properties_is_rejected_by_strict_only(self) -> None:
        schema = _closed({"inner": {"type": "object", "properties": {}, "required": []}})
        problems = dialect_problems(schema, SchemaDialect.STRICT)
        assert problems == ["#/properties/inner must set additionalProperties: false"]
        assert dialect_problems(schema, SchemaDialect.OPEN) == []

    def test_a_nullable_object_is_checked_as_an_object(self) -> None:
        schema = _closed({"maybe": {"type": ["object", "null"], "properties": {}}})
        problems = dialect_problems(schema, SchemaDialect.STRICT)
        assert problems == ["#/properties/maybe must set additionalProperties: false"]

    def test_a_root_any_of_is_rejected_by_strict_only(self) -> None:
        schema = _closed({}, anyOf=[_closed({"a": {"type": "string"}})])
        assert "# root must not use 'anyOf'" in dialect_problems(schema, SchemaDialect.STRICT)
        assert dialect_problems(schema, SchemaDialect.OPEN) == []

    def test_root_recursion_is_accepted(self) -> None:
        schema = _closed({"children": {"type": "array", "items": {"$ref": "#"}}})
        assert dialect_problems(schema, SchemaDialect.STRICT) == []
        assert dialect_problems(schema, SchemaDialect.OPEN) == []

    def test_a_dangling_ref_is_rejected_by_strict(self) -> None:
        schema = _closed({"a": {"$ref": "#/$defs/Missing"}}, **{"$defs": {}})
        problems = dialect_problems(schema, SchemaDialect.STRICT)
        assert problems == ["#/properties/a has a $ref '#/$defs/Missing' that resolves to nothing"]

    def test_a_ref_into_an_array_resolves_by_index(self) -> None:
        schema = _closed(
            {
                "a": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "b": {"$ref": "#/properties/a/anyOf/0"},
            }
        )
        assert dialect_problems(schema, SchemaDialect.STRICT) == []
        schema["properties"]["b"] = {"$ref": "#/properties/a/anyOf/2"}
        assert len(dialect_problems(schema, SchemaDialect.STRICT)) == 1

    def test_enum_and_const_values_are_not_schemas(self) -> None:
        schema = _closed(
            {
                "e": {"enum": [{"$ref": "https://example.invalid"}, None]},
                "c": {"const": {"type": "object"}},
            }
        )
        assert dialect_problems(schema, SchemaDialect.STRICT) == []


# Field names include keyword look-alikes and pointer metacharacters.
_names = st.one_of(
    st.sampled_from(["if", "required", "properties", "$ref", "a/b", "c~d", "type"]),
    st.text(alphabet="abcxyz_", min_size=1, max_size=6),
)
_leaves = st.one_of(
    st.sampled_from(
        [
            {"type": "string"},
            {"type": "integer"},
            {"type": "number", "minimum": 0},
            {"type": "boolean"},
            {"type": ["string", "null"]},
            {"enum": ["a", "b", None]},
            {"$ref": "#/$defs/Leaf"},
            {"$ref": "#"},
        ]
    ),
)


def _extend(children: st.SearchStrategy[dict[str, Any]]) -> st.SearchStrategy[dict[str, Any]]:
    objects = st.builds(
        lambda properties, nullable: _closed(
            properties, **({"type": ["object", "null"]} if nullable else {})
        ),
        st.dictionaries(_names, children, max_size=4),
        st.booleans(),
    )
    arrays = st.builds(lambda item: {"type": "array", "items": item}, children)
    unions = st.builds(
        lambda options: {"anyOf": [*options, {"type": "null"}]},
        st.lists(children, min_size=1, max_size=3),
    )
    return st.one_of(objects, arrays, unions)


_subschemas = st.recursive(_leaves, _extend, max_leaves=12)
_strict_schemas = st.builds(
    lambda properties: _closed(
        properties, **{"$defs": {"Leaf": _closed({"value": {"type": "string"}})}}
    ),
    st.dictionaries(_names, _subschemas, min_size=1, max_size=4),
)


def _escape(name: str) -> str:
    return name.replace("~", "~0").replace("/", "~1")


def _object_nodes(node: object, path: tuple[str | int, ...] = ()) -> list[tuple[str | int, ...]]:
    """Paths to every object node, found by the schema structure the generator emits."""
    found: list[tuple[str | int, ...]] = []
    if isinstance(node, list):
        for index, value in enumerate(cast("list[object]", node)):
            found += _object_nodes(value, (*path, index))
        return found
    if not isinstance(node, dict):
        return found
    mapping = cast("dict[str, Any]", node)
    if "additionalProperties" in mapping:
        found.append(path)
    for key in ("properties", "$defs"):
        for name, child in cast("dict[str, Any]", mapping.get(key, {})).items():
            found += _object_nodes(child, (*path, key, name))
    for key in ("items", "anyOf"):
        if key in mapping:
            found += _object_nodes(mapping[key], (*path, key))
    return found


def _at(root: dict[str, Any], path: tuple[str | int, ...]) -> dict[str, Any]:
    node: Any = root
    for token in path:
        node = node[token]
    return cast("dict[str, Any]", node)


def _pointer(path: tuple[str | int, ...]) -> str:
    return "".join(f"/{_escape(str(token))}" for token in path)


class TestStrictProperties:
    @given(_strict_schemas)
    def test_a_schema_built_to_the_strict_rules_is_accepted(self, schema: dict[str, Any]) -> None:
        assert dialect_problems(schema, SchemaDialect.STRICT) == []
        assert dialect_problems(schema, SchemaDialect.OPEN) == []

    @given(_strict_schemas)
    def test_dropping_any_one_required_name_is_rejected(self, schema: dict[str, Any]) -> None:
        for path in _object_nodes(schema):
            for name in _at(schema, path)["required"]:
                mutated = copy.deepcopy(schema)
                _at(mutated, path)["required"].remove(name)

                problems = dialect_problems(mutated, SchemaDialect.STRICT)

                pointer = f"#{_pointer((*path, 'properties', name))}"
                assert [p.split()[0] for p in problems] == [pointer]
                assert dialect_problems(mutated, SchemaDialect.OPEN) == []

    @given(_strict_schemas)
    def test_normalize_keeps_a_strict_schema_strict(self, schema: dict[str, Any]) -> None:
        assert dialect_problems(normalize(schema, SchemaDialect.STRICT), SchemaDialect.STRICT) == []


class TestNormalize:
    def test_an_object_without_properties_gets_an_empty_one(self) -> None:
        schema = {"type": "object", "properties": {"e": {"type": ["object", "null"]}}}
        result = normalize(schema, SchemaDialect.STRICT)
        assert result["properties"]["e"]["properties"] == {}
        assert dialect_problems(result, SchemaDialect.STRICT) == []

    def test_objects_are_closed_and_every_property_required(self) -> None:
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
        }
        result = normalize(schema, SchemaDialect.STRICT)
        assert result["additionalProperties"] is False
        assert result["required"] == ["a", "b"]

    def test_defaults_are_dropped(self) -> None:
        schema = {"type": "object", "properties": {"a": {"type": "string", "default": "x"}}}
        result = normalize(schema, SchemaDialect.STRICT)
        assert "default" not in result["properties"]["a"]

    def test_ref_siblings_are_stripped(self) -> None:
        schema = {
            "type": "object",
            "properties": {"a": {"$ref": "#/$defs/Inner", "description": "doc"}},
            "$defs": {"Inner": {"type": "object", "properties": {}}},
        }
        result = normalize(schema, SchemaDialect.STRICT)
        assert result["properties"]["a"] == {"$ref": "#/$defs/Inner"}

    def test_open_dialect_keeps_a_schema_valued_additional_properties(self) -> None:
        result = normalize(_MAPPING, SchemaDialect.OPEN)
        assert result["properties"]["metrics"]["additionalProperties"] == {"type": "number"}

    def test_strict_dialect_keeps_an_open_map_and_reports_it(self) -> None:
        """Closing ``dict[str, float]`` would only admit ``{}``; it stays a problem."""
        result = normalize(_MAPPING, SchemaDialect.STRICT)
        assert result["properties"]["metrics"]["additionalProperties"] == {"type": "number"}
        assert dialect_problems(result, SchemaDialect.STRICT) == [
            "#/properties/metrics allows arbitrary object keys"
        ]

    def test_input_is_not_mutated(self) -> None:
        schema = {"type": "object", "properties": {"a": {"type": "string", "default": "x"}}}
        normalize(schema, SchemaDialect.STRICT)
        assert schema["properties"]["a"]["default"] == "x"  # pyright: ignore[reportIndexIssue]

    @pytest.mark.parametrize("dialect", [SchemaDialect.OPEN, SchemaDialect.STRICT])
    def test_metadata_is_dropped(self, dialect: SchemaDialect) -> None:
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://example.invalid/report",
            "title": "Report",
            "description": "what the agent found",
            "examples": [{"a": 1}],
            "type": "object",
            "properties": {"a": {"type": "integer", "title": "A", "description": "a count"}},
            "additionalProperties": False,
        }

        result = normalize(schema, dialect)

        assert result == {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "additionalProperties": False,
            "required": ["a"],
        }

    def test_normalize_repairs_what_the_strict_dialect_rejects(self) -> None:
        """``dialect_problems`` reports metadata under STRICT; ``normalize`` fixes it."""
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://example.invalid/report",
            "type": "object",
            "properties": {"a": {"type": "integer"}},
        }
        assert dialect_problems(schema, SchemaDialect.STRICT) != []

        assert dialect_problems(normalize(schema, SchemaDialect.STRICT), SchemaDialect.STRICT) == []

    def test_a_property_named_like_metadata_survives(self) -> None:
        schema = {
            "type": "object",
            "properties": {"title": {"type": "string"}, "description": {"type": "string"}},
            "additionalProperties": False,
        }

        result = normalize(schema, SchemaDialect.STRICT)

        assert sorted(result["properties"]) == ["description", "title"]
        assert result["required"] == ["title", "description"]


class TestCompactJson:
    def test_output_has_no_whitespace(self) -> None:
        rendered = compact_json(_SIMPLE)
        assert " " not in rendered
        assert json.loads(rendered) == _SIMPLE


class TestMaterialize:
    def test_writes_a_content_addressed_file(self, tmp_path: Path) -> None:
        path = materialize(_SIMPLE, tmp_path)
        assert path.parent == tmp_path
        assert path.suffix == ".json"
        assert json.loads(path.read_text(encoding="utf-8")) == _SIMPLE

    def test_same_schema_reuses_one_file(self, tmp_path: Path) -> None:
        first = materialize(_SIMPLE, tmp_path)
        second = materialize(dict(_SIMPLE), tmp_path)
        assert first == second
        assert list(tmp_path.iterdir()) == [first]

    def test_different_schemas_get_different_names(self, tmp_path: Path) -> None:
        assert materialize(_SIMPLE, tmp_path) != materialize(_MAPPING, tmp_path)

    def test_creates_missing_directories(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "dir"
        assert materialize(_SIMPLE, target).exists()

    def test_leaves_no_temporary_files_behind(self, tmp_path: Path) -> None:
        materialize(_SIMPLE, tmp_path)
        assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []
