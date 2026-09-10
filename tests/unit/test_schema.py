"""Output-schema dialect checks and materialization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentshim import SchemaDialect, compact_json, dialect_problems, materialize, normalize

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
        assert len(problems) == 1
        assert "arbitrary object keys" in problems[0]
        assert "#/properties/metrics" in problems[0]
        assert dialect_problems(_MAPPING, SchemaDialect.OPEN) == []

    def test_true_additional_properties_is_arbitrary_keys(self) -> None:
        schema = {"type": "object", "properties": {}, "additionalProperties": True}
        assert dialect_problems(schema, SchemaDialect.STRICT) != []
        assert dialect_problems(schema, SchemaDialect.OPEN) == []

    def test_non_object_root_is_a_problem(self) -> None:
        assert dialect_problems({"type": "array"}, SchemaDialect.OPEN) != []

    @pytest.mark.parametrize("keyword", ["allOf", "oneOf", "not", "if", "patternProperties", "$schema"])
    def test_unsupported_keywords_are_reported(self, keyword: str) -> None:
        schema = {"type": "object", "properties": {"a": {"type": "string"}}, keyword: {}}
        problems = dialect_problems(schema, SchemaDialect.OPEN)
        assert any(keyword in problem for problem in problems)

    def test_a_property_named_like_a_keyword_is_fine(self) -> None:
        schema = {"type": "object", "properties": {"if": {"type": "string"}}, "additionalProperties": False}
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


class TestNormalize:
    def test_objects_are_closed_and_every_property_required(self) -> None:
        schema = {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "integer"}}}
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

    def test_input_is_not_mutated(self) -> None:
        schema = {"type": "object", "properties": {"a": {"type": "string", "default": "x"}}}
        normalize(schema, SchemaDialect.STRICT)
        assert schema["properties"]["a"]["default"] == "x"  # pyright: ignore[reportIndexIssue]


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
