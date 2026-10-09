"""The generator: reproducibility, pruning and the checked-in outputs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from .conftest import ALLOWLIST_PATH, PROTOCOL_PATH, SCHEMA_PATH

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType


def _schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text())


def _write_export(directory: Path, definitions: dict[str, Any], methods: dict[str, Any]) -> None:
    """A tiny stand-in for ``codex app-server generate-json-schema`` output."""

    def union(entries: dict[str, str | None]) -> dict[str, Any]:
        return {
            "oneOf": [
                {
                    "properties": {
                        "method": {"enum": [method]},
                        **({"params": {"$ref": f"#/definitions/v2/{ref}"}} if ref else {}),
                    }
                }
                for method, ref in entries.items()
            ]
        }

    bundle = {
        "definitions": {
            "ClientRequest": union(methods.get("client", {})),
            "ServerNotification": union(methods.get("notification", {})),
            "ServerRequest": union(methods.get("server", {})),
            "v2": definitions,
        }
    }
    (directory / "codex_app_server_protocol.schemas.json").write_text(json.dumps(bundle))


PING_ALLOWLIST: dict[str, Any] = {
    "client_requests": [{"method": "ping", "params": "PingParams", "result": "PingResult"}],
    "client_notifications": [],
    "server_notifications": [{"method": "pong", "params": "PongNotification"}],
    "server_requests": [{"method": "ask", "params": "AskParams", "response": "AskResponse"}],
}
METHODS = {
    "client": {"ping": "PingParams"},
    "notification": {"pong": "PongNotification"},
    "server": {"ask": "AskParams"},
}
PING_DEFINITIONS: dict[str, Any] = {
    "PingParams": {
        "type": "object",
        "description": "prose is dropped",
        "properties": {"mode": {"$ref": "#/definitions/v2/PingMode"}},
    },
    "PingMode": {"type": "string", "enum": ["fast", "slow"]},
    "PingResult": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
    "PongNotification": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
    "AskParams": {"type": "object", "properties": {"q": {"type": "string"}}},
    "AskResponse": {"type": "object"},
    "Unrelated": {"type": "object", "properties": {"x": {"type": "string"}}},
}


class TestCheckedInOutputs:
    def test_protocol_module_is_reproduced_byte_for_byte(self, generator: ModuleType) -> None:
        regenerated = generator.generate(_schema())
        assert regenerated == PROTOCOL_PATH.read_text()

    def test_check_mode_reports_no_drift(self, generator: ModuleType) -> None:
        assert generator.main(["--check"]) == 0

    def test_pruned_schema_embeds_the_allowlist_and_version(self) -> None:
        schema = _schema()
        assert schema["messages"] == json.loads(ALLOWLIST_PATH.read_text())
        assert schema["codex_version"].count(".") == 2

    def test_pruned_schema_is_closed_and_has_no_unreachable_types(
        self, generator: ModuleType
    ) -> None:
        schema = _schema()
        definitions = schema["definitions"]
        referenced: set[str] = set()
        generator.collect_refs(definitions, referenced)
        roots = set(generator.allowlist_roots(schema["messages"]))
        assert referenced <= set(definitions), "a reference points outside the pruned schema"
        assert roots <= set(definitions)
        reachable: set[str] = set()
        stack = list(roots)
        while stack:
            name = stack.pop()
            if name not in reachable:
                reachable.add(name)
                refs: set[str] = set()
                generator.collect_refs(definitions[name], refs)
                stack.extend(refs)
        assert reachable == set(definitions)


class TestPrune:
    def test_keeps_only_the_closure_of_the_allowlist(
        self, generator: ModuleType, tmp_path: Path
    ) -> None:
        _write_export(tmp_path, PING_DEFINITIONS, METHODS)
        pruned = generator.prune(tmp_path, PING_ALLOWLIST, "9.9.9")
        assert set(pruned["definitions"]) == {
            "PingParams",
            "PingMode",
            "PingResult",
            "PongNotification",
            "AskParams",
            "AskResponse",
        }
        assert pruned["codex_version"] == "9.9.9"

    def test_rewrites_refs_and_drops_prose(self, generator: ModuleType, tmp_path: Path) -> None:
        _write_export(tmp_path, PING_DEFINITIONS, METHODS)
        params = generator.prune(tmp_path, PING_ALLOWLIST, "9.9.9")["definitions"]["PingParams"]
        assert params == {
            "type": "object",
            "properties": {"mode": {"$ref": "#/definitions/PingMode"}},
        }

    def test_is_deterministic(self, generator: ModuleType, tmp_path: Path) -> None:
        _write_export(tmp_path, PING_DEFINITIONS, METHODS)
        first = generator.dump_schema(generator.prune(tmp_path, PING_ALLOWLIST, "1.0.0"))
        second = generator.dump_schema(generator.prune(tmp_path, PING_ALLOWLIST, "1.0.0"))
        assert first == second

    def test_method_whose_params_type_changed_is_reported(
        self, generator: ModuleType, tmp_path: Path
    ) -> None:
        _write_export(tmp_path, PING_DEFINITIONS, {**METHODS, "client": {"ping": "OtherParams"}})
        with pytest.raises(generator.GeneratorError, match="ping"):
            generator.prune(tmp_path, PING_ALLOWLIST, "9.9.9")

    def test_method_removed_from_the_export_is_reported(
        self, generator: ModuleType, tmp_path: Path
    ) -> None:
        _write_export(tmp_path, PING_DEFINITIONS, {**METHODS, "client": {}})
        with pytest.raises(generator.GeneratorError, match="absent"):
            generator.prune(tmp_path, PING_ALLOWLIST, "9.9.9")

    def test_allowlisted_type_missing_from_the_export_is_reported(
        self, generator: ModuleType, tmp_path: Path
    ) -> None:
        definitions = {k: v for k, v in PING_DEFINITIONS.items() if k != "PingResult"}
        _write_export(tmp_path, definitions, METHODS)
        with pytest.raises(generator.GeneratorError, match="PingResult"):
            generator.prune(tmp_path, PING_ALLOWLIST, "9.9.9")


class TestGenerate:
    def _generate(self, generator: ModuleType, definitions: dict[str, Any]) -> str:
        wanted = {"PingParams", "PingResult", "PongNotification", "AskParams", "AskResponse"}
        base = {k: v for k, v in PING_DEFINITIONS.items() if k in wanted | {"PingMode"}}
        schema = {
            "codex_version": "9.9.9",
            "messages": PING_ALLOWLIST,
            "definitions": {**base, **definitions},
        }
        return generator.generate(schema)

    def test_minimal_schema_produces_compilable_code_with_the_version(
        self, generator: ModuleType
    ) -> None:
        text = self._generate(generator, {})
        compile(text, "protocol.py", "exec")
        assert "codex-cli 9.9.9" in text
        assert "METHOD: ClassVar[str] = 'ping'" in text

    def test_unsupported_schema_form_names_the_offending_type(self, generator: ModuleType) -> None:
        bad = {
            "type": "object",
            "properties": {"mode": {"allOf": [{"type": "string"}, {"type": "integer"}]}},
        }
        with pytest.raises(generator.GeneratorError, match="PingParamsMode"):
            self._generate(generator, {"PingParams": bad})

    def test_schema_type_colliding_with_an_envelope_is_rejected(
        self, generator: ModuleType
    ) -> None:
        with pytest.raises(generator.GeneratorError, match="Notification"):
            self._generate(generator, {"Notification": {"type": "object"}})

    @pytest.mark.parametrize(
        "section", ["client_requests", "server_notifications", "server_requests"]
    )
    def test_empty_allowlist_section_is_rejected(self, generator: ModuleType, section: str) -> None:
        schema = {
            "codex_version": "9.9.9",
            "messages": {**PING_ALLOWLIST, section: []},
            "definitions": {},
        }
        with pytest.raises(generator.GeneratorError, match=section):
            generator.generate(schema)
