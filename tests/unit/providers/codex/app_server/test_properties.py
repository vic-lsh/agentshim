"""Property tests over every generated type: round trips, tolerance, and no stray exceptions."""

from __future__ import annotations

import contextlib
import dataclasses
import json
from typing import Any

import pytest
from agentshim.providers.codex.app_server import CodexProtocolError
from agentshim.providers.codex.app_server import protocol as p
from hypothesis import given, settings
from hypothesis import strategies as st

from .conftest import ALLOWLIST_PATH, SCHEMA_PATH
from .strategies import (
    for_dataclass,
    json_objects,
    json_values,
    protocol_dataclasses,
    request_ids,
)

CLASSES = protocol_dataclasses()
ALLOWLIST = json.loads(ALLOWLIST_PATH.read_text())
SCHEMA = json.loads(SCHEMA_PATH.read_text())
#: Per-class example budget: ~130 classes share the profile's budget (the fuzz profile scales it).
PER_CLASS = max(6, (settings().max_examples) // 25)


def over_wire(wire: Any) -> Any:
    """Push ``wire`` through real JSON text, as the transport does."""
    return json.loads(json.dumps(wire))


def has_both_directions(cls: type) -> bool:
    return hasattr(cls, "from_wire") and hasattr(cls, "to_wire")


def test_every_generated_dataclass_has_both_directions() -> None:
    assert len(CLASSES) > 100
    assert [c.__name__ for c in CLASSES if not has_both_directions(c)] == []


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
class TestEveryGeneratedType:
    def test_from_wire_inverts_to_wire(self, cls: Any) -> None:
        @settings(max_examples=PER_CLASS, deadline=None)
        @given(for_dataclass(cls))
        def check(value: Any) -> None:
            assert cls.from_wire(over_wire(value.to_wire())) == value

        check()

    def test_unknown_keys_are_ignored(self, cls: Any) -> None:
        @settings(max_examples=PER_CLASS, deadline=None)
        @given(for_dataclass(cls), json_values)
        def check(value: Any, extra: Any) -> None:
            noisy = {**over_wire(value.to_wire()), "zzNewServerField": extra}
            assert cls.from_wire(noisy) == value

        check()

    def test_arbitrary_json_is_decoded_or_rejected_with_a_protocol_error(self, cls: Any) -> None:
        @settings(max_examples=PER_CLASS, deadline=None)
        @given(st.one_of(json_values, json_objects))
        def check(wire: Any) -> None:
            with contextlib.suppress(CodexProtocolError):
                cls.from_wire(wire)

        check()


@pytest.mark.parametrize(
    "cls",
    [c for c in CLASSES if "properties" in SCHEMA["definitions"].get(c.__name__, {})],
    ids=lambda c: c.__name__,
)
class TestPresenceMatchesTheSchema:
    def test_none_is_omitted_unless_the_schema_requires_null(
        self, cls: Any, generator: Any
    ) -> None:
        node = SCHEMA["definitions"][cls.__name__]
        required = set(node.get("required", []))
        by_python_name = {generator.snake(key): key for key in node["properties"]}

        @settings(max_examples=PER_CLASS, deadline=None)
        @given(for_dataclass(cls))
        def check(value: Any) -> None:
            wire = value.to_wire()
            for field in dataclasses.fields(cls):
                key = by_python_name[field.name]
                if getattr(value, field.name) is not None:
                    assert key in wire
                elif key in required:
                    assert key in wire
                    assert wire[key] is None
                else:
                    assert key not in wire

        check()


class TestEnvelopes:
    @given(st.data())
    def test_notifications_round_trip_for_every_listed_method(self, data: st.DataObject) -> None:
        entry = data.draw(st.sampled_from(ALLOWLIST["server_notifications"]))
        params = data.draw(for_dataclass(getattr(p, entry["params"])))
        emitted = data.draw(st.none() | st.integers(min_value=0, max_value=2**40))
        message = p.Notification(method=entry["method"], params=params, emitted_at_ms=emitted)
        assert p.parse_server_message(over_wire(message.to_wire())) == message

    @given(st.data())
    def test_server_requests_round_trip_for_every_listed_method(self, data: st.DataObject) -> None:
        entry = data.draw(st.sampled_from(ALLOWLIST["server_requests"]))
        params = data.draw(for_dataclass(getattr(p, entry["params"])))
        message = p.ServerRequest(id=data.draw(request_ids), method=entry["method"], params=params)
        assert p.parse_server_message(over_wire(message.to_wire())) == message

    @given(st.data())
    def test_client_requests_round_trip_and_their_results_decode(self, data: st.DataObject) -> None:
        entry = data.draw(st.sampled_from(ALLOWLIST["client_requests"]))
        params = data.draw(for_dataclass(getattr(p, entry["params"])))
        message = p.ClientRequest(id=data.draw(request_ids), params=params)
        assert p.parse_client_message(over_wire(message.to_wire())) == message
        result = data.draw(for_dataclass(getattr(p, entry["result"])))
        assert params.parse_result(over_wire(result.to_wire())) == result

    @given(st.data())
    def test_replies_to_server_requests_carry_the_typed_result(self, data: st.DataObject) -> None:
        entry = data.draw(st.sampled_from(ALLOWLIST["server_requests"]))
        result = data.draw(for_dataclass(getattr(p, entry["response"])))
        request_id = data.draw(request_ids)
        wire = over_wire(p.reply(request_id, result).to_wire())
        parsed = p.parse_client_message(wire)
        assert isinstance(parsed, p.Response)
        assert parsed.id == request_id
        assert getattr(p, entry["response"]).from_wire(parsed.result) == result

    @given(request_ids, st.integers(), st.text(max_size=10), json_values)
    def test_error_responses_round_trip(
        self, request_id: Any, code: int, message: str, data: Any
    ) -> None:
        error = p.ErrorResponse(
            id=request_id, error=p.RpcError(code=code, message=message, data=data)
        )
        assert p.parse_server_message(over_wire(error.to_wire())) == error

    @pytest.mark.parametrize("parse", [p.parse_server_message, p.parse_client_message])
    @given(st.one_of(json_values, json_objects))
    def test_parsing_arbitrary_json_yields_a_message_or_a_protocol_error(
        self, parse: Any, wire: Any
    ) -> None:
        with contextlib.suppress(CodexProtocolError):
            parse(wire)

    @given(
        st.text(min_size=1, max_size=12).filter(
            lambda m: m not in {n["method"] for n in ALLOWLIST["server_notifications"]}
        ),
        json_values,
    )
    def test_unlisted_notification_methods_keep_their_params_verbatim(
        self, method: str, params: Any
    ) -> None:
        message = p.parse_server_message({"method": method, "params": params})
        assert message == p.Notification(method=method, params=p.UnknownParams(params))
        assert over_wire(message.to_wire()) == {"method": method, "params": params}
