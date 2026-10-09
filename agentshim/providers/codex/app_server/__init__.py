"""Codex ``app-server`` protocol: generated message types and their wire helpers.

``protocol`` is generated from the CLI's own JSON Schema by
``scripts/generate_codex_protocol.py``; ``_wire`` holds the hand-written
decoding helpers it calls. Nothing outside ``providers`` imports this package
yet.
"""

from __future__ import annotations

from ._wire import CodexProtocolError, JsonObject, JsonValue
from .protocol import (
    CODEX_VERSION,
    INITIALIZED,
    ClientMessage,
    ClientNotification,
    ClientRequest,
    ErrorResponse,
    Notification,
    Response,
    RpcError,
    ServerMessage,
    ServerRequest,
    UnknownParams,
    parse_client_message,
    parse_server_message,
    reply,
)

__all__ = [
    "CODEX_VERSION",
    "INITIALIZED",
    "ClientMessage",
    "ClientNotification",
    "ClientRequest",
    "CodexProtocolError",
    "ErrorResponse",
    "JsonObject",
    "JsonValue",
    "Notification",
    "Response",
    "RpcError",
    "ServerMessage",
    "ServerRequest",
    "UnknownParams",
    "parse_client_message",
    "parse_server_message",
    "reply",
]
