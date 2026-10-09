"""Codex ``app-server`` protocol: generated message types and their wire helpers.

``protocol`` is generated from the CLI's own JSON Schema by
``scripts/generate_codex_protocol.py``; ``_wire`` holds the hand-written
decoding helpers it calls. ``transport`` is ``CodexAppServerTransport``, which
speaks that protocol to a long-lived ``codex app-server`` process.
"""

from __future__ import annotations

from ._wire import CodexProtocolError, JsonObject, JsonValue
from .profile import APP_SERVER_PROFILE
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
from .transport import CodexAppServerTransport

__all__ = [
    "APP_SERVER_PROFILE",
    "CODEX_VERSION",
    "INITIALIZED",
    "ClientMessage",
    "ClientNotification",
    "ClientRequest",
    "CodexAppServerTransport",
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
