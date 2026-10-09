"""What Codex can do over ``app-server``, as opposed to one ``codex exec`` per turn."""

from __future__ import annotations

from dataclasses import replace

from agentshim.core.permissions import NativeMode
from agentshim.core.profile import ConfigScope, OutputSchemaStyle
from agentshim.providers.codex.provider import PROFILE

#: The Codex profile for the long-lived transport.
#:
#: All three native permission modes are enforced here (the one-shot provider
#: keeps its bypass-only declaration). The output schema is sent inline with
#: the turn, not as a file. ``ConfigScope.PROJECT`` is not offered: it needs a
#: dedicated ``CODEX_HOME`` prepared outside the process, which this transport
#: does not manage. Renewal, resume and the skill and MCP scopes are Codex's own.
APP_SERVER_PROFILE = replace(
    PROFILE,
    output_schema=OutputSchemaStyle.INLINE_JSON,
    config_scopes=frozenset({ConfigScope.ALL}),
    config_home_files=(),
    native_permission_modes=frozenset(NativeMode),
    supports_steer=True,
)
