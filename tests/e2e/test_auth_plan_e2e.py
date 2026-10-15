"""A real Claude Code binary logs in with exactly the credential a plan chose.

The home holds a planted subscription login and the host environment a
stale base URL, so a CLI that saw anything besides the plan would use the
wrong credential or endpoint. The endpoint is a local server that records the
headers of a request and answers 401; the test stops the CLI once the first
request arrives, so no model is called, no credential is real, and the CLI's
own retries never run. A host plan is checked the other way round: the host
CLI's first request, from the host's whole home and environment, carries the
same credential as a copy's that got only the plan. Skipped unless ``claude``
is on PATH.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING

import pytest
from agentshim import (
    AuthPolicy,
    CredentialKind,
    HostAuthGap,
    HostAuthUnplannableError,
    get_provider,
    host_auth_plan,
    plan_auth,
    probe_provider,
)

from tests.e2e.conftest import requires_binary

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

pytestmark = [pytest.mark.e2e, requires_binary("claude")]

_SUBSCRIPTION_LOGIN = {
    "claudeAiOauth": {
        "accessToken": "sk-ant-oat01-planted-subscription",
        "refreshToken": "sk-ant-ort01-planted-subscription",
        "expiresAt": 9_999_999_999_999,
        "scopes": ["user:inference"],
        "subscriptionType": "max",
    }
}
#: The first request arrives within seconds; this only guards a hang.
_HANG_GUARD_S = 120


class _Recorder(BaseHTTPRequestHandler):
    seen: list[dict[str, str]]
    arrived: threading.Event

    def do_POST(self) -> None:
        if not self.path.startswith("/v1/messages"):
            self.send_response(404)
            self.end_headers()
            return
        self.seen.append({key.lower(): value for key, value in self.headers.items()})
        self.arrived.set()
        body = json.dumps({"type": "error", "error": {"type": "authentication_error"}})
        self.send_response(401)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode())

    def do_HEAD(self) -> None:
        # The CLI checks the endpoint is reachable (``HEAD /api/hello``) first.
        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        del format, args


@pytest.fixture
def endpoint() -> Iterator[tuple[str, list[dict[str, str]], threading.Event]]:
    seen: list[dict[str, str]] = []
    arrived = threading.Event()
    handler = type("Recorder", (_Recorder,), {"seen": seen, "arrived": arrived})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", seen, arrived
    finally:
        server.shutdown()
        thread.join(_HANG_GUARD_S)


def _home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    (home / ".claude" / ".credentials.json").write_text(json.dumps(_SUBSCRIPTION_LOGIN))
    return home


def _first_request(env: dict[str, str], arrived: threading.Event) -> None:
    """Start a turn, wait for its first request to reach the endpoint, then stop the CLI."""
    process = subprocess.Popen(
        ["claude", "-p", "hi", "--max-turns", "1", "--output-format", "json"],  # noqa: S607
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert arrived.wait(_HANG_GUARD_S), "the CLI never reached the planned endpoint"
    finally:
        process.kill()
        process.wait(_HANG_GUARD_S)


@pytest.mark.parametrize(
    ("variable", "header", "expected"),
    [
        ("ANTHROPIC_API_KEY", "x-api-key", "sk-ant-api-planned"),
        ("ANTHROPIC_AUTH_TOKEN", "authorization", "Bearer sk-ant-api-planned"),
    ],
)
def test_the_cli_sends_only_the_planned_credential_to_the_planned_endpoint(
    tmp_path: Path,
    endpoint: tuple[str, list[dict[str, str]], threading.Event],
    variable: str,
    header: str,
    expected: str,
) -> None:
    url, seen, arrived = endpoint
    home = _home(tmp_path)
    host_env = {
        variable: "sk-ant-api-planned",
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:9/stale",
        "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-planted-env-subscription",
    }
    policy = AuthPolicy(frozenset({CredentialKind.API_KEY}), url)
    plan = plan_auth(get_provider("claude").profile, policy, env=host_env, home=home)
    assert plan.files == ()

    # The CLI sees the plan in a home with no login: what the caller delivers.
    agent_home = tmp_path / "agent-home"
    agent_home.mkdir()
    env = {"PATH": os.environ["PATH"], "HOME": str(agent_home), **plan.env}
    status = probe_provider("claude", env=env)
    assert status.credential is CredentialKind.API_KEY
    _first_request(env, arrived)

    for headers in seen:
        assert headers.get(header) == expected
        assert "planted" not in json.dumps(headers)


def test_a_subscription_plan_hands_over_the_login_file(tmp_path: Path) -> None:
    home = _home(tmp_path)
    policy = AuthPolicy(frozenset({CredentialKind.SUBSCRIPTION}))
    plan = plan_auth(
        get_provider("claude").profile,
        policy,
        env={"ANTHROPIC_API_KEY": "sk-ant-api-withheld"},
        home=home,
    )
    assert plan.env == {}
    agent_home = tmp_path / "agent-home"
    for staged in plan.files:
        target = agent_home / staged.home_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(staged.host_path.read_bytes())
    status = probe_provider("claude", env={"PATH": os.environ["PATH"], "HOME": str(agent_home)})
    assert status.credential is CredentialKind.SUBSCRIPTION


def _credential_headers(headers: dict[str, str]) -> dict[str, str]:
    return {key: headers[key] for key in ("authorization", "x-api-key") if key in headers}


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        # A subscription token at a custom endpoint: the CLI sends it there.
        ("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-host-subscription"),
        ("ANTHROPIC_API_KEY", "sk-ant-api-host-key"),
    ],
)
def test_a_copy_given_the_host_plan_sends_what_the_host_cli_sends(
    tmp_path: Path,
    endpoint: tuple[str, list[dict[str, str]], threading.Event],
    variable: str,
    value: str,
) -> None:
    url, seen, arrived = endpoint
    home = _home(tmp_path)
    (home / ".claude" / "settings.json").write_text(json.dumps({"theme": "dark"}))
    host_env = {"PATH": os.environ["PATH"], "HOME": str(home), variable: value}
    host_env["ANTHROPIC_BASE_URL"] = url
    _first_request(host_env, arrived)
    host_sent = _credential_headers(seen[0])
    assert value in json.dumps(host_sent)

    seen.clear()
    arrived.clear()
    plan = host_auth_plan(get_provider("claude").profile, env=host_env, home=home)
    assert plan.files == ()
    agent_home = tmp_path / "agent-home"
    agent_home.mkdir()
    _first_request({"PATH": os.environ["PATH"], "HOME": str(agent_home), **plan.env}, arrived)
    assert _credential_headers(seen[0]) == host_sent


def test_an_api_key_helper_outranks_a_subscription_token(
    tmp_path: Path, endpoint: tuple[str, list[dict[str, str]], threading.Event]
) -> None:
    # Pins the declared precedence to the real CLI: the helper wins, so no plan
    # of the token alone would reproduce this host.
    url, seen, arrived = endpoint
    home = _home(tmp_path)
    helper = {"apiKeyHelper": "echo sk-ant-api-from-helper"}
    (home / ".claude" / "settings.json").write_text(json.dumps(helper))
    host_env = {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-host-subscription",
        "ANTHROPIC_BASE_URL": url,
    }
    _first_request(host_env, arrived)
    sent = json.dumps(_credential_headers(seen[0]))
    assert "sk-ant-api-from-helper" in sent
    assert "host-subscription" not in sent
    with pytest.raises(HostAuthUnplannableError) as raised:
        host_auth_plan(get_provider("claude").profile, env=host_env, home=home)
    assert raised.value.reason is HostAuthGap.EXTERNAL_CREDENTIAL
