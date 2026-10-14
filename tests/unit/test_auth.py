"""Credential policy: plan_auth hands a CLI exactly one allowed credential."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from agentshim import (
    AuthPolicy,
    AuthUnavailableError,
    CredentialKind,
    EnvCredential,
    FileCredential,
    ProviderProfile,
    plan_auth,
)
from agentshim.core.status import ProbeOutput
from agentshim.providers import get_provider, provider_names
from agentshim.providers.claude.probe import read_auth as read_claude_auth
from agentshim.providers.codex.probe import read_auth as read_codex_auth
from hypothesis import given, settings
from hypothesis import strategies as st

PROFILES = [get_provider(name).profile for name in provider_names()]
ENFORCEABLE = [profile for profile in PROFILES if profile.credential_sources]
ENDPOINT = "https://gateway.example/v1"

# What a credential file of each kind looks like on disk. Codex classifies its
# file by content; a file whose content is unrecognised classifies as nothing.
CODEX_AUTH = {
    CredentialKind.SUBSCRIPTION: json.dumps({"auth_mode": "chatgpt", "tokens": {}}),
    CredentialKind.API_KEY: json.dumps({"auth_mode": "apikey", "OPENAI_API_KEY": "sk"}),
    None: "not json",
}

Case = tuple[int, dict[str, str], dict[str, str], AuthPolicy]

kinds = st.frozensets(st.sampled_from(list(CredentialKind)), min_size=1)


def _secret(name: str) -> str:
    return f"secret-of-{name}"


@st.composite
def hosts(draw: st.DrawFn, profile_index: int) -> tuple[dict[str, str], dict[str, str]]:
    """A host environment and home-relative files for one profile."""
    profile = ENFORCEABLE[profile_index]
    names = list(profile.auth_env_vars)
    if profile.endpoint is not None:
        names += [profile.endpoint.base_url_env, *profile.endpoint.companion_env]
    env: dict[str, str] = {}
    for name in names:
        value = draw(st.sampled_from([None, "", "  ", _secret(name)]))
        if value is not None:
            env[name] = value
    files: dict[str, str] = {}
    for path in profile.auth_files:
        if draw(st.booleans()):
            if path == ".codex/auth.json":
                files[path] = CODEX_AUTH[draw(st.sampled_from(list(CODEX_AUTH)))]
            else:
                files[path] = "{}"
    return env, files


def _plant(home: Path, files: dict[str, str]) -> None:
    for path, contents in files.items():
        target = home / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents)


def _present(
    profile_index: int, env: dict[str, str], files: dict[str, str]
) -> list[tuple[str, CredentialKind]]:
    """The present sources, in precedence order, with the kind each holds."""
    profile = ENFORCEABLE[profile_index]
    out: list[tuple[str, CredentialKind]] = []
    for source in profile.credential_sources:
        if isinstance(source, EnvCredential):
            if env.get(source.name, "").strip():
                out.append((source.name, source.kind))
        elif source.path in files:
            kind = source.kind_of(files[source.path].encode())
            if kind is not None:
                out.append((source.path, kind))
    return out


@st.composite
def cases(draw: st.DrawFn) -> Case:
    index = draw(st.integers(0, len(ENFORCEABLE) - 1))
    env, files = draw(hosts(index))
    policy = AuthPolicy(draw(kinds), draw(st.sampled_from([None, ENDPOINT])))
    return index, env, files, policy


# Each example writes a home to disk; the input space is small (a few
# providers, four states per variable, a file each), so 60 examples cover it.
@settings(max_examples=60)
@given(cases())
def test_a_plan_delivers_exactly_the_first_present_allowed_credential(case: Case) -> None:
    index, env, files, policy = case
    profile = ENFORCEABLE[index]
    allowed = policy.credentials
    if policy.endpoint is not None and profile.endpoint is not None:
        allowed = allowed & profile.endpoint.credentials
    expected = next(
        ((name, kind) for name, kind in _present(index, env, files) if kind in allowed), None
    )
    endpoint_impossible = policy.endpoint is not None and (profile.endpoint is None or not allowed)
    with tempfile.TemporaryDirectory() as directory:
        home = Path(directory)
        _plant(home, files)
        if endpoint_impossible or expected is None:
            with pytest.raises(AuthUnavailableError) as raised:
                plan_auth(profile, policy, env=env, home=home)
            assert not any(value in str(raised.value) for value in env.values() if value.strip())
            return
        plan = plan_auth(profile, policy, env=env, home=home)

    assert (plan.source, plan.credential) == expected
    assert plan.credential in policy.credentials
    credential_names = {
        source.name for source in profile.credential_sources if isinstance(source, EnvCredential)
    }
    delivered_credentials = (set(plan.env) & credential_names) | {f.home_path for f in plan.files}
    assert delivered_credentials == {plan.source}
    endpoint_names: set[str] = set()
    if profile.endpoint is not None:
        endpoint_names = {profile.endpoint.base_url_env, *profile.endpoint.companion_env}
    assert set(plan.env) <= {plan.source} | endpoint_names
    if policy.endpoint is None:
        assert not set(plan.env) & endpoint_names
    else:
        assert profile.endpoint is not None
        assert plan.env[profile.endpoint.base_url_env] == policy.endpoint
    for name in set(plan.env) - {profile.endpoint.base_url_env if profile.endpoint else ""}:
        assert plan.env[name] == env[name]
    for staged in plan.files:
        assert staged.host_path == home / staged.home_path
        assert staged.refreshable is (plan.credential is CredentialKind.SUBSCRIPTION)


# Each example writes a home to disk; the input space is small (a few
# providers, four states per variable, a file each), so 60 examples cover it.
@settings(max_examples=60)
@given(cases())
def test_withheld_names_every_present_credential_left_out_and_no_value(case: Case) -> None:
    index, env, files, policy = case
    profile = ENFORCEABLE[index]
    with tempfile.TemporaryDirectory() as directory:
        home = Path(directory)
        _plant(home, files)
        try:
            plan = plan_auth(profile, policy, env=env, home=home)
        except AuthUnavailableError:
            return
    delivered = set(plan.env) | {staged.home_path for staged in plan.files}
    present_env = {name for name, value in env.items() if value.strip()}
    assert set(plan.withheld).isdisjoint(delivered)
    assert (present_env | set(files)) - delivered <= set(plan.withheld)
    assert not set(plan.withheld) & set(env.values())


@pytest.mark.parametrize("profile", PROFILES, ids=lambda profile: profile.name)
def test_every_declared_credential_is_one_the_profile_already_lists(
    profile: ProviderProfile,
) -> None:
    for source in profile.credential_sources:
        if isinstance(source, EnvCredential):
            assert source.name in profile.auth_env_vars
        else:
            assert isinstance(source, FileCredential)
            assert source.path in profile.credential_files
    if profile.endpoint is not None:
        for name in (profile.endpoint.base_url_env, *profile.endpoint.companion_env):
            assert name in profile.auth_env_vars


def test_a_cli_with_unclassified_credentials_refuses_every_policy(tmp_path: Path) -> None:
    opencode = get_provider("opencode").profile
    with pytest.raises(AuthUnavailableError, match="no credential sources"):
        plan_auth(opencode, AuthPolicy(frozenset(CredentialKind)), env={}, home=tmp_path)


def test_a_gateway_token_beats_a_subscription_login_only_when_api_keys_are_allowed(
    tmp_path: Path,
) -> None:
    claude = get_provider("claude").profile
    _plant(tmp_path, {".claude/.credentials.json": "{}", ".claude/settings.json": "{}"})
    env = {
        "ANTHROPIC_AUTH_TOKEN": "tok",
        "ANTHROPIC_BASE_URL": "https://left.in.shell",
        "PATH": "/usr/bin",
    }
    plan = plan_auth(
        claude,
        AuthPolicy(frozenset({CredentialKind.API_KEY}), ENDPOINT),
        env=env,
        home=tmp_path,
    )
    assert dict(plan.env) == {"ANTHROPIC_AUTH_TOKEN": "tok", "ANTHROPIC_BASE_URL": ENDPOINT}
    assert plan.files == ()
    assert plan.withheld == (".claude/.credentials.json", ".claude/settings.json")
    subscription = plan_auth(
        claude,
        AuthPolicy(frozenset({CredentialKind.SUBSCRIPTION})),
        env=env,
        home=tmp_path,
    )
    assert subscription.source == ".claude/.credentials.json"
    assert "ANTHROPIC_BASE_URL" in subscription.withheld


@pytest.mark.parametrize(
    "endpoint", ["", "gateway.example", "ftp://gateway.example", "https://", "file:///x"]
)
def test_a_policy_endpoint_must_be_an_http_url(endpoint: str) -> None:
    with pytest.raises(ValueError, match="endpoint"):
        AuthPolicy(frozenset({CredentialKind.API_KEY}), endpoint)


def test_a_policy_must_allow_some_credential() -> None:
    with pytest.raises(ValueError, match="at least one"):
        AuthPolicy(frozenset())


@pytest.mark.parametrize(
    ("method", "env", "credential"),
    [
        ("claude.ai", {}, CredentialKind.SUBSCRIPTION),
        ("api_key", {}, CredentialKind.API_KEY),
        ("oauth_token", {"ANTHROPIC_AUTH_TOKEN": "t"}, CredentialKind.API_KEY),
        ("oauth_token", {"CLAUDE_CODE_OAUTH_TOKEN": "t"}, CredentialKind.SUBSCRIPTION),
        ("oauth_token", {"ANTHROPIC_AUTH_TOKEN": "t", "CLAUDE_CODE_OAUTH_TOKEN": "t"}, None),
        ("oauth_token", {}, None),
        ("something_new", {}, None),
    ],
)
def test_claude_status_names_the_credential_in_use(
    method: str, env: dict[str, str], credential: CredentialKind | None
) -> None:
    status = json.dumps({"loggedIn": True, "authMethod": method})
    assert read_claude_auth(ProbeOutput(0, status, ""), env).credential is credential


@given(st.dictionaries(st.text(), st.text()), st.text())
def test_claude_reports_no_credential_unless_logged_in(env: dict[str, str], method: str) -> None:
    status = json.dumps({"loggedIn": False, "authMethod": method})
    assert read_claude_auth(ProbeOutput(1, status, ""), env).credential is None


@pytest.mark.parametrize(
    ("text", "credential"),
    [
        ("Logged in using ChatGPT\n", CredentialKind.SUBSCRIPTION),
        ("WARNING: noise\nLogged in using an API key - sk-***\n", CredentialKind.API_KEY),
        ("Logged in somehow else\n", None),
        ("Not logged in\n", None),
    ],
)
def test_codex_status_names_the_credential_in_use(
    text: str, credential: CredentialKind | None
) -> None:
    assert read_codex_auth(ProbeOutput(0, text, ""), {}).credential is credential
