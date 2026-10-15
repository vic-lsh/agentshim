"""host_auth_plan reproduces the login a CLI picks by itself, or says why it cannot."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from agentshim import (
    AuthPolicy,
    CredentialKind,
    EnvCredential,
    ExternalCredential,
    HostAuthGap,
    HostAuthUnplannableError,
    ProviderProfile,
    host_auth_plan,
    plan_auth,
)
from agentshim.providers import get_provider, provider_names
from hypothesis import given, settings
from hypothesis import strategies as st

PROFILES = [get_provider(name).profile for name in provider_names()]
CLASSIFIED = [profile for profile in PROFILES if profile.credential_sources]
CLAUDE = get_provider("claude").profile

# Contents a file can hold. Codex classifies auth.json by content; a settings
# file may set nothing, a key an ExternalCredential looks for, or an env block.
_CODEX_AUTH = [
    json.dumps({"auth_mode": "chatgpt"}),
    json.dumps({"auth_mode": "apikey"}),
    "not json",
]

Host = tuple[int, dict[str, str], dict[str, str]]


def _secret(name: str) -> str:
    return f"secret-of-{name}"


def _env_names(profile: ProviderProfile) -> list[str]:
    names = list(profile.auth_env_vars)
    if profile.endpoint is not None:
        names += [profile.endpoint.base_url_env, *profile.endpoint.companion_env]
    for source in profile.credential_sources:
        if isinstance(source, ExternalCredential):
            names += source.env
    return sorted(set(names))


_BENIGN_SETTINGS = ["{}", json.dumps({"theme": "dark"}), "not json"]


def _external_env(profile: ProviderProfile) -> set[str]:
    return {
        name
        for source in profile.credential_sources
        if isinstance(source, ExternalCredential)
        for name in source.env
    }


def _settings_contents(profile: ProviderProfile) -> list[str]:
    """What a settings file may hold that changes the login: an external key, an env block."""
    contents = [json.dumps({"env": {"UNRELATED": "1"}})]
    for source in profile.credential_sources:
        if isinstance(source, ExternalCredential):
            contents += [json.dumps({key: "x"}) for key in source.settings_keys]
    if profile.settings is not None and profile.settings.env_key is not None:
        key = profile.settings.env_key
        contents += [json.dumps({key: {name: "x"}}) for name in _env_names(profile)]
    return contents


@st.composite
def hosts(draw: st.DrawFn) -> Host:
    """A host environment and home-relative files for one classified profile."""
    index = draw(st.integers(0, len(CLASSIFIED) - 1))
    profile = CLASSIFIED[index]
    env: dict[str, str] = {}
    # A host rarely sets a variable that selects an external credential, or a
    # settings file that changes the login; draw those one time in four so most
    # examples reach a plan.
    for name in _env_names(profile):
        if name in _external_env(profile) and draw(st.integers(0, 3)):
            continue
        value = draw(st.sampled_from([None, "", "  ", _secret(name), "not a url"]))
        if value is not None:
            env[name] = value
    paths = set(profile.auth_files)
    for source in profile.credential_sources:
        if isinstance(source, ExternalCredential):
            paths.update(path for path in source.paths if not draw(st.integers(0, 3)))
    settings_files = set(profile.settings.files) if profile.settings is not None else set()
    files: dict[str, str] = {}
    for path in sorted(paths):
        if not draw(st.booleans()):
            continue
        if path == ".codex/auth.json":
            files[path] = draw(st.sampled_from(_CODEX_AUTH))
        elif path in settings_files:
            relevant = not draw(st.integers(0, 3))
            choices = _settings_contents(profile) if relevant else _BENIGN_SETTINGS
            files[path] = draw(st.sampled_from(choices))
        else:
            files[path] = "{}"
    return index, env, files


def _plant(home: Path, files: dict[str, str]) -> None:
    for path, contents in files.items():
        target = home / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents)


def _settings(profile: ProviderProfile, files: dict[str, str]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for path in profile.settings.files if profile.settings is not None else ():
        try:
            data = json.loads(files.get(path, "null"))
        except ValueError:
            continue
        if isinstance(data, dict):
            out.append(data)
    return out


def _settings_env_set(profile: ProviderProfile, loaded: list[dict[str, object]]) -> bool:
    key = profile.settings.env_key if profile.settings is not None else None
    blocks = [data.get(key) for data in loaded] if key is not None else []
    return any(
        isinstance(block, dict) and set(block) & set(_env_names(profile)) for block in blocks
    )


def _external_found(
    source: ExternalCredential,
    env: dict[str, str],
    files: dict[str, str],
    loaded: list[dict[str, object]],
) -> bool:
    return (
        any(env.get(name, "").strip() for name in source.env)
        or any(path in files for path in source.paths)
        or any(key in data for data in loaded for key in source.settings_keys)
    )


def _expected(profile: ProviderProfile, env: dict[str, str], files: dict[str, str]) -> object:
    """The CLI's own choice, worked out from the declaration: a source, or a gap."""
    loaded = _settings(profile, files)
    if _settings_env_set(profile, loaded):
        return HostAuthGap.SETTINGS_ENV
    for source in profile.credential_sources:
        if isinstance(source, ExternalCredential):
            if _external_found(source, env, files, loaded):
                return HostAuthGap.EXTERNAL_CREDENTIAL
        elif isinstance(source, EnvCredential):
            if env.get(source.name, "").strip():
                return source.name, source.kind
        elif source.path in files:
            kind = source.kind_of(files[source.path].encode())
            if kind is not None:
                return source.path, kind
    return HostAuthGap.NO_CREDENTIAL


# Each example writes a home to disk; the input space is a few providers, five
# states per variable and a handful of file contents, so 100 examples cover it.
@settings(max_examples=100)
@given(hosts())
def test_the_plan_is_the_cli_s_own_choice_and_reproduces_its_endpoint(host: Host) -> None:
    index, env, files = host
    profile = CLASSIFIED[index]
    expected = _expected(profile, env, files)
    with tempfile.TemporaryDirectory() as directory:
        home = Path(directory)
        _plant(home, files)
        if isinstance(expected, HostAuthGap):
            with pytest.raises(HostAuthUnplannableError) as raised:
                host_auth_plan(profile, env=env, home=home)
            assert raised.value.reason is expected
            assert not any(value in str(raised.value) for value in env.values() if value.strip())
            return
        plan = host_auth_plan(profile, env=env, home=home)

    assert (plan.source, plan.credential) == expected
    endpoint_names: set[str] = set()
    if profile.endpoint is not None:
        endpoint_names = {profile.endpoint.base_url_env, *profile.endpoint.companion_env}
    credential_names = set(_env_names(profile)) - endpoint_names
    delivered_credentials = (set(plan.env) & credential_names) | {f.home_path for f in plan.files}
    assert delivered_credentials == {plan.source}
    # Every endpoint variable the host sets reaches the copy verbatim, and only those.
    assert {name for name in endpoint_names if env.get(name, "").strip()} == (
        set(plan.env) & endpoint_names
    )
    assert all(plan.env[name] == env[name] for name in plan.env)
    base_url = profile.endpoint.base_url_env if profile.endpoint is not None else None
    assert plan.endpoint == (plan.env.get(base_url) if base_url is not None else None)
    for staged in plan.files:
        assert staged.host_path == home / staged.home_path
        assert staged.refreshable is (plan.credential is CredentialKind.SUBSCRIPTION)
    delivered = set(plan.env) | {f.home_path for f in plan.files}
    assert set(plan.withheld).isdisjoint(delivered)
    assert set(files) & set(profile.auth_files) - delivered <= set(plan.withheld)
    assert not set(plan.withheld) & {value for value in env.values() if value.strip()}


# Same input space as above.
@settings(max_examples=100)
@given(hosts())
def test_at_the_vendor_endpoint_the_host_plan_is_plan_auth_allowing_every_kind(
    host: Host,
) -> None:
    index, env, files = host
    profile = CLASSIFIED[index]
    if profile.endpoint is not None:
        for name in (profile.endpoint.base_url_env, *profile.endpoint.companion_env):
            env.pop(name, None)
    with tempfile.TemporaryDirectory() as directory:
        home = Path(directory)
        _plant(home, files)
        try:
            plan = host_auth_plan(profile, env=env, home=home)
        except HostAuthUnplannableError:
            return
        policy = plan_auth(profile, AuthPolicy(frozenset(CredentialKind)), env=env, home=home)
    assert (plan.source, plan.credential, dict(plan.env), plan.files) == (
        policy.source,
        policy.credential,
        dict(policy.env),
        policy.files,
    )
    assert plan.withheld == policy.withheld


def test_a_cli_with_unclassified_credentials_cannot_be_planned(tmp_path: Path) -> None:
    opencode = get_provider("opencode").profile
    with pytest.raises(HostAuthUnplannableError) as raised:
        host_auth_plan(opencode, env={}, home=tmp_path)
    assert raised.value.reason is HostAuthGap.UNCLASSIFIED


def test_a_subscription_token_goes_to_the_host_s_custom_endpoint(tmp_path: Path) -> None:
    # Claude Code 2.1.300 sends CLAUDE_CODE_OAUTH_TOKEN to ANTHROPIC_BASE_URL as
    # a bearer token, so the copy gets both.
    _plant(tmp_path, {".claude/.credentials.json": "{}", ".claude/settings.json": "{}"})
    env = {
        "CLAUDE_CODE_OAUTH_TOKEN": "tok",
        "ANTHROPIC_BASE_URL": "https://gateway.example",
        "ANTHROPIC_CUSTOM_HEADERS": "X-Team: a",
    }
    plan = host_auth_plan(CLAUDE, env=env, home=tmp_path)
    assert plan.credential is CredentialKind.SUBSCRIPTION
    assert dict(plan.env) == env
    assert plan.endpoint == "https://gateway.example"
    assert plan.files == ()
    assert plan.withheld == (".claude/.credentials.json", ".claude/settings.json")


@pytest.mark.parametrize("base_url", ["not a url", "127.0.0.1:8080", " https://gateway.example "])
def test_a_host_base_url_is_reproduced_not_validated(tmp_path: Path, base_url: str) -> None:
    # The host CLI fails every request on a value it cannot use ("Invalid URL")
    # and never falls back to the vendor; the copy must do the same.
    env = {"ANTHROPIC_API_KEY": "key", "ANTHROPIC_BASE_URL": base_url}
    plan = host_auth_plan(CLAUDE, env=env, home=tmp_path)
    assert dict(plan.env) == env
    assert plan.endpoint == base_url


def test_an_api_key_helper_outranks_a_subscription_token_but_not_an_api_key(
    tmp_path: Path,
) -> None:
    _plant(tmp_path, {".claude/settings.json": json.dumps({"apiKeyHelper": "vault-get"})})
    with pytest.raises(HostAuthUnplannableError, match="apiKeyHelper") as raised:
        host_auth_plan(CLAUDE, env={"CLAUDE_CODE_OAUTH_TOKEN": "tok"}, home=tmp_path)
    assert raised.value.reason is HostAuthGap.EXTERNAL_CREDENTIAL
    plan = host_auth_plan(CLAUDE, env={"ANTHROPIC_API_KEY": "key"}, home=tmp_path)
    assert plan.source == "ANTHROPIC_API_KEY"
    assert "apiKeyHelper in .claude/settings.json" in plan.withheld


def test_a_cloud_provider_outranks_every_key(tmp_path: Path) -> None:
    env = {"CLAUDE_CODE_USE_BEDROCK": "1", "ANTHROPIC_AUTH_TOKEN": "tok"}
    with pytest.raises(HostAuthUnplannableError, match="CLAUDE_CODE_USE_BEDROCK") as raised:
        host_auth_plan(CLAUDE, env=env, home=tmp_path)
    assert raised.value.reason is HostAuthGap.EXTERNAL_CREDENTIAL
    assert "tok" not in str(raised.value)


def test_a_settings_env_block_that_sets_an_auth_variable_is_refused(tmp_path: Path) -> None:
    settings_file = {"env": {"ANTHROPIC_BASE_URL": "https://elsewhere.example", "FOO": "1"}}
    _plant(tmp_path, {".claude/settings.local.json": json.dumps(settings_file)})
    with pytest.raises(HostAuthUnplannableError, match="ANTHROPIC_BASE_URL") as raised:
        host_auth_plan(CLAUDE, env={"ANTHROPIC_API_KEY": "key"}, home=tmp_path)
    assert raised.value.reason is HostAuthGap.SETTINGS_ENV
    assert "elsewhere" not in str(raised.value)


def test_a_host_with_no_credential_cannot_be_planned(tmp_path: Path) -> None:
    _plant(tmp_path, {".claude/settings.json": "{}"})
    with pytest.raises(HostAuthUnplannableError) as raised:
        host_auth_plan(CLAUDE, env={"ANTHROPIC_API_KEY": "  "}, home=tmp_path)
    assert raised.value.reason is HostAuthGap.NO_CREDENTIAL
