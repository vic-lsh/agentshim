"""Credential policy: which credential a CLI may log in with, and what to hand it.

A CLI can log in several ways, and it picks one by its own precedence rules
when more than one is present: a pay-per-token key in the environment, a
subscription login in a file under its state root, a token that a proxy
issued. A caller that must control which one is used (bill an API key, never
the operator's subscription) cannot do it by setting one variable: whatever
else the CLI can see may win. It has to hand the CLI exactly one credential
and nothing else.

Each provider declares its credentials as data (``ProviderProfile.
credential_sources``, in the CLI's own precedence order, and
``ProviderProfile.endpoint``). ``plan_auth`` turns that, a caller's
``AuthPolicy`` and the host's environment and home into an ``AuthPlan``: the
exact environment variables and files to deliver. The caller delivers the
plan and nothing more of the provider's ``auth_env_vars`` or ``auth_files``;
the readiness probe then reports which kind of credential the CLI says it is
using (``ProviderStatus.credential``), so the caller can confirm it.

A caller with no policy of its own, that wants a confined copy of the CLI to
log in exactly as the CLI would on the host, asks ``host_auth_plan`` instead:
it applies the CLI's own precedence with every kind allowed and the host's
endpoint as it is, and refuses (``HostAuthUnplannableError``) when the host
CLI would log in some way a plan cannot reproduce.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

from .errors import AgentShimError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from .profile import ProviderProfile


class CredentialKind(str, Enum):
    """What a credential bills and how it lives.

    ``API_KEY`` is a static secret read from the environment or a key file: a
    vendor API key, or a token a proxy or gateway issued. ``SUBSCRIPTION`` is a
    personal or team plan login (Claude.ai, ChatGPT, a Google or GitHub
    account), usually an OAuth token the CLI refreshes.
    """

    API_KEY = "api_key"
    SUBSCRIPTION = "subscription"


@dataclass(frozen=True)
class EnvCredential:
    """A credential the CLI reads from one environment variable."""

    name: str
    kind: CredentialKind


@dataclass(frozen=True)
class FileCredential:
    """A credential the CLI reads from a home-relative file.

    The file's kind is either fixed (``kind``) or read from its contents
    (``classify``, which returns ``None`` for contents it does not recognise);
    exactly one is set. The path is also one of the profile's
    ``credential_files``.
    """

    path: str
    kind: CredentialKind | None = None
    classify: Callable[[bytes], CredentialKind | None] | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        """Require exactly one of ``kind`` and ``classify``."""
        if (self.kind is None) == (self.classify is None):
            msg = f"FileCredential {self.path!r} needs exactly one of kind and classify"
            raise ValueError(msg)

    def kind_of(self, contents: bytes) -> CredentialKind | None:
        """The kind of credential *contents* hold, or ``None`` if unrecognised."""
        if self.kind is not None:
            return self.kind
        assert self.classify is not None  # noqa: S101 - __post_init__ guarantees it
        return self.classify(contents)


@dataclass(frozen=True)
class ExternalCredential:
    """A way the CLI logs in that no plan can hand to another copy of it.

    A cloud provider selected by a variable (whose own credentials live in
    other tools' state), a helper command named in a settings file, a
    credential store outside the CLI's state root. It is declared at its place
    in the precedence order so ``host_auth_plan`` knows when the host CLI would
    use it; ``plan_auth`` never chooses it. It is present when any variable in
    ``env`` is set, any home-relative path in ``paths`` exists, or any of the
    profile's ``settings`` files has one of ``settings_keys`` at its top level.
    """

    label: str
    env: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()
    settings_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Require some way to detect the credential."""
        if not (self.env or self.paths or self.settings_keys):
            msg = f"ExternalCredential {self.label!r} needs an env, paths or settings_keys entry"
            raise ValueError(msg)


CredentialSource = EnvCredential | FileCredential | ExternalCredential
#: The sources a plan can hand over.
_DeliverableSource = EnvCredential | FileCredential


@dataclass(frozen=True)
class SettingsSpec:
    """The JSON settings files a CLI reads from its home when it starts.

    ``files`` are home-relative paths. ``env_key`` names a top-level object
    whose string entries the CLI applies over its own environment (Claude
    Code's ``env``), or ``None`` when the CLI has no such block. Settings can
    change which credential or endpoint the CLI uses, so ``host_auth_plan``
    reads them; neither plan function delivers them.
    """

    files: tuple[str, ...]
    env_key: str | None = None


@dataclass(frozen=True)
class EndpointSpec:
    """How a CLI is pointed at an API endpoint other than its vendor's.

    ``base_url_env`` takes the endpoint URL. ``companion_env`` are variables
    that only make sense with a custom endpoint (extra headers a gateway
    needs); they are forwarded from the host only when the policy names an
    endpoint. ``credentials`` are the kinds the CLI sends to a custom endpoint.
    """

    base_url_env: str
    companion_env: tuple[str, ...] = ()
    credentials: frozenset[CredentialKind] = frozenset({CredentialKind.API_KEY})


@dataclass(frozen=True)
class AuthPolicy:
    """Which credentials a caller allows, and the endpoint to use.

    ``endpoint`` is ``None`` for the vendor's own endpoint: a base URL left in
    the host environment is then withheld, not forwarded. A URL sends every
    request there.
    """

    credentials: frozenset[CredentialKind]
    endpoint: str | None = None

    def __post_init__(self) -> None:
        """Reject an empty credential set and an endpoint that is not an http(s) URL."""
        if not self.credentials:
            msg = "AuthPolicy.credentials must allow at least one CredentialKind"
            raise ValueError(msg)
        if self.endpoint is not None:
            parts = urlsplit(self.endpoint)
            if parts.scheme not in {"http", "https"} or not parts.hostname:
                msg = (
                    f"AuthPolicy.endpoint must be an http(s) URL with a host, got {self.endpoint!r}"
                )
                raise ValueError(msg)


@dataclass(frozen=True)
class StagedCredentialFile:
    """A host credential file the CLI must see at ``home_path`` under its home.

    ``refreshable`` files hold a login the CLI rewrites when it refreshes the
    token; share those writable so the refreshed token reaches the host.
    """

    host_path: Path
    home_path: str
    refreshable: bool


@dataclass(frozen=True)
class AuthPlan:
    """Exactly what to hand a CLI so it logs in with one allowed credential.

    ``env`` and ``files`` are the whole delivery: a caller that also forwards
    any other of the profile's ``auth_env_vars`` or ``auth_files`` (settings
    files included, which can carry credentials of their own) defeats the
    plan. ``source`` names the chosen credential (a variable name or a
    home-relative path); ``withheld`` names, never values, the credentials and
    endpoint settings present on the host that the plan leaves out, for a log.
    """

    provider: str
    credential: CredentialKind
    source: str
    endpoint: str | None
    env: Mapping[str, str]
    files: tuple[StagedCredentialFile, ...]
    withheld: tuple[str, ...]


class AuthUnavailableError(AgentShimError):
    """No credential the policy allows is present, or the policy cannot be met.

    The message names what each allowed kind would need, and never a value.
    """


class HostAuthGap(str, Enum):
    """Why ``host_auth_plan`` cannot reproduce how the host CLI logs in.

    ``UNCLASSIFIED``: the profile declares no credential sources. ``NO_CREDENTIAL``:
    none of the declared sources is present, so the CLI either cannot log in
    or does it some way agentshim does not know. ``EXTERNAL_CREDENTIAL``: the
    CLI would use an ``ExternalCredential``. ``SETTINGS_ENV``: a settings file
    sets a credential or endpoint variable of its own, so the environment the
    CLI logs in with is not the caller's.
    """

    UNCLASSIFIED = "unclassified"
    NO_CREDENTIAL = "no_credential"
    EXTERNAL_CREDENTIAL = "external_credential"
    SETTINGS_ENV = "settings_env"


class HostAuthUnplannableError(AgentShimError):
    """The host CLI's login cannot be reproduced by a plan; ``reason`` says why.

    The message names the variables or files involved, never a value. A caller
    that still wants the CLI logged in elsewhere has only the host's whole auth
    state (``auth_env_vars`` and ``auth_files``) to hand over.
    """

    def __init__(self, reason: HostAuthGap, message: str) -> None:
        """Record *reason* beside the message."""
        super().__init__(message)
        self.reason = reason


def plan_auth(
    profile: ProviderProfile,
    policy: AuthPolicy,
    *,
    env: Mapping[str, str],
    home: Path,
) -> AuthPlan:
    """Choose the credential *profile*'s CLI will log in with under *policy*.

    Walks the profile's ``credential_sources`` in the CLI's precedence order
    and takes the first one present on the host (a non-blank variable in
    *env*, or an existing file under *home* whose contents classify) whose kind
    the policy allows. Only that credential is delivered, so the CLI cannot
    prefer another.

    Raises:
        AuthUnavailableError: the profile declares no credential sources, the
            policy names an endpoint the CLI cannot take or that accepts none
            of the allowed kinds, or no allowed credential is present.
    """
    allowed = _allowed_kinds(profile, policy)
    chosen, kind, present = _choose(profile, allowed, env, home)
    delivered_env: dict[str, str] = {}
    files: tuple[StagedCredentialFile, ...] = ()
    if isinstance(chosen, EnvCredential):
        delivered_env[chosen.name] = env[chosen.name]
    else:
        refreshable = kind is CredentialKind.SUBSCRIPTION
        files = (StagedCredentialFile(home / chosen.path, chosen.path, refreshable),)
    if policy.endpoint is not None and profile.endpoint is not None:
        delivered_env[profile.endpoint.base_url_env] = policy.endpoint
        delivered_env.update(
            {name: env[name] for name in profile.endpoint.companion_env if _set(env, name)}
        )

    return AuthPlan(
        provider=profile.name,
        credential=kind,
        source=_name(chosen),
        endpoint=policy.endpoint,
        env=MappingProxyType(delivered_env),
        files=files,
        withheld=_withheld(profile, env, home, present, _delivered(delivered_env, files)),
    )


def host_auth_plan(profile: ProviderProfile, *, env: Mapping[str, str], home: Path) -> AuthPlan:
    """Say how *profile*'s CLI logs in by itself on a host, as a plan a confined copy can use.

    The CLI's own choice: the first present source in ``credential_sources``,
    of any kind, at whatever endpoint *env* points it to. The plan delivers
    that credential, and the host's endpoint variables (``EndpointSpec``
    ``base_url_env`` and ``companion_env``) exactly as they are set, so a
    confined copy given only the plan sends the same credential to the same
    place. ``AuthPlan.endpoint`` is the host's base URL, or ``None`` for the
    vendor's endpoint.

    The host's base URL is reproduced, not validated: a value the CLI cannot
    use fails the copy the same way it fails the host CLI (Claude Code fails
    every request with "Invalid URL" and never falls back to its vendor). The
    credential's kind is not checked against ``EndpointSpec.credentials``
    either: what the CLI does with a credential and a custom endpoint is the
    CLI's behaviour, and the copy reproduces it.

    Raises:
        HostAuthUnplannableError: the profile declares no credential sources,
            none is present, the CLI would log in with an
            ``ExternalCredential``, or a settings file sets a credential or
            endpoint variable; ``reason`` is the matching ``HostAuthGap``.
    """
    if not profile.credential_sources:
        msg = f"{profile.name} declares no credential sources agentshim can classify"
        raise HostAuthUnplannableError(HostAuthGap.UNCLASSIFIED, msg)
    settings = _read_settings(profile, home)
    _refuse_settings_env(profile, settings)
    chosen, kind = _host_choice(profile, env, home, settings)
    delivered_env: dict[str, str] = {}
    files: tuple[StagedCredentialFile, ...] = ()
    if isinstance(chosen, EnvCredential):
        delivered_env[chosen.name] = env[chosen.name]
    else:
        refreshable = kind is CredentialKind.SUBSCRIPTION
        files = (StagedCredentialFile(home / chosen.path, chosen.path, refreshable),)
    endpoint: str | None = None
    if profile.endpoint is not None:
        names = (profile.endpoint.base_url_env, *profile.endpoint.companion_env)
        delivered_env.update({name: env[name] for name in names if _set(env, name)})
        endpoint = delivered_env.get(profile.endpoint.base_url_env)
    present = _present_names(profile, env, home, settings)
    return AuthPlan(
        provider=profile.name,
        credential=kind,
        source=_name(chosen),
        endpoint=endpoint,
        env=MappingProxyType(delivered_env),
        files=files,
        withheld=_withheld(profile, env, home, present, _delivered(delivered_env, files)),
    )


def classify_json_field(
    key: str, kinds: Mapping[str, CredentialKind]
) -> Callable[[bytes], CredentialKind | None]:
    """A ``FileCredential.classify`` that maps a top-level JSON string field to a kind.

    Contents that are not a JSON object, lack *key*, or hold a value not in
    *kinds* classify as ``None``.
    """

    def classify(contents: bytes) -> CredentialKind | None:
        try:
            data: Any = json.loads(contents)
        except ValueError:
            return None
        if not isinstance(data, dict):
            return None
        value = cast("dict[str, Any]", data).get(key)
        return kinds.get(value) if isinstance(value, str) else None

    return classify


def _allowed_kinds(profile: ProviderProfile, policy: AuthPolicy) -> frozenset[CredentialKind]:
    """The kinds *policy* allows that *profile*'s CLI can use with the policy's endpoint."""
    if not profile.credential_sources:
        msg = (
            f"{profile.name} declares no credential sources agentshim can classify, "
            "so no credential policy can be enforced for it"
        )
        raise AuthUnavailableError(msg)
    if policy.endpoint is None:
        return policy.credentials
    if profile.endpoint is None:
        msg = f"{profile.name} cannot be pointed at another endpoint ({policy.endpoint})"
        raise AuthUnavailableError(msg)
    allowed = policy.credentials & profile.endpoint.credentials
    if not allowed:
        msg = (
            f"{profile.name} sends only {_kinds(profile.endpoint.credentials)} credentials "
            f"to a custom endpoint, and the policy allows {_kinds(policy.credentials)}"
        )
        raise AuthUnavailableError(msg)
    return allowed


def _choose(
    profile: ProviderProfile,
    allowed: frozenset[CredentialKind],
    env: Mapping[str, str],
    home: Path,
) -> tuple[_DeliverableSource, CredentialKind, list[str]]:
    """The first present allowed source, its kind, and every present source's name."""
    chosen: tuple[_DeliverableSource, CredentialKind] | None = None
    present: list[str] = []
    for source in profile.credential_sources:
        if isinstance(source, ExternalCredential):
            present += _external_found(profile, source, env, home)
            continue
        kind = _present_kind(source, env, home)
        if kind is None:
            continue
        present.append(_name(source))
        if chosen is None and kind in allowed:
            chosen = (source, kind)
    if chosen is None:
        raise AuthUnavailableError(_unavailable(profile, allowed, present))
    return chosen[0], chosen[1], present


def _host_choice(
    profile: ProviderProfile,
    env: Mapping[str, str],
    home: Path,
    settings: Mapping[str, dict[str, Any]],
) -> tuple[_DeliverableSource, CredentialKind]:
    """The source the CLI itself would log in with: the first present one, of any kind."""
    for source in profile.credential_sources:
        if isinstance(source, ExternalCredential):
            found = _external_found(profile, source, env, home, settings)
            if found:
                msg = (
                    f"{profile.name} would log in with {source.label} "
                    f"({', '.join(found)}), which no plan can hand to another copy of it"
                )
                raise HostAuthUnplannableError(HostAuthGap.EXTERNAL_CREDENTIAL, msg)
            continue
        kind = _present_kind(source, env, home)
        if kind is not None:
            return source, kind
    wanted = ", ".join(
        _name(source)
        for source in profile.credential_sources
        if not isinstance(source, ExternalCredential)
    )
    msg = f"no credential {profile.name} logs in with is present: none of {wanted}"
    raise HostAuthUnplannableError(HostAuthGap.NO_CREDENTIAL, msg)


def _read_settings(profile: ProviderProfile, home: Path) -> dict[str, dict[str, Any]]:
    """Each of *profile*'s settings files that holds a JSON object, by path."""
    out: dict[str, dict[str, Any]] = {}
    for path in profile.settings.files if profile.settings is not None else ():
        try:
            data: Any = json.loads((home / path).read_bytes())
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            out[path] = cast("dict[str, Any]", data)
    return out


def _refuse_settings_env(profile: ProviderProfile, settings: Mapping[str, dict[str, Any]]) -> None:
    """Raise when a settings file's env block sets a variable the CLI logs in with."""
    key = profile.settings.env_key if profile.settings is not None else None
    if key is None:
        return
    names = _host_auth_names(profile) | _credential_env_names(profile)
    for path, data in settings.items():
        block: Any = data.get(key)
        if not isinstance(block, dict):
            continue
        overridden = sorted(names & set(cast("dict[str, Any]", block)))
        if overridden:
            msg = (
                f"{path} sets {', '.join(overridden)} in its {key!r} block, so "
                f"{profile.name} does not log in with the caller's environment"
            )
            raise HostAuthUnplannableError(HostAuthGap.SETTINGS_ENV, msg)


def _external_found(
    profile: ProviderProfile,
    source: ExternalCredential,
    env: Mapping[str, str],
    home: Path,
    settings: Mapping[str, dict[str, Any]] | None = None,
) -> list[str]:
    """What of *source* is on the host: set variables, existing paths, settings keys."""
    found = [name for name in source.env if _set(env, name)]
    found += [path for path in source.paths if (home / path).exists()]
    if source.settings_keys:
        loaded = _read_settings(profile, home) if settings is None else settings
        found += [
            f"{key} in {path}"
            for path, data in loaded.items()
            for key in source.settings_keys
            if key in data
        ]
    return found


def _present_names(
    profile: ProviderProfile,
    env: Mapping[str, str],
    home: Path,
    settings: Mapping[str, dict[str, Any]],
) -> list[str]:
    """The names of every declared source present on the host."""
    names: list[str] = []
    for source in profile.credential_sources:
        if isinstance(source, ExternalCredential):
            names += _external_found(profile, source, env, home, settings)
        elif _present_kind(source, env, home) is not None:
            names.append(_name(source))
    return names


def _withheld(
    profile: ProviderProfile,
    env: Mapping[str, str],
    home: Path,
    present: list[str],
    delivered: set[str],
) -> tuple[str, ...]:
    """The names of the host's auth state a plan delivering *delivered* leaves out."""
    on_host = {name for name in _host_auth_names(profile) if _set(env, name)}
    on_host |= {path for path in profile.auth_files if (home / path).is_file()}
    on_host |= set(present)
    return tuple(sorted(on_host - delivered))


def _delivered(env: Mapping[str, str], files: tuple[StagedCredentialFile, ...]) -> set[str]:
    return set(env) | {staged.home_path for staged in files}


def _credential_env_names(profile: ProviderProfile) -> set[str]:
    names: set[str] = set()
    for source in profile.credential_sources:
        if isinstance(source, EnvCredential):
            names.add(source.name)
        elif isinstance(source, ExternalCredential):
            names.update(source.env)
    return names


def _present_kind(
    source: _DeliverableSource, env: Mapping[str, str], home: Path
) -> CredentialKind | None:
    if isinstance(source, EnvCredential):
        return source.kind if _set(env, source.name) else None
    path = home / source.path
    try:
        contents = path.read_bytes()
    except OSError:
        return None
    return source.kind_of(contents)


def _set(env: Mapping[str, str], name: str) -> bool:
    value = env.get(name)
    return bool(value and value.strip())


def _name(source: _DeliverableSource) -> str:
    return source.name if isinstance(source, EnvCredential) else source.path


def _host_auth_names(profile: ProviderProfile) -> set[str]:
    names = set(profile.auth_env_vars)
    if profile.endpoint is not None:
        names.add(profile.endpoint.base_url_env)
        names.update(profile.endpoint.companion_env)
    return names


def _kinds(kinds: frozenset[CredentialKind]) -> str:
    return ", ".join(sorted(kind.value for kind in kinds))


def _unavailable(
    profile: ProviderProfile, allowed: frozenset[CredentialKind], present: list[str]
) -> str:
    wanted = [
        _name(source)
        for source in profile.credential_sources
        if not isinstance(source, ExternalCredential)
        and (source.kind is None or source.kind in allowed)
    ]
    found = f"; present but not allowed: {', '.join(present)}" if present else ""
    return (
        f"no {_kinds(allowed)} credential for {profile.name} is available: "
        f"provide one of {', '.join(wanted)}{found}"
    )
