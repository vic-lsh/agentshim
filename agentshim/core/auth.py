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


CredentialSource = EnvCredential | FileCredential


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

    delivered = set(delivered_env) | {staged.home_path for staged in files}
    on_host = {name for name in _host_auth_names(profile) if _set(env, name)}
    on_host |= {path for path in profile.auth_files if (home / path).is_file()}
    on_host |= set(present)
    return AuthPlan(
        provider=profile.name,
        credential=kind,
        source=_name(chosen),
        endpoint=policy.endpoint,
        env=MappingProxyType(delivered_env),
        files=files,
        withheld=tuple(sorted(on_host - delivered)),
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
) -> tuple[CredentialSource, CredentialKind, list[str]]:
    """The first present allowed source, its kind, and every present source's name."""
    chosen: tuple[CredentialSource, CredentialKind] | None = None
    present: list[str] = []
    for source in profile.credential_sources:
        kind = _present_kind(source, env, home)
        if kind is None:
            continue
        present.append(_name(source))
        if chosen is None and kind in allowed:
            chosen = (source, kind)
    if chosen is None:
        raise AuthUnavailableError(_unavailable(profile, allowed, present))
    return chosen[0], chosen[1], present


def _present_kind(
    source: CredentialSource, env: Mapping[str, str], home: Path
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


def _name(source: CredentialSource) -> str:
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
        if source.kind is None or source.kind in allowed
    ]
    found = f"; present but not allowed: {', '.join(present)}" if present else ""
    return (
        f"no {_kinds(allowed)} credential for {profile.name} is available: "
        f"provide one of {', '.join(wanted)}{found}"
    )
