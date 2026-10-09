"""The readiness probe: version and authentication without a turn."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from agentshim import Agent, ApprovalPolicy, AuthState, NativePermissions, probe_provider
from agentshim.core.status import ProbeOutput, parse_version
from agentshim.providers import get_probe_spec, provider_names
from agentshim.providers.claude.probe import read_auth as read_claude_auth
from agentshim.providers.codex.probe import read_auth as read_codex_auth
from agentshim.testing import FakeConfinement, FakeExecutor, FakeRun, probe_executor
from hypothesis import given
from hypothesis import strategies as st

if TYPE_CHECKING:
    from agentshim.execution.executor import CommandRequest

ENV = {"PATH": "/usr/bin"}
HAS_STATUS_COMMAND = ("claude", "codex")
NO_STATUS_COMMAND = ("copilot", "gemini", "opencode")


def _probe(provider: str, executor: FakeExecutor, env: dict[str, str] | None = None):  # noqa: ANN202
    return probe_provider(provider, executor=executor, env=ENV if env is None else env)


def _ran(executor: FakeExecutor) -> list[list[str]]:
    return [list(request.argv)[1:] for request in executor.requests]


def test_the_provider_set_is_covered() -> None:
    assert sorted((*HAS_STATUS_COMMAND, *NO_STATUS_COMMAND)) == provider_names()


@pytest.mark.parametrize("provider", provider_names())
def test_an_installed_provider_reports_its_path_and_version(provider: str) -> None:
    executor = probe_executor(provider, version="4.5.6")
    status = _probe(provider, executor)
    assert status.provider == provider
    assert status.binary_found
    assert status.path == f"/usr/local/bin/{provider}"
    assert status.version == "4.5.6"


@pytest.mark.parametrize("provider", provider_names())
def test_a_probe_runs_only_the_declared_commands(provider: str) -> None:
    executor = probe_executor(provider)
    _probe(provider, executor)
    spec = get_probe_spec(provider)
    expected = [list(spec.version_argv)]
    if spec.auth is not None:
        expected.append(list(spec.auth.argv))
    assert _ran(executor) == expected


@pytest.mark.parametrize("provider", HAS_STATUS_COMMAND)
@pytest.mark.parametrize("state", [AuthState.KNOWN_OK, AuthState.FAILED, AuthState.UNKNOWN])
def test_the_status_command_decides_authentication(provider: str, state: AuthState) -> None:
    status = _probe(provider, probe_executor(provider, auth=state))
    assert status.auth is state
    assert status.auth_detail
    if state is AuthState.FAILED:
        assert (
            f"{provider} auth login" in status.auth_detail
            or f"{provider} login" in status.auth_detail
        )


@pytest.mark.parametrize("provider", NO_STATUS_COMMAND)
@pytest.mark.parametrize("state", list(AuthState))
def test_a_cli_without_a_status_command_is_unknown_never_failed(
    provider: str, state: AuthState
) -> None:
    status = _probe(provider, probe_executor(provider, auth=state))
    assert status.auth is AuthState.UNKNOWN
    assert "no authentication status command" in status.auth_detail


@pytest.mark.parametrize("provider", provider_names())
def test_a_missing_binary_is_a_result_and_runs_nothing(provider: str) -> None:
    executor = probe_executor(provider, installed=False)
    status = _probe(provider, executor)
    assert not status.binary_found
    assert (status.path, status.version) == (None, None)
    assert status.auth is AuthState.UNKNOWN
    assert "not found" in status.auth_detail
    assert executor.requests == []


@pytest.mark.parametrize("provider", provider_names())
def test_a_version_that_times_out_leaves_the_version_unknown(provider: str) -> None:
    spec = get_probe_spec(provider)

    def pick(request: CommandRequest) -> FakeRun:
        return FakeRun(timeout=list(request.argv)[1:] == list(spec.version_argv))

    status = _probe(provider, FakeExecutor(pick))
    assert status.binary_found
    assert status.version is None


@pytest.mark.parametrize("provider", HAS_STATUS_COMMAND)
def test_a_status_command_that_times_out_is_unknown_not_failed(provider: str) -> None:
    spec = get_probe_spec(provider)
    assert spec.auth is not None

    def pick(request: CommandRequest) -> FakeRun:
        if list(request.argv)[1:] == list(spec.auth.argv):  # type: ignore[union-attr]
            return FakeRun(timeout=True)
        return FakeRun(stdout=["1.0.0\n"])

    status = _probe(provider, FakeExecutor(pick))
    assert status.auth is AuthState.UNKNOWN
    assert status.version == "1.0.0"
    assert "timed out" in status.auth_detail


def test_codex_with_an_api_key_in_the_environment_is_not_failed() -> None:
    executor = probe_executor("codex", auth=AuthState.FAILED)
    assert _probe("codex", executor).auth is AuthState.FAILED
    for key in ("CODEX_API_KEY", "OPENAI_API_KEY"):
        keyed = _probe("codex", probe_executor("codex", auth=AuthState.FAILED), {**ENV, key: "k"})
        assert keyed.auth is AuthState.UNKNOWN
        assert "API key" in keyed.auth_detail


def test_the_probe_runs_through_the_confinement() -> None:
    confinement = FakeConfinement(env={"HOME": "/agent"})
    executor = probe_executor("claude")
    status = probe_provider("claude", executor=executor, confinement=confinement)
    assert status.auth is AuthState.KNOWN_OK
    assert status.version == "1.2.3"
    assert [wrapped[0] for wrapped, _ in confinement.wraps] == ["claude", "claude"]
    assert all(r.argv[0] == "fake-confine" for r in executor.requests)
    assert all(r.env["HOME"] == "/agent" for r in executor.requests)


def test_a_confinement_and_an_env_conflict() -> None:
    with pytest.raises(ValueError, match="env conflicts with confinement"):
        probe_provider("claude", confinement=FakeConfinement(), env=ENV)


def test_an_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown provider"):
        probe_provider("nope", executor=probe_executor("claude"), env=ENV)


def test_an_agent_probes_the_executor_it_runs_turns_on() -> None:
    executor = probe_executor("codex", auth=AuthState.FAILED)
    agent = Agent(
        "codex",
        executor=executor,
        env=ENV,
        permissions=NativePermissions.bypass(),
        approvals=ApprovalPolicy.DENY,
    )
    status = agent.probe()
    assert (status.provider, status.auth) == ("codex", AuthState.FAILED)
    assert status.version == "1.2.3"


def test_an_agent_built_on_a_transport_cannot_probe() -> None:
    inner = Agent(
        "claude",
        executor=probe_executor("claude"),
        env=ENV,
        permissions=NativePermissions.bypass(),
        approvals=ApprovalPolicy.DENY,
    )
    agent = Agent(
        inner.transport,
        permissions=NativePermissions.bypass(),
        approvals=ApprovalPolicy.DENY,
    )
    with pytest.raises(ValueError, match="provider name"):
        agent.probe()


# -- reading output


@pytest.mark.parametrize(
    ("text", "version"),
    [
        ("2.1.295 (Claude Code)\n", "2.1.295"),
        ("codex-cli 0.160.0\n", "0.160.0"),
        ("0.26.0\n", "0.26.0"),
        ("GitHub Copilot CLI 1.0.94.\nRun 'copilot update'.\n", "1.0.94"),
        ("tool 1.2.3-beta.4 build\n", "1.2.3-beta.4"),
        ("no number here\n", None),
        ("", None),
    ],
)
def test_the_version_is_the_first_dotted_number(text: str, version: str | None) -> None:
    assert parse_version(text) == version


@given(st.text())
def test_a_parsed_version_is_always_a_substring(text: str) -> None:
    version = parse_version(text)
    assert version is None or version in text


@given(
    st.recursive(
        st.none() | st.booleans() | st.integers() | st.text(),
        lambda c: st.lists(c) | st.dictionaries(st.text(), c),
    )
)
def test_claude_is_only_ok_or_failed_when_logged_in_is_a_boolean(payload: object) -> None:
    check = read_claude_auth(ProbeOutput(0, json.dumps(payload), ""), {})
    logged_in = payload.get("loggedIn") if isinstance(payload, dict) else None
    if logged_in is True:
        assert check.state is AuthState.KNOWN_OK
    elif logged_in is False:
        assert check.state is AuthState.FAILED
    else:
        assert check.state is AuthState.UNKNOWN


@given(st.text(), st.integers(0, 3))
def test_claude_never_fails_on_text_that_is_not_a_status(stdout: str, code: int) -> None:
    check = read_claude_auth(ProbeOutput(code, stdout, ""), {})
    assert check.state in set(AuthState)


@given(st.text(alphabet=st.characters(blacklist_categories=["Cs"])), st.integers(0, 3))
def test_codex_failure_needs_the_not_logged_in_line(text: str, code: int) -> None:
    check = read_codex_auth(ProbeOutput(code, text, ""), {})
    lines = [line.strip() for line in text.splitlines()]
    if check.state is AuthState.FAILED:
        assert any(line.startswith("Not logged in") for line in lines)
    if check.state is AuthState.KNOWN_OK:
        assert any(line.startswith("Logged in") for line in lines)
