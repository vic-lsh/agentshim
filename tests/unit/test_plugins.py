import json
from pathlib import Path

import pytest
from agentshim import (
    Agent,
    ApprovalPolicy,
    CliAgent,
    CliTimeoutError,
    CommandRequest,
    ConfigScope,
    McpScope,
    NativePermissions,
    PluginRef,
    ProviderCapabilityError,
    SessionPlugins,
    SkillScope,
    TransportKind,
)
from agentshim.testing import (
    ClaudeStreamPeers,
    CodexScript,
    FakeClock,
    FakeExecutor,
    FakeRun,
    SequentialIds,
    scripted_turn,
)
from hypothesis import given
from hypothesis import strategies as st


def test_plugin_paths_and_names_are_unambiguous(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        PluginRef(Path("relative"), "example")
    with pytest.raises(ValueError, match="invalid plugin name"):
        PluginRef(tmp_path, "example@marketplace")
    with pytest.raises(ValueError, match="absolute"):
        SessionPlugins(Path("relative"))
    assert PluginRef(tmp_path, "example").marketplace == tmp_path


def test_unsupported_provider_refuses_even_an_empty_selection(tmp_path: Path) -> None:
    agent = CliAgent("gemini", executor=FakeExecutor([]))
    with pytest.raises(ProviderCapabilityError, match="session plugins"):
        agent.start_session(plugins=SessionPlugins(tmp_path / "session"))


def _marketplace(root: Path) -> PluginRef:

    manifest = root / ".claude-plugin" / "marketplace.json"
    manifest.parent.mkdir(parents=True)
    (root / "skills").mkdir()
    manifest.write_text(
        json.dumps(
            {
                "name": "example-market",
                "plugins": [
                    {
                        "name": "example",
                        "source": "./skills",
                        "strict": False,
                        "skills": ["./nested/optimize"],
                    },
                ],
            }
        )
    )
    return PluginRef(root, "example")


@pytest.mark.parametrize("provider", ["claude", "codex"])
@pytest.mark.parametrize("project_skills", [False, True])
@pytest.mark.parametrize("project_config", [False, True])
def test_plugin_session_contract(
    tmp_path: Path,
    provider: str,
    *,
    project_skills: bool,
    project_config: bool,
) -> None:

    ref = _marketplace(tmp_path / "marketplace")
    user = tmp_path / "user"
    state = user / (".claude" if provider == "claude" else ".codex")
    state.mkdir(parents=True)
    credential = ".credentials.json" if provider == "claude" else "auth.json"
    (state / credential).write_text('{"token":"example"}')
    (state / "config.toml").write_text('notify = ["operator"]')
    skill = user / ".agents" / "skills" / "operator" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("operator skill")

    def run(request: CommandRequest) -> FakeRun:
        if request.argv[1] == "plugin":
            if request.argv[2] in {"install", "add"}:
                state_home = Path(request.cwd or "")
                if provider == "claude":
                    (state_home / "settings.json").write_text(
                        json.dumps(
                            {
                                "enabledPlugins": {"example@example-market": True},
                            }
                        )
                    )
                else:
                    (state_home / "config.toml").write_text(
                        '[plugins."example@example-market"]\nenabled = true\n'
                    )
            return FakeRun()
        return scripted_turn(provider, text="ok")

    executor = FakeExecutor(run)
    agent = CliAgent(provider, executor=executor, env={"HOME": str(user)})
    home = tmp_path / "session"
    session = agent.start_session(
        plugins=SessionPlugins(home, (ref,)),
        skill_scope=SkillScope.PROJECT if project_skills else SkillScope.ALL,
        config_scope=ConfigScope.PROJECT if project_config else ConfigScope.ALL,
    )
    assert session.turn("hello").text == "ok"
    marketplace, install, turn = executor.requests
    assert list(marketplace.argv[1:]) == ["plugin", "marketplace", "add", str(ref.marketplace)]
    assert list(install.argv[1:]) == [
        "plugin",
        "install" if provider == "claude" else "add",
        "example@example-market",
    ]
    variable = "CLAUDE_CONFIG_DIR" if provider == "claude" else "CODEX_HOME"
    assert all(r.env[variable] == str(home) for r in executor.requests)
    assert (home / credential).is_symlink()
    assert (home / credential).resolve() == state / credential
    installed = home / ("settings.json" if provider == "claude" else "config.toml")
    assert "example@example-market" in installed.read_text()
    assert "operator" not in installed.read_text()
    if provider == "claude":
        assert turn.argv[turn.argv.index("--setting-sources") + 1] == "project,local"
        settings = json.loads(turn.argv[turn.argv.index("--settings") + 1])
        assert settings["enabledPlugins"] == {"example@example-market": True}
    else:
        assert "--ignore-user-config" in turn.argv
        overrides = [turn.argv[i + 1] for i, flag in enumerate(turn.argv) if flag == "--config"]
        assert overrides.index("features.plugins=true") > overrides.index("features.plugins=false")
        assert 'plugins={"example@example-market" = {enabled = true}}' in overrides
        assert str(skill) in next(v for v in overrides if v.startswith("skills.config="))
    session.turn("again")
    assert len(executor.requests) == 4  # install is not repeated on resume


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_installer_failure_cleans_home_and_never_starts_agent(
    tmp_path: Path, provider: str
) -> None:

    ref = _marketplace(tmp_path / "marketplace")
    executor = FakeExecutor(FakeRun(returncode=1, stderr=["installer failed"]))
    agent = CliAgent(provider, executor=executor, env={"HOME": str(tmp_path / "user")})
    home = tmp_path / "session"
    session = agent.start_session(plugins=SessionPlugins(home, (ref,)))
    with pytest.raises(ValueError, match="installer failed"):
        session.turn("hello")
    assert not home.exists()
    assert len(executor.requests) == 1


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_plugin_home_never_adopts_existing_configuration(tmp_path: Path, provider: str) -> None:
    home = tmp_path / "session"
    home.mkdir()
    config = home / "config.toml"
    config.write_text('notify = ["operator"]')
    executor = FakeExecutor([])
    agent = CliAgent(provider, executor=executor, env={"HOME": str(tmp_path / "user")})
    session = agent.start_session(plugins=SessionPlugins(home))
    with pytest.raises(ValueError, match="fresh"):
        session.turn("hello")
    assert config.read_text() == 'notify = ["operator"]'
    assert executor.requests == []


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_stream_plugin_contract(tmp_path: Path, provider: str) -> None:
    from_data = _marketplace(tmp_path / "marketplace")
    home = tmp_path / "session"
    factory = ClaudeStreamPeers().build if provider == "claude" else CodexScript().peer
    executor = FakeExecutor([], peers=factory)
    agent = Agent(
        provider,
        transport=TransportKind.STREAM,
        executor=executor,
        env={"HOME": str(tmp_path / "user")},
        permissions=NativePermissions.bypass(),
        approvals=ApprovalPolicy.DENY,
        clock=FakeClock(),
        ids=SequentialIds(),
    )
    session = agent.session(cwd=str(tmp_path), plugins=SessionPlugins(home, (from_data,)))
    try:
        session.warm()
        assert len(executor.requests) == 2
        spawn = executor.spawns[0]
        variable = "CLAUDE_CONFIG_DIR" if provider == "claude" else "CODEX_HOME"
        assert spawn.env[variable] == str(home)
        assert "example@example-market" in " ".join(spawn.argv)
        if provider == "codex":
            assert "--ignore-user-config" not in spawn.argv
    finally:
        session.close()


def test_codex_session_mcp_refuses_a_plugin_with_bundled_servers(tmp_path: Path) -> None:
    ref = _marketplace(tmp_path / "marketplace")
    source = ref.marketplace / "skills"
    source.mkdir(exist_ok=True)
    (source / ".mcp.json").write_text('{"server":{"command":"example"}}')
    agent = CliAgent("codex", executor=FakeExecutor([]), env={"HOME": str(tmp_path / "user")})
    session = agent.start_session(
        plugins=SessionPlugins(tmp_path / "session", (ref,)),
        mcp_scope=McpScope.SESSION,
    )
    with pytest.raises(ProviderCapabilityError, match=r"McpScope\.SESSION"):
        session.turn("hello")


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_empty_selection_is_isolated_without_running_installers(
    tmp_path: Path, provider: str
) -> None:
    executor = FakeExecutor(scripted_turn(provider, text="ok"))
    agent = CliAgent(provider, executor=executor, env={"HOME": str(tmp_path / "user")})
    home = tmp_path / "session"
    assert agent.run("hello", plugins=SessionPlugins(home)).text == "ok"
    assert len(executor.requests) == 1
    assert executor.requests[0].env[
        "CLAUDE_CONFIG_DIR" if provider == "claude" else "CODEX_HOME"
    ] == str(home)
    assert (home / ".agentshim-plugins.json").is_file()


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_home_is_bound_to_the_original_selection(tmp_path: Path, provider: str) -> None:
    ref = _marketplace(tmp_path / "marketplace")
    executor = FakeExecutor(scripted_turn(provider, text="ok"))
    agent = CliAgent(provider, executor=executor, env={"HOME": str(tmp_path / "user")})
    home = tmp_path / "session"
    agent.run("hello", plugins=SessionPlugins(home))
    with pytest.raises(ValueError, match="exact session plugin selection"):
        agent.run("hello", plugins=SessionPlugins(home, (ref,)))
    assert len(executor.requests) == 1


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_installer_timeout_cleans_the_home(tmp_path: Path, provider: str) -> None:
    ref = _marketplace(tmp_path / "marketplace")
    executor = FakeExecutor(FakeRun(timeout=True))
    agent = CliAgent(provider, executor=executor, env={"HOME": str(tmp_path / "user")})
    home = tmp_path / "session"
    session = agent.start_session(plugins=SessionPlugins(home, (ref,)))
    with pytest.raises(CliTimeoutError):
        session.turn("hello")
    assert not home.exists()
    assert executor.requests[0].timeout == 60
    assert session.adopt("other")  # no failed preparation leaves the session busy


@given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_@./ ", max_size=50))
def test_plugin_identifiers_cannot_be_cli_options_or_qualified_selectors(name: str) -> None:
    valid = (
        bool(name)
        and name[0] in "abcdefghijklmnopqrstuvwxyz0123456789"
        and all(c in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in name)
    )
    if valid:
        assert PluginRef(Path("/marketplace"), name).name == name
    else:
        with pytest.raises(ValueError, match="invalid plugin name"):
            PluginRef(Path("/marketplace"), name)


def test_claude_project_plugin_selection_cannot_expand_the_session(tmp_path: Path) -> None:
    ref = _marketplace(tmp_path / "marketplace")
    workspace = tmp_path / "project"
    settings = workspace / ".claude/settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"enabledPlugins": {"operator@other": True}}))
    executor = FakeExecutor(scripted_turn("claude", text="ok"))
    agent = CliAgent("claude", executor=executor, env={"HOME": str(tmp_path / "user")})
    agent.run("hello", cwd=str(workspace), plugins=SessionPlugins(tmp_path / "session", (ref,)))
    argv = executor.requests[-1].argv
    inline = json.loads(argv[argv.index("--settings") + 1])
    assert inline["enabledPlugins"] == {"operator@other": False, "example@example-market": True}
