"""Codex's CLI sandbox as a provider option.

The properties here are what a caller relies on: a config either renders to
argv that imposes exactly that sandbox, or is rejected before any process
starts; the sandbox never disturbs the rest of the command line; and without
a config nothing changes from earlier releases.
"""

from __future__ import annotations

import shlex
import sys
from dataclasses import replace

import pytest
from agentshim import ArgvContext, CliAgent, ProviderCapabilityError
from agentshim.providers.codex import (
    BYPASS_FLAG,
    IGNORE_RULES_FLAG,
    SANDBOX_MODES,
    CodexProvider,
    CodexSandboxConfig,
    parse_sandbox,
)
from agentshim.testing import FakeExecutor, scripted_turn
from hypothesis import assume, example, given
from hypothesis import strategies as st

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on the 3.10 CI leg
    import tomli as tomllib  # pyright: ignore[reportMissingImports]

_WW_KEYS = {
    "sandbox_workspace_write.writable_roots",
    "sandbox_workspace_write.network_access",
    "sandbox_workspace_write.exclude_slash_tmp",
    "sandbox_workspace_write.exclude_tmpdir_env_var",
}

# -- strategies -------------------------------------------------------------

#: Path segments with the characters that break naive quoting: quotes,
#: backslashes, control characters, spaces, non-BMP code points.
_segments = st.text(
    alphabet=st.one_of(
        st.characters(exclude_characters="\x00/", exclude_categories=["Cs"]),
        st.sampled_from(['"', "\\", "\n", "\t", " ", "'", "$", "\x7f", "\U0001f600"]),
    ),
    min_size=1,
    max_size=12,
)
absolute_paths = st.lists(_segments, min_size=1, max_size=4).map(
    lambda parts: "/" + "/".join(parts)
)
relative_paths = st.lists(_segments, min_size=1, max_size=3).map("/".join)

workspace_write = st.builds(
    CodexSandboxConfig,
    mode=st.just("workspace-write"),
    writable_roots=st.lists(absolute_paths, max_size=4).map(tuple),
    network_access=st.booleans(),
    writable_tmp=st.booleans(),
)
configs = st.one_of(
    workspace_write,
    st.sampled_from(["read-only", "danger-full-access"]).map(
        lambda m: CodexSandboxConfig(mode=m)  # pyright: ignore[reportArgumentType]
    ),
)

#: Shell words with the characters that could break out of a rule's string
#: literal or a naive split: quotes, backslashes, ``$``, spaces, non-BMP.
_words = st.text(
    alphabet=st.one_of(
        st.characters(exclude_categories=["Cs", "Cc"]),
        st.sampled_from(['"', "\\", "'", "$", " ", "(", ")", "]", ",", "\U0001f600"]),
    ),
    min_size=1,
    max_size=10,
)
#: ``excluded_commands`` entries, quoted with ``shlex.join`` so they split back.
excluded_commands = st.lists(
    st.lists(_words, min_size=1, max_size=4).map(shlex.join), min_size=1, max_size=3
).map(tuple)
exempting = st.builds(
    CodexSandboxConfig,
    mode=st.sampled_from(["read-only", "workspace-write"]),
    excluded_commands=excluded_commands,
)

#: A Codex home outside every directory the strategies make writable.
_HOME = "/srv/agent/codex-home"

#: Caller extras that do not themselves configure the sandbox.
_extra_args = st.lists(
    st.text(min_size=1, max_size=20).filter(
        lambda a: a not in {"--config", BYPASS_FLAG, IGNORE_RULES_FLAG}
    ),
    max_size=3,
)
contexts = st.builds(
    ArgvContext,
    binary_path=st.just("/usr/local/bin/codex"),
    model=st.one_of(st.none(), st.sampled_from(["gpt-6-luna", "o3"])),
    env=st.one_of(st.just({}), st.builds(lambda p: {"PATH": p}, absolute_paths)),
    resume_session_id=st.one_of(st.none(), st.uuids().map(str)),
    reasoning_effort=st.one_of(st.none(), st.sampled_from(["low", "high"])),
    schema_inline=st.none(),
    schema_path=st.one_of(st.none(), st.just("/w/schema.json")),
    mcp_argv=st.just(("--config", 'mcp_servers.srv.command="srv"')),
    extra_args=_extra_args.map(tuple),
)


def _overrides(argv: list[str]) -> list[tuple[str, str]]:
    return [
        tuple(argv[i + 1].partition("=")[::2])  # type: ignore[misc]
        for i in range(len(argv) - 1)
        if argv[i] == "--config"
    ]


def _without_sandbox(argv: list[str]) -> list[str]:
    """Drop the sandbox segment: the bypass flag or the sandbox overrides."""
    out: list[str] = []
    skip = False
    for index, arg in enumerate(argv):
        if skip:
            skip = False
            continue
        if arg in {BYPASS_FLAG, IGNORE_RULES_FLAG}:
            continue
        if arg == "--config" and index + 1 < len(argv):
            key = argv[index + 1].partition("=")[0]
            if key in {"sandbox_mode", "approval_policy"} | _WW_KEYS:
                skip = True
                continue
        out.append(arg)
    return out


# -- construction -----------------------------------------------------------


class TestConfigValidation:
    def test_the_default_is_workspace_write_with_codex_defaults(self) -> None:
        config = CodexSandboxConfig()
        assert config.mode == "workspace-write"
        assert config.writable_roots == ()
        assert config.network_access is False
        assert config.writable_tmp is True
        assert config.excluded_commands == ()

    @given(st.text().filter(lambda m: m not in SANDBOX_MODES))
    def test_an_unknown_mode_is_rejected(self, mode: str) -> None:
        with pytest.raises(ValueError, match="mode must be one of"):
            CodexSandboxConfig(mode=mode)  # type: ignore[arg-type]

    @given(relative_paths.filter(lambda p: not p.startswith("/")))
    def test_a_relative_root_is_rejected(self, root: str) -> None:
        """Codex would resolve it against a cwd the caller may not expect."""
        with pytest.raises(ValueError, match="absolute"):
            CodexSandboxConfig(writable_roots=[root])

    @given(absolute_paths, st.integers(min_value=0, max_value=10))
    def test_a_nul_byte_in_a_root_is_rejected(self, root: str, at: int) -> None:
        with pytest.raises(ValueError, match="NUL"):
            CodexSandboxConfig(writable_roots=[root[:at] + "\x00" + root[at:]])

    @pytest.mark.parametrize("surrogate", [chr(0xD800), chr(0xDC80), chr(0xDFFF)])
    def test_a_root_that_cannot_be_encoded_is_rejected(self, surrogate: str) -> None:
        """It could reach neither TOML nor argv; fail here, not at launch."""
        with pytest.raises(ValueError, match="UTF-8"):
            CodexSandboxConfig(writable_roots=[f"/data/{surrogate}"])

    def test_a_bare_string_is_not_iterated_into_roots(self) -> None:
        with pytest.raises(TypeError, match="sequence of paths"):
            CodexSandboxConfig(writable_roots="/data")

    def test_a_set_is_rejected_because_its_order_is_arbitrary(self) -> None:
        with pytest.raises(TypeError, match="sequence of paths"):
            CodexSandboxConfig(writable_roots={"/a", "/b"})  # type: ignore[arg-type]

    @pytest.mark.parametrize("root", [1, None, b"/data"])
    def test_a_non_string_root_is_rejected(self, root: object) -> None:
        with pytest.raises(TypeError, match="must be str"):
            CodexSandboxConfig(writable_roots=[root])  # type: ignore[list-item]

    @pytest.mark.parametrize("field", ["network_access", "writable_tmp"])
    @pytest.mark.parametrize("value", [1, 0, "true", None])
    def test_a_truthy_non_bool_flag_is_rejected(self, field: str, value: object) -> None:
        with pytest.raises(TypeError, match="must be a bool"):
            CodexSandboxConfig(**{field: value})  # type: ignore[arg-type]

    @given(
        st.sampled_from(["read-only", "danger-full-access"]),
        workspace_write.filter(
            lambda c: bool(c.writable_roots) or c.network_access or not c.writable_tmp
        ),
    )
    def test_workspace_write_options_are_rejected_on_other_modes(
        self, mode: str, widened: CodexSandboxConfig
    ) -> None:
        """Codex would ignore them, so the caller's intent would be lost silently."""
        with pytest.raises(ValueError, match="only apply to workspace-write"):
            replace(widened, mode=mode)  # type: ignore[arg-type]

    @given(st.lists(absolute_paths, max_size=4))
    def test_equal_configs_compare_and_hash_equal_whatever_the_sequence_type(
        self, roots: list[str]
    ) -> None:
        from_list = CodexSandboxConfig(writable_roots=roots)
        from_tuple = CodexSandboxConfig(writable_roots=tuple(roots))
        assert from_list == from_tuple
        assert hash(from_list) == hash(from_tuple)
        assert isinstance(from_list.writable_roots, tuple)

    def test_a_bare_command_string_is_not_iterated_into_commands(self) -> None:
        """``"sdo detector check"`` would otherwise exempt ``s``, ``d``, ``o``..."""
        with pytest.raises(TypeError, match="sequence of commands"):
            CodexSandboxConfig(excluded_commands="sdo detector check")

    def test_a_set_of_commands_is_rejected_because_its_order_is_arbitrary(self) -> None:
        with pytest.raises(TypeError, match="sequence of commands"):
            CodexSandboxConfig(excluded_commands={"a", "b"})  # type: ignore[arg-type]

    @pytest.mark.parametrize("command", [1, None, b"sdo", ["sdo", "check"]])
    def test_a_non_string_command_is_rejected(self, command: object) -> None:
        with pytest.raises(TypeError, match="must be str"):
            CodexSandboxConfig(excluded_commands=[command])  # type: ignore[list-item]

    @given(_words, st.sampled_from(["\x00", "\n", "\r", "\t", "\x1b", "\x7f"]))
    def test_a_control_character_is_rejected(self, word: str, control: str) -> None:
        """A newline would be a second command on the same line, not a word."""
        with pytest.raises(ValueError, match="control character"):
            CodexSandboxConfig(excluded_commands=[f"sdo {word}{control}"])

    @pytest.mark.parametrize("surrogate", [chr(0xD800), chr(0xDFFF)])
    def test_a_command_that_cannot_be_encoded_is_rejected(self, surrogate: str) -> None:
        with pytest.raises(ValueError, match="UTF-8"):
            CodexSandboxConfig(excluded_commands=[f"sdo {surrogate}"])

    @pytest.mark.parametrize("command", ["sdo 'detector", 'sdo "check', "sdo \\"])
    def test_a_command_that_does_not_split_into_words_is_rejected(self, command: str) -> None:
        with pytest.raises(ValueError, match="not valid shell words"):
            CodexSandboxConfig(excluded_commands=[command])

    @pytest.mark.parametrize("command", ["", "   ", "''", "sdo ''"])
    def test_an_empty_command_or_word_is_rejected(self, command: str) -> None:
        """An empty prefix would exempt every command."""
        with pytest.raises(ValueError, match="non-empty words"):
            CodexSandboxConfig(excluded_commands=[command])

    @given(excluded_commands)
    def test_exemptions_are_rejected_without_a_sandbox(self, commands: tuple[str, ...]) -> None:
        with pytest.raises(ValueError, match="danger-full-access has none"):
            CodexSandboxConfig(mode="danger-full-access", excluded_commands=commands)

    @given(st.sampled_from(["read-only", "workspace-write"]), excluded_commands)
    def test_exemptions_are_accepted_on_both_sandboxed_modes(
        self, mode: str, commands: tuple[str, ...]
    ) -> None:
        config = CodexSandboxConfig(mode=mode, excluded_commands=list(commands))  # type: ignore[arg-type]
        assert config.excluded_commands == commands
        assert hash(config) == hash(replace(config, excluded_commands=commands))

    def test_the_config_is_immutable(self) -> None:
        config = CodexSandboxConfig()
        with pytest.raises(AttributeError):
            config.mode = "read-only"  # type: ignore[misc]


class TestProviderOption:
    def test_the_default_provider_has_no_sandbox(self) -> None:
        assert CodexProvider().sandbox is None

    @pytest.mark.parametrize("value", [True, False, "read-only", {"mode": "read-only"}])
    def test_anything_but_a_config_or_none_is_rejected(self, value: object) -> None:
        """No ``True`` shorthand: no one mode is right for every caller."""
        with pytest.raises(TypeError, match="CodexSandboxConfig or None"):
            CodexProvider(sandbox=value)  # type: ignore[arg-type]


# -- rendering properties ---------------------------------------------------


class TestRenderedArgv:
    @given(configs, contexts)
    @example(
        CodexSandboxConfig(mode="read-only"),
        ArgvContext("/bin/codex", None, {}, "t", None, None, None),
    )
    def test_a_sandboxed_turn_never_carries_the_bypass_flag(
        self, config: CodexSandboxConfig, ctx: ArgvContext
    ) -> None:
        assert BYPASS_FLAG not in CodexProvider(sandbox=config).build_argv(ctx)

    @given(configs, contexts)
    def test_parse_sandbox_recovers_exactly_the_config(
        self, config: CodexSandboxConfig, ctx: ArgvContext
    ) -> None:
        """The rendered argv means this config and nothing else."""
        argv = CodexProvider(sandbox=config).build_argv(ctx)
        assert parse_sandbox(argv) == config

    @given(configs, contexts)
    def test_every_sandbox_override_is_valid_toml_with_the_intended_value(
        self, config: CodexSandboxConfig, ctx: ArgvContext
    ) -> None:
        overrides = dict(_overrides(CodexProvider(sandbox=config).build_argv(ctx)))
        parsed = {
            key: tomllib.loads(f"{key} = {value}")
            for key, value in overrides.items()
            if key in {"sandbox_mode", "approval_policy"} | _WW_KEYS
        }
        assert parsed["sandbox_mode"]["sandbox_mode"] == config.mode
        assert parsed["approval_policy"]["approval_policy"] == "never"
        if config.mode == "workspace-write":
            table = {
                key.partition(".")[2]: doc["sandbox_workspace_write"][key.partition(".")[2]]
                for key, doc in parsed.items()
                if key in _WW_KEYS
            }
            assert table == {
                "writable_roots": list(config.writable_roots),
                "network_access": config.network_access,
                "exclude_slash_tmp": not config.writable_tmp,
                "exclude_tmpdir_env_var": not config.writable_tmp,
            }

    @given(configs, contexts)
    def test_each_sandbox_key_is_set_exactly_once(
        self, config: CodexSandboxConfig, ctx: ArgvContext
    ) -> None:
        keys = [key for key, _ in _overrides(CodexProvider(sandbox=config).build_argv(ctx))]
        expected = {"sandbox_mode", "approval_policy"}
        if config.mode == "workspace-write":
            expected |= _WW_KEYS
        for key in expected:
            assert keys.count(key) == 1, key
        assert not (set(keys) & _WW_KEYS - expected)

    @given(workspace_write, contexts)
    def test_workspace_write_pins_every_key_even_at_its_default(
        self, config: CodexSandboxConfig, ctx: ArgvContext
    ) -> None:
        """A user's config.toml must not be able to widen the sandbox."""
        keys = {key for key, _ in _overrides(CodexProvider(sandbox=config).build_argv(ctx))}
        assert keys >= _WW_KEYS

    @given(configs, contexts)
    def test_the_sandbox_does_not_disturb_the_rest_of_the_argv(
        self, config: CodexSandboxConfig, ctx: ArgvContext
    ) -> None:
        """Removing the sandbox segment leaves the unsandboxed argv, in order."""
        sandboxed = CodexProvider(sandbox=config).build_argv(ctx)
        bypassed = CodexProvider().build_argv(ctx)
        assert _without_sandbox(sandboxed) == _without_sandbox(bypassed)

    @given(configs, contexts)
    def test_positionals_come_first_and_caller_extras_come_last(
        self, config: CodexSandboxConfig, ctx: ArgvContext
    ) -> None:
        argv = CodexProvider(sandbox=config).build_argv(ctx)
        assert argv[:2] == [ctx.binary_path, "exec"]
        if ctx.resume_session_id:
            assert argv[2:5] == ["resume", ctx.resume_session_id, "-"]
        else:
            assert "resume" not in argv[:3]
        if ctx.extra_args:
            assert argv[-len(ctx.extra_args) :] == list(ctx.extra_args)

    @given(configs, contexts)
    def test_rendering_is_deterministic(self, config: CodexSandboxConfig, ctx: ArgvContext) -> None:
        assert CodexProvider(sandbox=config).build_argv(ctx) == CodexProvider(
            sandbox=replace(config)
        ).build_argv(ctx)


class TestBypassIsUnchanged:
    @given(contexts)
    def test_without_a_sandbox_the_bypass_flag_is_rendered_and_no_sandbox_key(
        self, ctx: ArgvContext
    ) -> None:
        argv = CodexProvider().build_argv(ctx)
        assert argv.count(BYPASS_FLAG) == 1
        keys = {key for key, _ in _overrides(argv)}
        assert not keys & ({"sandbox_mode", "approval_policy"} | _WW_KEYS)
        assert parse_sandbox(argv) is None

    def test_the_bypass_flag_sits_where_it_always_has(self) -> None:
        argv = CodexProvider().build_argv(
            ArgvContext("/bin/codex", None, {}, None, None, None, None)
        )
        assert argv == ["/bin/codex", "exec", BYPASS_FLAG, "--skip-git-repo-check", "--json"]


class TestParseSandbox:
    def test_argv_with_neither_bypass_nor_mode_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="neither bypasses nor selects"):
            parse_sandbox(["/bin/codex", "exec", "--json"])

    @given(configs)
    def test_unrelated_overrides_are_ignored(self, config: CodexSandboxConfig) -> None:
        argv = CodexProvider(sandbox=config).build_argv(
            ArgvContext("/bin/codex", None, {"PATH": "/usr/bin"}, None, "high", None, None)
        )
        assume("--config" in argv)
        assert parse_sandbox(argv) == config


# -- exec-policy rules ------------------------------------------------------


def _ctx(
    env: dict[str, str] | None = None, *, cwd: str | None = None, resume: str | None = None
) -> ArgvContext:
    return ArgvContext(
        "/bin/codex",
        None,
        {"CODEX_HOME": _HOME} if env is None else env,
        resume,
        None,
        None,
        None,
        cwd=cwd,
    )


class TestRulesIsolation:
    @given(configs, contexts)
    def test_a_sandboxed_turn_without_exemptions_ignores_every_rules_file(
        self, config: CodexSandboxConfig, ctx: ArgvContext
    ) -> None:
        """A ``prefix_rule`` in the user's ``~/.codex/rules`` would run outside the sandbox."""
        assert CodexProvider(sandbox=config).build_argv(ctx).count(IGNORE_RULES_FLAG) == 1

    @given(contexts)
    def test_the_bypass_leaves_rules_alone(self, ctx: ArgvContext) -> None:
        assert IGNORE_RULES_FLAG not in CodexProvider().build_argv(ctx)

    @given(exempting, st.one_of(st.none(), st.uuids().map(str)))
    def test_exemptions_load_rules_on_fresh_and_resumed_turns_alike(
        self, config: CodexSandboxConfig, resume: str | None
    ) -> None:
        provider = CodexProvider(sandbox=config)
        fresh = provider.build_argv(_ctx())
        resumed = provider.build_argv(_ctx(resume=resume or "t"))
        assert IGNORE_RULES_FLAG not in fresh
        assert IGNORE_RULES_FLAG not in resumed
        assert _overrides(fresh) == _overrides(resumed)

    @given(exempting)
    def test_the_rest_of_the_sandbox_is_rendered_unchanged(
        self, config: CodexSandboxConfig
    ) -> None:
        """Exemptions add rules; they never loosen the ``--config`` sandbox."""
        exempt = CodexProvider(sandbox=config).build_argv(_ctx())
        plain = CodexProvider(sandbox=replace(config, excluded_commands=())).build_argv(_ctx())
        assert [a for a in plain if a != IGNORE_RULES_FLAG] == exempt
        assert parse_sandbox(exempt) == replace(config, excluded_commands=())


class TestRulesHome:
    @pytest.mark.parametrize("env", [{}, {"CODEX_HOME": ""}, {"CODEX_HOME": "codex-home"}])
    def test_exemptions_need_an_absolute_codex_home(self, env: dict[str, str]) -> None:
        """Without one Codex reads ``~/.codex/rules``, where the rules are not."""
        provider = CodexProvider(sandbox=CodexSandboxConfig(excluded_commands=["sdo check"]))
        with pytest.raises(ProviderCapabilityError, match="CODEX_HOME"):
            provider.build_argv(_ctx(env))

    @pytest.mark.parametrize(
        ("home", "cwd", "roots", "tmp", "env"),
        [
            ("/work/.codex-home", "/work", [], False, {}),
            ("/work", "/work", [], False, {}),
            ("/data/cache/home", None, ["/data/cache"], False, {}),
            ("/data/home", None, ["/data/home/rules"], False, {}),
            ("/tmp/codex-home", None, [], True, {}),  # noqa: S108 - Codex's own /tmp
            ("/scratch/t/home", None, [], True, {"TMPDIR": "/scratch/t"}),
            ("/work/sub/../.codex-home", "/work", [], False, {}),
        ],
    )
    def test_a_home_the_sandbox_can_write_is_refused(
        self,
        home: str,
        cwd: str | None,
        roots: list[str],
        tmp: bool,  # noqa: FBT001 - a parametrized value
        env: dict[str, str],
    ) -> None:
        """A command could write its own ``allow`` rule there and escape on the next turn."""
        config = CodexSandboxConfig(
            writable_roots=roots, writable_tmp=tmp, excluded_commands=["sdo check"]
        )
        with pytest.raises(ProviderCapabilityError, match="lets commands write"):
            CodexProvider(sandbox=config).build_argv(_ctx({"CODEX_HOME": home, **env}, cwd=cwd))

    @pytest.mark.parametrize(
        ("home", "cwd", "roots", "tmp"),
        [
            ("/srv/codex-home", "/work", ["/data"], True),
            ("/work-home", "/work", [], False),
        ],
    )
    def test_a_home_outside_every_writable_dir_is_accepted(
        self,
        home: str,
        cwd: str,
        roots: list[str],
        tmp: bool,  # noqa: FBT001 - a parametrized value
    ) -> None:
        config = CodexSandboxConfig(
            writable_roots=roots, writable_tmp=tmp, excluded_commands=["sdo check"]
        )
        argv = CodexProvider(sandbox=config).build_argv(_ctx({"CODEX_HOME": home}, cwd=cwd))
        assert IGNORE_RULES_FLAG not in argv

    def test_read_only_writes_nothing_so_any_home_is_safe(self) -> None:
        config = CodexSandboxConfig(mode="read-only", excluded_commands=["sdo check"])
        argv = CodexProvider(sandbox=config).build_argv(
            _ctx({"CODEX_HOME": "/work/.codex-home"}, cwd="/work")
        )
        assert IGNORE_RULES_FLAG not in argv

    def test_a_session_passes_its_cwd_to_the_check(self) -> None:
        """The turn's cwd is the workspace Codex lets commands write."""
        config = CodexSandboxConfig(excluded_commands=["sdo detector check"])
        executor = FakeExecutor(scripted_turn("codex", text="ok"))
        agent = CliAgent(
            CodexProvider(sandbox=config),
            executor=executor,
            env={"PATH": "/usr/bin", "CODEX_HOME": "/work/.codex-home"},
        )
        with pytest.raises(ProviderCapabilityError, match="lets commands write"):
            agent.run("hi", cwd="/work")
        assert executor.requests == []

        agent.run("hi", cwd="/elsewhere")
        assert IGNORE_RULES_FLAG not in executor.requests[-1].argv
