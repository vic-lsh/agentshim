"""A tool policy as a Claude provider option.

What a caller relies on: with a policy, the turn never carries
``--dangerously-skip-permissions``; the CLI is told exactly the built-in tools
and allow rules the policy names (read back by an independent reader of the
flags), under ``--permission-mode dontAsk``; a schema turn can still return its
answer; an entry the CLI would misread is rejected at construction; and
without a policy the argv is what it was.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from agentshim import (
    ArgvContext,
    ClaudeProvider,
    ClaudeToolPolicy,
    CliAgent,
    McpScope,
    OutputSchema,
    StdioMcpServer,
    TurnRequest,
)
from agentshim.testing import FakeExecutor, scripted_turn
from hypothesis import given
from hypothesis import strategies as st

_names = st.from_regex(r"[A-Za-z][A-Za-z0-9_]{0,15}", fullmatch=True)
_rules = _names | st.sampled_from(
    ["mcp__rs", "mcp__rs__list_facts", "Bash(git log:*)", "Read(./src/**)"]
)
policies = st.builds(
    ClaudeToolPolicy,
    builtin=st.none() | st.lists(_names, max_size=4),
    allowed=st.lists(_rules, max_size=4),
)


def _ctx(*, schema: str | None = None) -> ArgvContext:
    return ArgvContext(
        binary_path="/usr/local/bin/claude",
        model="claude-opus-5-5",
        env={},
        resume_session_id=None,
        reasoning_effort=None,
        schema_inline=schema,
        schema_path=None,
        mcp_argv=(),
        extra_args=("--append-system-prompt", "be brief"),
    )


def _read_policy(argv: list[str]) -> tuple[list[str] | None, list[str], str | None]:
    """Recover (builtin, allowed, permission mode) the way Claude Code's parser would.

    ``--allowedTools`` is variadic: it takes every following argument up to the
    next one that starts with ``-``.
    """
    builtin: list[str] | None = None
    if "--tools" in argv:
        value = argv[argv.index("--tools") + 1]
        builtin = value.split(",") if value else []
    allowed: list[str] = []
    if "--allowedTools" in argv:
        for arg in argv[argv.index("--allowedTools") + 1 :]:
            if arg.startswith("-"):
                break
            allowed.append(arg)
    mode = argv[argv.index("--permission-mode") + 1] if "--permission-mode" in argv else None
    return builtin, allowed, mode


class TestArgv:
    @given(policies)
    def test_the_cli_reads_back_exactly_the_policy(self, policy: ClaudeToolPolicy) -> None:
        argv = ClaudeProvider(tools=policy).build_argv(_ctx())
        builtin, allowed, mode = _read_policy(argv)
        assert "--dangerously-skip-permissions" not in argv
        assert mode == "dontAsk"
        assert builtin == (None if policy.builtin is None else list(policy.builtin))
        assert allowed == list(policy.allowed)

    @given(policies)
    def test_a_schema_turn_may_return_its_answer(self, policy: ClaudeToolPolicy) -> None:
        argv = ClaudeProvider(tools=policy).build_argv(_ctx(schema='{"type":"object"}'))
        _, allowed, _ = _read_policy(argv)
        assert allowed.count("StructuredOutput") == 1
        assert [rule for rule in allowed if rule != "StructuredOutput"] == [
            rule for rule in policy.allowed if rule != "StructuredOutput"
        ]

    def test_no_builtin_tools_renders_an_empty_list(self) -> None:
        argv = ClaudeProvider(tools=ClaudeToolPolicy.only(["mcp__rs", "mcp__sregym"])).build_argv(
            _ctx()
        )
        assert argv[1:9] == [
            "-p",
            "--tools",
            "",
            "--allowedTools",
            "mcp__rs",
            "mcp__sregym",
            "--permission-mode",
            "dontAsk",
        ]

    def test_without_a_policy_the_argv_is_unchanged(self) -> None:
        argv = ClaudeProvider().build_argv(_ctx())
        assert argv[1:3] == ["-p", "--dangerously-skip-permissions"]
        assert "--tools" not in argv
        assert "--permission-mode" not in argv


class TestValidation:
    @pytest.mark.parametrize(
        ("builtin", "allowed"),
        [
            ("Read", ()),
            ((), "mcp__rs"),
            (("Read,Grep",), ()),
            (("Read Grep",), ()),
            (("Bash(git:*)",), ()),
            ((), ("",)),
            ((), (" Read",)),
            ((), ("--dangerously-skip-permissions",)),
            ((), ("Re\x00ad",)),
            ((), (3,)),
        ],
    )
    def test_entries_the_cli_would_misread_are_rejected(
        self, builtin: object, allowed: object
    ) -> None:
        with pytest.raises((TypeError, ValueError)):
            ClaudeToolPolicy(builtin=builtin, allowed=allowed)  # pyright: ignore[reportArgumentType]

    def test_a_non_policy_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="ClaudeToolPolicy"):
            ClaudeProvider(tools=["Read"])  # pyright: ignore[reportArgumentType]


class TestTurn:
    def test_an_answer_only_turn_reaches_the_cli_locked_down(self, tmp_path: Path) -> None:
        """A consumer's whole turn: no built-ins, one session MCP server, a schema answer."""
        executor = FakeExecutor(scripted_turn("claude", structured_output={"answer": 42}))
        agent = CliAgent(
            ClaudeProvider(tools=ClaudeToolPolicy.only(["mcp__rs"])),
            model="claude-opus-5-5",
            executor=executor,
            env={"PATH": "/usr/bin"},
        )
        result = agent.run(
            TurnRequest(
                prompt="how many?",
                output_schema=OutputSchema({"type": "object"}, host_dir=tmp_path),
                mcp_servers=[StdioMcpServer(name="rs", command="/bin/rs")],
            ),
            cwd=str(tmp_path),
            mcp_scope=McpScope.SESSION,
        )
        assert result.structured_output == {"answer": 42}
        argv = list(executor.requests[0].argv)
        builtin, allowed, mode = _read_policy(argv)
        assert (builtin, allowed, mode) == ([], ["mcp__rs", "StructuredOutput"], "dontAsk")
        assert "--strict-mcp-config" in argv
        assert "--dangerously-skip-permissions" not in argv
