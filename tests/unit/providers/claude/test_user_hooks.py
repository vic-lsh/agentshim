"""Caller hooks as a Claude provider option.

What a caller relies on: a hook either reaches Claude Code's settings intact
(its argv survives the shell Claude runs it through, its event, matcher and
timeout are preserved, in order) or is rejected at construction; adding
hooks never changes the sandbox or agentshim's own confinement hook; and
without hooks the argv is what it was.
"""

from __future__ import annotations

import json
import math
import shlex
import shutil
import subprocess
import sys
from typing import Any

import pytest
from agentshim import ArgvContext, ClaudeHook, ClaudeProvider, SandboxConfig
from agentshim.providers.claude import build_settings
from agentshim.providers.claude.sandbox import CONFINE_READS_HOOK
from hypothesis import example, given, settings
from hypothesis import strategies as st

# -- strategies -------------------------------------------------------------

#: Arguments with everything a shell would otherwise interpret.
_args = st.text(
    alphabet=st.one_of(
        st.characters(exclude_characters="\x00"),
        st.sampled_from(list("$`\"'\\;&|<>(){}*?~!# \t\n")),
    ),
    max_size=16,
) | st.sampled_from(["$(id)", "${HOME}", "-", "--", "-c"])
events = st.sampled_from(["PreToolUse", "PostToolUse", "UserPromptSubmit", "Stop"]) | st.from_regex(
    r"[A-Z][A-Za-z]{0,15}", fullmatch=True
)
hooks = st.builds(
    ClaudeHook,
    event=events,
    command=st.lists(_args, min_size=1, max_size=5).map(lambda a: ["/usr/bin/hook", *a]),
    matcher=st.none() | st.sampled_from(["Bash", "Edit|Write", "mcp__.*"]),
    timeout_s=st.none() | st.floats(min_value=0.001, max_value=3600, allow_nan=False),
)
hook_lists = st.lists(hooks, max_size=6)
sandboxes = st.none() | st.builds(
    SandboxConfig,
    allowed_domains=st.lists(st.sampled_from(["github.com", "*.npmjs.org"]), max_size=2),
    confine_native_reads_to=st.lists(st.sampled_from(["/work", "/data"]), max_size=2),
    excluded_commands=st.lists(st.sampled_from(["docker *", "sdo detector check"]), max_size=2),
)
contexts = st.builds(
    ArgvContext,
    binary_path=st.just("/usr/local/bin/claude"),
    model=st.none() | st.just("haiku"),
    env=st.just({}),
    resume_session_id=st.none() | st.uuids().map(str),
    reasoning_effort=st.none() | st.just("low"),
    schema_inline=st.none() | st.just('{"type":"object"}'),
    schema_path=st.none(),
    mcp_argv=st.just(()),
    extra_args=st.lists(st.sampled_from(["--verbose", "--max-turns", "3"]), max_size=3).map(tuple),
)


def _settings(argv: list[str]) -> dict[str, object] | None:
    assert argv.count("--settings") <= 1
    if "--settings" not in argv:
        return None
    return json.loads(argv[argv.index("--settings") + 1])


def _without_settings(argv: list[str]) -> list[str]:
    if "--settings" not in argv:
        return argv
    index = argv.index("--settings")
    return argv[:index] + argv[index + 2 :]


def _user_entries(
    settings: dict[str, Any], sandbox: SandboxConfig | None
) -> list[tuple[str, dict[str, Any]]]:
    """Hook entries in settings order, minus agentshim's own confinement hook."""
    block = settings.get("hooks", {})
    assert isinstance(block, dict)
    entries: list[tuple[str, dict[str, Any]]] = []
    for event, event_entries in block.items():
        for entry in event_entries:
            if entry["hooks"][0]["command"].startswith(shlex.quote(sys.executable)) and (
                CONFINE_READS_HOOK in entry["hooks"][0]["command"]
            ):
                assert sandbox is not None
                continue
            entries.append((event, entry))
    return entries


# -- construction -----------------------------------------------------------


class TestHookValidation:
    def test_a_minimal_hook(self) -> None:
        hook = ClaudeHook(event="PreToolUse", command=["/bin/true"])
        assert hook.command == ("/bin/true",)
        assert hook.matcher is None
        assert hook.timeout_s is None

    @given(st.text().filter(lambda e: not (e[:1].isascii() and e[:1].isupper() and e.isalpha())))
    @example("preToolUse")
    @example("Pre-Tool")
    @example("")
    def test_an_event_that_is_not_pascal_case_is_rejected(self, event: str) -> None:
        with pytest.raises(ValueError, match="PascalCase"):
            ClaudeHook(event=event, command=["/bin/true"])

    def test_a_bare_string_command_is_not_split_or_iterated(self) -> None:
        """``"python3 hook.py"`` as one argv element would name a missing file."""
        with pytest.raises(TypeError, match="argv sequence"):
            ClaudeHook(event="PreToolUse", command="python3 hook.py")

    def test_an_empty_command_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            ClaudeHook(event="PreToolUse", command=[])

    @pytest.mark.parametrize("arg", [1, None, b"/bin/true"])
    def test_a_non_string_argument_is_rejected(self, arg: object) -> None:
        with pytest.raises(TypeError, match="must be str"):
            ClaudeHook(event="PreToolUse", command=["/bin/hook", arg])  # type: ignore[list-item]

    @given(_args, st.integers(min_value=0, max_value=8))
    def test_a_nul_byte_anywhere_is_rejected(self, arg: str, at: int) -> None:
        """A NUL cannot reach a process through argv, so the hook could never run as written."""
        with pytest.raises(ValueError, match="NUL"):
            ClaudeHook(event="PreToolUse", command=["/bin/hook", arg[:at] + "\x00" + arg[at:]])

    @pytest.mark.parametrize("matcher", ["", 3])
    def test_an_empty_or_non_string_matcher_is_rejected(self, matcher: object) -> None:
        with pytest.raises(ValueError, match="matcher"):
            ClaudeHook(event="PreToolUse", command=["/bin/true"], matcher=matcher)  # type: ignore[arg-type]

    @pytest.mark.parametrize("timeout", [0, -1, math.nan, math.inf, True, "5"])
    def test_a_timeout_that_is_not_positive_and_finite_is_rejected(self, timeout: object) -> None:
        with pytest.raises(ValueError, match="timeout_s"):
            ClaudeHook(event="PreToolUse", command=["/bin/true"], timeout_s=timeout)  # type: ignore[arg-type]

    @given(hooks)
    def test_a_hook_is_frozen_hashable_and_equal_across_sequence_types(
        self, hook: ClaudeHook
    ) -> None:
        again = ClaudeHook(
            event=hook.event,
            command=list(hook.command),
            matcher=hook.matcher,
            timeout_s=hook.timeout_s,
        )
        assert again == hook
        assert hash(again) == hash(hook)
        with pytest.raises(AttributeError):
            hook.event = "Stop"  # type: ignore[misc]


class TestProviderOption:
    def test_no_hooks_by_default(self) -> None:
        assert ClaudeProvider().hooks == ()

    @given(hook_lists)
    def test_hooks_are_stored_in_order_as_a_tuple(self, given_hooks: list[ClaudeHook]) -> None:
        assert ClaudeProvider(hooks=given_hooks).hooks == tuple(given_hooks)

    def test_a_single_hook_must_be_wrapped_in_a_sequence(self) -> None:
        hook = ClaudeHook(event="PreToolUse", command=["/bin/true"])
        with pytest.raises(TypeError, match="sequence of ClaudeHook"):
            ClaudeProvider(hooks=hook)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bad", [[{"event": "PreToolUse", "command": ["/bin/true"]}], ["/bin/true"], None]
    )
    def test_anything_but_claude_hooks_is_rejected(self, bad: object) -> None:
        with pytest.raises(TypeError, match="ClaudeHook"):
            ClaudeProvider(hooks=bad)  # type: ignore[arg-type]


# -- rendering properties ---------------------------------------------------


class TestRenderedSettings:
    @given(sandboxes, hook_lists)
    def test_every_hook_appears_once_in_order_with_its_fields(
        self, sandbox: SandboxConfig | None, given_hooks: list[ClaudeHook]
    ) -> None:
        rendered = build_settings(sandbox, hooks=given_hooks)
        events_in_order = list(rendered.get("hooks", {}))
        # Grouped by event in first-appearance order; the caller's order within each.
        expected = [h for event in events_in_order for h in given_hooks if h.event == event]
        entries = _user_entries(rendered, sandbox)
        assert len(entries) == len(given_hooks)
        for (event, entry), hook in zip(entries, expected, strict=True):
            assert event == hook.event
            assert entry.get("matcher") == hook.matcher
            (handler,) = entry["hooks"]
            assert handler["type"] == "command"
            assert shlex.split(handler["command"]) == list(hook.command)
            assert handler.get("timeout") == hook.timeout_s

    @given(sandboxes, hook_lists)
    def test_hooks_never_change_the_sandbox_block(
        self, sandbox: SandboxConfig | None, given_hooks: list[ClaudeHook]
    ) -> None:
        with_hooks = build_settings(sandbox, hooks=given_hooks)
        without = build_settings(sandbox)
        assert with_hooks.get("sandbox") == without.get("sandbox")

    @given(st.lists(st.sampled_from(["/work", "/data"]), min_size=1, max_size=2), hook_lists)
    def test_the_confinement_hook_stays_first_on_its_event(
        self, roots: list[str], given_hooks: list[ClaudeHook]
    ) -> None:
        sandbox = SandboxConfig(confine_native_reads_to=roots)
        own = build_settings(sandbox)["hooks"]["PreToolUse"][0]
        assert build_settings(sandbox, hooks=given_hooks)["hooks"]["PreToolUse"][0] == own

    @given(sandboxes, hook_lists)
    def test_the_settings_survive_a_json_round_trip(
        self, sandbox: SandboxConfig | None, given_hooks: list[ClaudeHook]
    ) -> None:
        rendered = build_settings(sandbox, hooks=given_hooks)
        assert json.loads(json.dumps(rendered)) == rendered

    def test_nothing_to_set_is_an_empty_object(self) -> None:
        assert build_settings(None) == {}


class TestRenderedArgv:
    @given(sandboxes, hook_lists, contexts)
    def test_one_settings_flag_exactly_when_there_is_something_to_set(
        self, sandbox: SandboxConfig | None, given_hooks: list[ClaudeHook], ctx: ArgvContext
    ) -> None:
        argv = ClaudeProvider(sandbox=sandbox, hooks=given_hooks).build_argv(ctx)
        rendered = _settings(argv)
        if sandbox is None and not given_hooks:
            assert rendered is None
        else:
            assert rendered == build_settings(sandbox, hooks=given_hooks)

    @given(sandboxes, hook_lists, contexts)
    def test_hooks_do_not_disturb_the_rest_of_the_argv(
        self, sandbox: SandboxConfig | None, given_hooks: list[ClaudeHook], ctx: ArgvContext
    ) -> None:
        hooked = ClaudeProvider(sandbox=sandbox, hooks=given_hooks).build_argv(ctx)
        plain = ClaudeProvider(sandbox=sandbox).build_argv(ctx)
        assert _without_settings(hooked) == _without_settings(plain)

    @given(sandboxes, hook_lists, contexts)
    def test_caller_extras_stay_last(
        self, sandbox: SandboxConfig | None, given_hooks: list[ClaudeHook], ctx: ArgvContext
    ) -> None:
        argv = ClaudeProvider(sandbox=sandbox, hooks=given_hooks).build_argv(ctx)
        if ctx.extra_args:
            assert argv[-len(ctx.extra_args) :] == list(ctx.extra_args)

    @given(contexts)
    def test_without_sandbox_or_hooks_the_argv_is_unchanged(self, ctx: ArgvContext) -> None:
        assert "--settings" not in ClaudeProvider(hooks=[]).build_argv(ctx)


_ECHO_ARGV = "import json, sys; print(json.dumps(sys.argv[1:]))"


@pytest.mark.parametrize("shell", [s for s in ("sh", "bash") if shutil.which(s)])
class TestTheShellRunsExactlyTheArgv:
    """Claude Code runs a hook command through a shell; nothing may be interpreted."""

    @settings(max_examples=60)
    @given(st.lists(_args, max_size=5))
    @example(["$(touch /tmp/pwned)", "`id`", "a;b", "*", "~", "'", '"', "\\", "\n"])
    def test_every_argument_arrives_literally(self, shell: str, args: list[str]) -> None:
        hook = ClaudeHook(event="PreToolUse", command=[sys.executable, "-c", _ECHO_ARGV, *args])
        command = build_settings(None, hooks=[hook])["hooks"]["PreToolUse"][0]["hooks"][0][
            "command"
        ]
        completed = subprocess.run(  # noqa: S603 - the shell is what Claude Code uses
            [shutil.which(shell) or shell, "-c", command],
            capture_output=True,
            text=True,
            check=True,
        )
        assert json.loads(completed.stdout) == args
