"""TOML literals in Codex ``--config`` overrides.

Codex parses each override's value as TOML and falls back to the raw text
when that fails, so a malformed literal changes the value instead of raising.
These properties pin the encoder against a real TOML parser.
"""

from __future__ import annotations

import sys

import pytest
from agentshim import StdioMcpServer
from agentshim.providers.codex import CodexProvider, parse_mcp_servers
from agentshim.providers.codex._toml import toml_array, toml_bool, toml_str, unescape_toml
from hypothesis import example, given
from hypothesis import strategies as st

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on the 3.10 CI leg
    import tomli as tomllib  # pyright: ignore[reportMissingImports]

_CONTROL = "".join(chr(code) for code in (*range(0x20), 0x7F))

#: Any text, weighted toward what breaks a hand-rolled encoder.
texts = st.text(
    alphabet=st.one_of(
        st.characters(),
        st.sampled_from([*_CONTROL, "\\", '"', "'", "\u2028", "\ufeff", "\U0001f600"]),
    )
)


def _parse_value(literal: str) -> object:
    return tomllib.loads(f"k = {literal}")["k"]


class TestTomlStr:
    @given(texts)
    @example("line\nbreak")
    @example("tab\tand\x00nul")
    @example('C:\\path\\"quoted"')
    @example("\x7f")
    def test_a_toml_parser_reads_back_exactly_the_input(self, value: str) -> None:
        assert _parse_value(toml_str(value)) == value

    @given(texts)
    def test_unescape_inverts_the_encoder(self, value: str) -> None:
        assert unescape_toml(toml_str(value)[1:-1]) == value

    @given(texts)
    def test_the_literal_holds_no_raw_control_character(self, value: str) -> None:
        """TOML forbids them, and Codex would then keep the raw text."""
        assert not set(toml_str(value)) & set(_CONTROL)

    @given(texts)
    def test_the_literal_is_one_quoted_string(self, value: str) -> None:
        literal = toml_str(value)
        assert literal.startswith('"')
        assert literal.endswith('"')
        assert '"' not in literal[1:-1].replace('\\"', "").replace("\\\\", "")


class TestTomlArrayAndBool:
    @given(st.lists(texts, max_size=8))
    def test_an_array_parses_back_to_the_same_strings(self, values: list[str]) -> None:
        assert _parse_value(toml_array(values)) == values

    @given(value=st.booleans())
    def test_a_bool_parses_back(self, *, value: bool) -> None:
        assert _parse_value(toml_bool(value)) is value


class TestUnescapeRejectsWhatTheEncoderNeverWrites:
    @pytest.mark.parametrize("body", ["\\x41", "\\u12", "\\U0001F6", "\\q", "\\"])
    def test_an_invalid_or_truncated_escape_raises(self, body: str) -> None:
        with pytest.raises(ValueError, match="escape"):
            unescape_toml(body)


class TestMcpOverridesSurviveAnyText:
    """The MCP renderer shares the encoder, so its round trip is a property too."""

    @given(
        command=texts.filter(bool),
        args=st.lists(texts, max_size=5),
        env=st.dictionaries(st.from_regex(r"[A-Z][A-Z0-9_]{0,10}", fullmatch=True), texts),
    )
    @example(command="srv", args=["--flag\nvalue"], env={"TOKEN": "a\tb"})
    def test_parse_mcp_servers_inverts_install_mcp(
        self, command: str, args: list[str], env: dict[str, str]
    ) -> None:
        server = StdioMcpServer(name="srv", command=command, args=args, env=env)
        flags = CodexProvider().install_mcp(None, [server]).argv
        assert parse_mcp_servers(flags) == {"srv": {"command": command, "args": args, "env": env}}

    @given(texts.filter(bool))
    def test_every_rendered_value_is_valid_toml(self, value: str) -> None:
        server = StdioMcpServer(name="srv", command=value, args=[value], env={"V": value})
        flags = CodexProvider().install_mcp(None, [server]).argv
        for index in range(1, len(flags), 2):
            key, _, literal = flags[index].partition("=")
            assert tomllib.loads(f"{key} = {literal}"), flags[index]
