import pytest
from pydantic import ValidationError

from agentshim.mcp_config import HttpMcpServer, McpServerConfig, StdioMcpServer


class TestHttpMcpServer:
    def test_valid_http_server(self):
        s = HttpMcpServer(name="my-server", url="http://localhost:8080/sse")
        assert s.name == "my-server"
        assert s.url == "http://localhost:8080/sse"
        assert s.headers == {}

    def test_valid_http_server_with_headers(self):
        s = HttpMcpServer(
            name="auth-server",
            url="https://example.com/mcp",
            headers={"Authorization": "Bearer tok"},
        )
        assert s.headers == {"Authorization": "Bearer tok"}

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError, match="name"):
            HttpMcpServer(name="", url="http://localhost:8080")

    def test_empty_url_raises(self):
        with pytest.raises(ValidationError, match="url"):
            HttpMcpServer(name="bad", url="")

    def test_frozen(self):
        s = HttpMcpServer(name="x", url="http://localhost")
        with pytest.raises(ValidationError):
            s.name = "y"  # type: ignore[reportAttributeAccessIssue]


class TestStdioMcpServer:
    def test_valid_stdio_server(self):
        s = StdioMcpServer(name="tool", command="npx")
        assert s.name == "tool"
        assert s.command == "npx"
        assert s.args == []
        assert s.env == {}

    def test_valid_stdio_server_with_args_and_env(self):
        s = StdioMcpServer(
            name="tool",
            command="npx",
            args=["-y", "@some/pkg"],
            env={"API_KEY": "secret"},
        )
        assert s.args == ["-y", "@some/pkg"]
        assert s.env == {"API_KEY": "secret"}

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError, match="name"):
            StdioMcpServer(name="", command="npx")

    def test_empty_command_raises(self):
        with pytest.raises(ValidationError, match="command"):
            StdioMcpServer(name="bad", command="")

    def test_frozen(self):
        s = StdioMcpServer(name="x", command="npx")
        with pytest.raises(ValidationError):
            s.command = "other"  # type: ignore[reportAttributeAccessIssue]


class TestMcpServerConfigUnion:
    def test_isinstance_http(self):
        s: McpServerConfig = HttpMcpServer(name="h", url="http://x")
        assert isinstance(s, HttpMcpServer)
        assert not isinstance(s, StdioMcpServer)

    def test_isinstance_stdio(self):
        s: McpServerConfig = StdioMcpServer(name="s", command="cmd")
        assert isinstance(s, StdioMcpServer)
        assert not isinstance(s, HttpMcpServer)
