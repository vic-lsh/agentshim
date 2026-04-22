from .base import register_provider
from .cli_agent import CLICodingAgent
from .mcp_config import HttpMcpServer, McpServerConfig
from .sandbox import SandboxConfig


@register_provider("openai", "codex")
class CodexCodingAgent(CLICodingAgent):
    """Coding agent implementation using the Codex CLI tool."""

    def __init__(
        self,
        model: str | None = None,
        mcp_servers: list[McpServerConfig] | None = None,
        sandbox: bool | SandboxConfig = False,
    ):
        """Initialize the Codex coding agent.

        Args:
            model: Optional model name to use with codex. If None, uses default.
            mcp_servers: Optional list of MCP server configurations.
            sandbox: Not supported for Codex; must be False.
        """
        if sandbox:
            raise NotImplementedError("sandbox is not supported for CodexCodingAgent")
        super().__init__("codex", model, mcp_servers=mcp_servers)

    @property
    def codex_path(self) -> str:
        """Return path to codex binary (for backward compatibility)."""
        return self.binary_path

    @property
    def _log_prefix(self) -> str:
        """Return the log prefix for this agent."""
        return "[Codex]"

    def _build_mcp_args(self) -> list[str]:
        """Build -c flag arguments for MCP server configuration."""
        args: list[str] = []
        for s in self.mcp_servers:
            prefix = f"mcp_servers.{s.name}"
            if isinstance(s, HttpMcpServer):
                args.extend(["-c", f'{prefix}.url="{s.url}"'])
            else:
                args.extend(["-c", f'{prefix}.command="{s.command}"'])
                if s.args:
                    toml_arr = "[" + ", ".join(f'"{a}"' for a in s.args) + "]"
                    args.extend(["-c", f"{prefix}.args={toml_arr}"])
                for k, v in s.env.items():
                    args.extend(["-c", f'{prefix}.env.{k}="{v}"'])
        return args

    def _get_command(self, prompt: str) -> list[str]:
        cmd = [self.binary_path, "exec", "--dangerously-bypass-approvals-and-sandbox"]
        if self.model:
            cmd.extend(["--model", self.model])
        if self.mcp_servers:
            cmd.extend(self._build_mcp_args())
        return cmd
