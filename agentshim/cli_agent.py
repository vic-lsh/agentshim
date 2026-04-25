import io
import os
import shutil
import signal
import subprocess
import sys
import threading
from abc import abstractmethod
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from loguru import logger

from .base import BaseAgentSession, BaseCodingAgent
from .events import AgentEventHandler, compose_event_handlers, default_event_handler
from .mcp_config import McpServerConfig
from .usage import ProviderUsage
from .utils import get_interactive_env


class CLIGenerationSession:
    """Handles a single generation request lifecycle."""

    def __init__(
        self,
        binary_name: str,
        env: dict[str, str],
        log_prefix: str,
        cmd: list[str],
        logger: Any,
        cwd: str | None = None,
        timeout: int = 300,
        silent: bool = False,
        event_handler: AgentEventHandler | None = None,
        on_process_started: Callable[[subprocess.Popen[str]], None] | None = None,
    ):
        self.binary_name = binary_name
        self.env = env
        self.log_prefix = log_prefix
        self.cmd = cmd
        self.logger = logger
        self.cwd = cwd
        self.timeout = timeout
        self.silent = silent
        self.event_handler = default_event_handler(
            event_handler=event_handler,
            silent=silent,
            logger=logger,
            log_prefix=log_prefix,
        )
        self.on_process_started = on_process_started

        # State initialization
        self.stdout_lines: list[str] = []
        self.stderr_lines: list[str] = []
        # Providers populate this during event handling; stays at the
        # empty default if the session crashes before any terminal event.
        self.usage: ProviderUsage = ProviderUsage()
        # Final provider-reported accounting snapshot, when the CLI exposes
        # one on a terminal event.
        self.final_usage: dict[str, Any] | None = None
        self.total_cost_usd: float | None = None
        self.duration_ms: int | None = None
        # Provider session id captured from the event stream (set by subclasses
        # that parse JSON events). ``None`` if the underlying CLI did not emit
        # an id during this run.
        self.session_id: str | None = None

    def _process_stdout(self, line: str) -> None:
        """Process a line from stdout."""
        line_stripped = line.rstrip("\n")

        if self.event_handler and line_stripped:
            self.event_handler.on_thinking(line_stripped + "\n")

        self.stdout_lines.append(line)

    def _process_stderr(self, line: str) -> None:
        """Process a line from stderr."""
        line_stripped = line.rstrip("\n")
        on_stderr = getattr(self.event_handler, "on_stderr", None)
        if on_stderr is not None and line_stripped:
            on_stderr(line)
        self.stderr_lines.append(line)

    def run(self, prompt: str) -> str:
        """Execute the generation process."""
        on_run_start = getattr(self.event_handler, "on_run_start", None)
        if on_run_start is not None:
            on_run_start(self.cmd)
            sys.stdout.flush()

        def read_stdout(pipe: io.TextIOWrapper) -> None:
            for line in iter(pipe.readline, ""):
                if not line:
                    break
                self._process_stdout(line)
            pipe.close()

        def read_stderr(pipe: io.TextIOWrapper) -> None:
            for line in iter(pipe.readline, ""):
                if not line:
                    break
                self._process_stderr(line)
            pipe.close()

        process = subprocess.Popen(
            self.cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=self.cwd,
            env=self.env,
            start_new_session=True,
        )

        if self.on_process_started is not None:
            try:
                self.on_process_started(process)
            except Exception as exc:
                self.logger.warning(f"on_process_started callback raised: {exc}")

        stdout_thread = threading.Thread(target=read_stdout, args=(process.stdout,))
        stderr_thread = threading.Thread(target=read_stderr, args=(process.stderr,))

        stdout_thread.daemon = True
        stderr_thread.daemon = True

        stdout_thread.start()
        stderr_thread.start()

        try:
            if process.stdin:
                process.stdin.write(prompt)
                process.stdin.close()
        except BrokenPipeError:
            pass

        try:
            process.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise subprocess.TimeoutExpired(self.cmd, self.timeout) from None
        finally:
            if process.poll() is None:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    process.wait()
                except (ProcessLookupError, OSError):
                    pass

        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)

        stdout_data = "".join(self.stdout_lines)
        stderr_data = "".join(self.stderr_lines)

        on_run_end = getattr(self.event_handler, "on_run_end", None)
        if on_run_end is not None:
            on_run_end(process.returncode)

        if process.returncode != 0:
            raise RuntimeError(f"{self.binary_name} exited with code {process.returncode}: {stderr_data}")

        return stdout_data.strip()


class CLICodingAgent(BaseCodingAgent):
    """Base class for CLI-based coding agents."""

    CLI_CHECK_TIMEOUT_SECONDS = 15

    def __init__(
        self,
        binary_name: str,
        model: str | None = None,
        event_handler: AgentEventHandler | None = None,
        event_handlers: Iterable[AgentEventHandler] | None = None,
        mcp_servers: Sequence[McpServerConfig] | None = None,
    ):
        """Initialize the CLI coding agent.

        Args:
            binary_name: The name of the executable to use.
            model: Optional model name to use.
            event_handler: Optional event handler for UI updates.
            event_handlers: Optional event handlers to compose in order.
            mcp_servers: Optional list of MCP server configurations.

        Raises:
            RuntimeError: If binary is not found in PATH or is not working.
        """
        self.env = get_interactive_env()
        self.binary_name = binary_name
        self.model = model
        self.event_handler = compose_event_handlers(event_handler, event_handlers)
        self.mcp_servers: list[McpServerConfig] = list(mcp_servers or [])

        # Search for binary in the captured environment's PATH
        binary_path = shutil.which(binary_name, path=self.env.get("PATH"))

        if not binary_path:
            # Fallback to current PATH if not found in interactive env
            binary_path = shutil.which(binary_name)

        if not binary_path:
            raise RuntimeError(
                f"{binary_name} binary not found in PATH. Please ensure {binary_name} is installed and available."
            )
        self.binary_path = binary_path
        self._check_cli()
        self.logger = logger.bind(agent_prefix=self._log_prefix)
        # Populated after each generate() call from the session's usage.
        self.last_usage: ProviderUsage = ProviderUsage()

    def _check_cli(self):
        """Check if the CLI tool is available and executable."""
        try:
            result = subprocess.run(
                [self.binary_path, "--help"],
                capture_output=True,
                text=True,
                check=False,
                env=self.env,
                stdin=subprocess.DEVNULL,
                timeout=self.CLI_CHECK_TIMEOUT_SECONDS,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"{self.binary_name} CLI tool at '{self.binary_path}' is not working correctly. "
                    f"'{self.binary_path} --help' exited with code {result.returncode}. "
                    f"Stderr: {result.stderr}"
                )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"{self.binary_name} CLI tool at '{self.binary_path}' did not respond to "
                f"'--help' within {self.CLI_CHECK_TIMEOUT_SECONDS}s."
            ) from e
        except FileNotFoundError as e:
            raise RuntimeError(
                f"{self.binary_name} CLI tool not found at '{self.binary_path}'. "
                f"Please ensure {self.binary_name} is installed and in your PATH."
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to check {self.binary_name} CLI tool: {e}") from e

    @abstractmethod
    def _get_command(self, prompt: str, resume_session_id: str | None = None) -> list[str]:
        """Construct the command line arguments.

        Args:
            prompt: The prompt to send to the agent.
            resume_session_id: If set, the provider session id to resume.
        """

    @property
    def _log_prefix(self) -> str:
        """Return the log prefix for this agent."""
        return f"[{self.__class__.__name__}]"

    def _create_session(
        self,
        cmd: list[str],
        cwd: str | None = None,
        timeout: int = 300,
        silent: bool = False,
        on_process_started: Callable[[subprocess.Popen[str]], None] | None = None,
    ) -> CLIGenerationSession:
        """Create a session for a single generation request.

        Can be overridden by subclasses to return specialized sessions.
        """
        return CLIGenerationSession(
            binary_name=self.binary_name,
            env=self.env,
            log_prefix=self._log_prefix,
            cmd=cmd,
            logger=self.logger,
            cwd=cwd,
            timeout=timeout,
            silent=silent,
            event_handler=self.event_handler,
            on_process_started=on_process_started,
        )

    def start_session(
        self,
        cwd: str | None = None,
        timeout: int = 300,
        silent: bool = False,
    ) -> "CLIAgentSession":
        """Open a stateful conversation with the underlying CLI.

        Returns a :class:`CLIAgentSession` whose ``generate(prompt)`` may be
        called repeatedly; each call after the first automatically resumes
        the prior conversation via the provider's native resume flag.

        Args:
            cwd: Default working directory for ``session.generate`` calls.
            timeout: Default timeout in seconds.
            silent: If True, suppress stdout printing of the agent's output.
        """
        return CLIAgentSession(self, cwd=cwd, timeout=timeout, silent=silent)

    def generate(
        self,
        prompt: str,
        cwd: str | None = None,
        timeout: int = 300,
        silent: bool = False,
        on_process_started: Callable[[subprocess.Popen[str]], None] | None = None,
    ) -> str:
        """One-shot prompt → reply. Convenience wrapper around
        ``start_session(...).generate(prompt)``; no conversation state is
        retained across calls. Use :meth:`start_session` for multi-turn flows.
        """
        return self.start_session(cwd=cwd, timeout=timeout, silent=silent).generate(
            prompt, on_process_started=on_process_started
        )


class CLIAgentSession(BaseAgentSession):
    """Stateful, resumable conversation with a CLI agent.

    Holds the provider session id captured from the first ``generate`` call
    so subsequent calls automatically pass the right native resume flag
    (``claude --resume``, ``codex exec resume``, ``gemini --resume``,
    ``opencode run --session``). Callers do not see provider-specific
    plumbing.

    A session is single-threaded — concurrent calls into ``generate`` on the
    same instance are not supported.
    """

    def __init__(
        self,
        agent: CLICodingAgent,
        *,
        cwd: str | None = None,
        timeout: int = 300,
        silent: bool = False,
    ):
        self.agent = agent
        self._cwd = cwd
        self._timeout = timeout
        self._silent = silent
        # Provider session id, set after the first ``generate`` call (None
        # if the underlying CLI did not emit one).
        self.session_id: str | None = None

    def generate(
        self,
        prompt: str,
        cwd: str | None = None,
        timeout: int | None = None,
        silent: bool | None = None,
        on_process_started: Callable[[subprocess.Popen[str]], None] | None = None,
    ) -> str:
        """Send ``prompt``, returning the assistant's text reply.

        Per-call ``cwd`` / ``timeout`` / ``silent`` override the defaults
        captured by :meth:`CLICodingAgent.start_session`. The provider's
        native resume flag is added automatically on every call after the
        first.

        Args:
            prompt: The prompt to send.
            cwd: Override the session's default working directory.
            timeout: Override the session's default timeout (seconds).
            silent: Override the session's default silent flag.
            on_process_started: Optional callback invoked with the spawned
                ``subprocess.Popen`` immediately after the CLI subprocess
                starts (used by callers that need to kill it externally,
                e.g. crucible's short-circuit).
        """
        effective_cwd = cwd if cwd is not None else self._cwd
        effective_timeout = timeout if timeout is not None else self._timeout
        effective_silent = silent if silent is not None else self._silent

        cmd = self.agent._get_command(prompt, resume_session_id=self.session_id)  # pyright: ignore[reportPrivateUsage]
        run_session = self.agent._create_session(  # pyright: ignore[reportPrivateUsage]
            cmd,
            effective_cwd,
            effective_timeout,
            effective_silent,
            on_process_started=on_process_started,
        )
        result = run_session.run(prompt)
        self.agent.last_usage = getattr(run_session, "usage", ProviderUsage())

        # Capture the id on first run; refresh on later runs only if the
        # underlying CLI actually emitted one (defensive — providers always
        # echo the same id back when resumed).
        captured = getattr(run_session, "session_id", None)
        if captured:
            self.session_id = captured

        return result
