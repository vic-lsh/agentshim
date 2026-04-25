from collections.abc import Iterable, Sequence
from typing import Any, Protocol

from loguru import logger as default_logger

from .utils import truncate_content, truncate_tool_params


class AgentEventHandler(Protocol):
    """Protocol for handling agent events."""

    def on_thinking(self, text: str) -> None:
        """Handle agent thinking output."""
        ...

    def on_tool_call(self, tool: str, args: dict[str, Any] | str | None = None) -> None:
        """Handle tool execution start."""
        ...

    def on_tool_result(
        self,
        tool: str,
        stdout: str = "",
        stderr: str = "",
        exit_code: int | None = None,
        duration: float | None = None,
    ) -> None:
        """Handle tool execution result."""
        ...

    def on_usage(self, usage: dict[str, Any]) -> None:
        """Handle a per-turn usage update from the underlying CLI."""
        ...


class NullEventHandler:
    """Event handler that intentionally ignores every event."""

    def on_thinking(self, text: str) -> None:
        pass

    def on_tool_call(self, tool: str, args: dict[str, Any] | str | None = None) -> None:
        pass

    def on_tool_result(
        self,
        tool: str,
        stdout: str = "",
        stderr: str = "",
        exit_code: int | None = None,
        duration: float | None = None,
    ) -> None:
        pass

    def on_usage(self, usage: dict[str, Any]) -> None:
        pass

    def on_run_start(self, command: Sequence[str]) -> None:
        pass

    def on_run_end(self, exit_code: int | None = None) -> None:
        pass

    def on_stderr(self, text: str) -> None:
        pass


class CompositeEventHandler:
    """Dispatch every event to a sequence of handlers in order."""

    def __init__(self, handlers: Iterable[Any]):
        self.handlers = list(handlers)

    def on_thinking(self, text: str) -> None:
        for handler in self.handlers:
            handler.on_thinking(text)

    def on_tool_call(self, tool: str, args: dict[str, Any] | str | None = None) -> None:
        for handler in self.handlers:
            handler.on_tool_call(tool, args)

    def on_tool_result(
        self,
        tool: str,
        stdout: str = "",
        stderr: str = "",
        exit_code: int | None = None,
        duration: float | None = None,
    ) -> None:
        for handler in self.handlers:
            handler.on_tool_result(
                tool=tool,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                duration=duration,
            )

    def on_usage(self, usage: dict[str, Any]) -> None:
        for handler in self.handlers:
            on_usage = getattr(handler, "on_usage", None)
            if on_usage is not None:
                on_usage(usage)

    def on_run_start(self, command: Sequence[str]) -> None:
        for handler in self.handlers:
            on_run_start = getattr(handler, "on_run_start", None)
            if on_run_start is not None:
                on_run_start(command)

    def on_run_end(self, exit_code: int | None = None) -> None:
        for handler in self.handlers:
            on_run_end = getattr(handler, "on_run_end", None)
            if on_run_end is not None:
                on_run_end(exit_code)

    def on_stderr(self, text: str) -> None:
        for handler in self.handlers:
            on_stderr = getattr(handler, "on_stderr", None)
            if on_stderr is not None:
                on_stderr(text)


class ConsoleEventHandler:
    """Default terminal renderer for agent events.

    Console output is intentionally implemented as an event handler so callers
    can replace it or explicitly compose it with their own handlers.
    """

    def __init__(self, logger: Any | None = None, log_prefix: str | None = None):
        self.logger = logger
        self.log_prefix = log_prefix
        self._at_line_start = True

    def bind_context(self, *, logger: Any, log_prefix: str) -> None:
        """Bind session defaults when the caller did not configure them."""

        if self.logger is None:
            self.logger = logger
        if self.log_prefix is None:
            self.log_prefix = log_prefix

    @property
    def _logger(self) -> Any:
        if self.logger is not None:
            return self.logger
        return default_logger.bind(agent_prefix=self._prefix)

    @property
    def _prefix(self) -> str:
        return self.log_prefix or "[Agent]"

    def _log_raw(self, message: str) -> None:
        self._logger.opt(raw=True).info(message)

    def _newline_if_needed(self) -> None:
        if not self._at_line_start:
            self._log_raw("\n")
            self._at_line_start = True

    def _print_stream_content(self, content: str) -> None:
        if not content:
            return

        lines = content.split("\n")

        for i, line in enumerate(lines):
            is_last = i == len(lines) - 1

            if is_last:
                if line:
                    if self._at_line_start:
                        self._log_raw(f"{self._prefix} ")
                        self._at_line_start = False
                    self._log_raw(line)
            else:
                if self._at_line_start:
                    self._log_raw(f"{self._prefix} ")
                self._log_raw(line)
                self._log_raw("\n")
                self._at_line_start = True

    def on_run_start(self, command: Sequence[str]) -> None:
        self._logger.info(f"Running command: {' '.join(command)}")
        self._log_raw("=" * 80 + "\n")

    def on_run_end(self, exit_code: int | None = None) -> None:
        self._newline_if_needed()
        self._log_raw("=" * 80 + "\n")

    def on_stderr(self, text: str) -> None:
        self._logger.bind(stderr=True).info(f"[STDERR] {text.rstrip()}")

    def on_thinking(self, text: str) -> None:
        self._print_stream_content(text)

    def on_tool_call(self, tool: str, args: dict[str, Any] | str | None = None) -> None:
        self._newline_if_needed()
        truncated = truncate_tool_params(tool, args)
        self._log_raw(f"{self._prefix} \033[34m[Tool Use] {tool} {truncated}\033[0m\n")

    def on_tool_result(
        self,
        tool: str,
        stdout: str = "",
        stderr: str = "",
        exit_code: int | None = None,
        duration: float | None = None,
    ) -> None:
        self._newline_if_needed()
        output = stdout or stderr
        if not output:
            self._log_raw(f"{self._prefix} \033[32m{tool} ran successfully\033[0m\n")
            return
        truncated = truncate_content(output)
        self._log_raw(f"{self._prefix} \033[32m[Tool Result] {truncated}\033[0m\n")

    def on_usage(self, usage: dict[str, Any]) -> None:
        pass


def compose_event_handlers(
    event_handler: Any | None = None,
    event_handlers: Iterable[Any] | None = None,
) -> Any | None:
    """Normalize legacy ``event_handler`` and new ``event_handlers`` inputs."""

    if event_handler is not None and event_handlers is not None:
        raise ValueError("Pass either event_handler or event_handlers, not both")
    if event_handlers is None:
        return event_handler
    handlers = list(event_handlers)
    if len(handlers) == 1:
        return handlers[0]
    return CompositeEventHandler(handlers)


def default_event_handler(
    *,
    event_handler: Any | None,
    silent: bool,
    logger: Any,
    log_prefix: str,
) -> Any:
    """Return the effective handler for a generation session."""

    if event_handler is not None:
        bind_event_handler_context(event_handler, logger=logger, log_prefix=log_prefix)
        return event_handler
    if silent:
        return NullEventHandler()
    return ConsoleEventHandler(logger=logger, log_prefix=log_prefix)


def bind_event_handler_context(handler: Any, *, logger: Any, log_prefix: str) -> None:
    """Attach session context to handlers that opt into it."""

    bind_context = getattr(handler, "bind_context", None)
    if bind_context is not None:
        bind_context(logger=logger, log_prefix=log_prefix)
    for child in getattr(handler, "handlers", []):
        bind_event_handler_context(child, logger=logger, log_prefix=log_prefix)
