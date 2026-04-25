from unittest.mock import MagicMock

from agentshim.cli_agent import CLIGenerationSession
from agentshim.events import CompositeEventHandler, ConsoleEventHandler, NullEventHandler


def _logger() -> MagicMock:
    logger = MagicMock()
    logger.opt.return_value = MagicMock()
    logger.bind.return_value = MagicMock()
    return logger


def _session(*, event_handler=None, silent=False, logger=None) -> CLIGenerationSession:
    return CLIGenerationSession(
        binary_name="stub",
        env={},
        log_prefix="[Stub]",
        cmd=["stub"],
        logger=logger or _logger(),
        silent=silent,
        event_handler=event_handler,
    )


def test_default_console_handler_is_used_when_no_handler_and_not_silent():
    logger = _logger()
    session = _session(logger=logger)

    session._process_stdout("hello\n")

    logger.opt.return_value.info.assert_any_call("[Stub] ")
    logger.opt.return_value.info.assert_any_call("hello")


def test_no_default_console_handler_is_added_when_user_handler_is_provided():
    logger = _logger()
    handler = MagicMock()
    session = _session(event_handler=handler, logger=logger)

    session._process_stdout("hello\n")

    handler.on_thinking.assert_called_once_with("hello\n")
    logger.opt.return_value.info.assert_not_called()
    logger.info.assert_not_called()


def test_user_can_explicitly_compose_console_handler_with_custom_handler():
    logger = _logger()
    custom = MagicMock()
    handler = CompositeEventHandler(
        [
            ConsoleEventHandler(logger=logger, log_prefix="[Stub]"),
            custom,
        ]
    )
    session = _session(event_handler=handler, logger=_logger())

    session._process_stdout("hello\n")

    logger.opt.return_value.info.assert_any_call("[Stub] ")
    logger.opt.return_value.info.assert_any_call("hello")
    custom.on_thinking.assert_called_once_with("hello\n")


def test_composed_console_handler_uses_session_context_by_default():
    logger = _logger()
    handler = CompositeEventHandler([ConsoleEventHandler()])
    session = _session(event_handler=handler, logger=logger)

    session._process_stdout("hello\n")

    logger.opt.return_value.info.assert_any_call("[Stub] ")
    logger.opt.return_value.info.assert_any_call("hello")


def test_null_event_handler_ignores_events():
    handler = NullEventHandler()

    handler.on_thinking("hello")
    handler.on_tool_call("read", {"path": "x"})
    handler.on_tool_result("read", stdout="ok")
    handler.on_usage({"input_tokens": 1})
