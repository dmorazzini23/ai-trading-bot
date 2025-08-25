import logging
from ai_trading.logging import setup_logging, _LOGGING_LOCK
import ai_trading.logging as logging_module


def test_setup_logging_does_not_grow_handlers():
    """Repeated setup_logging calls should not grow root handlers."""
    root = logging.getLogger()
    original = root.handlers.copy()
    root.handlers.clear()
    try:
        with _LOGGING_LOCK:
            logging_module._LOGGING_CONFIGURED = False
            logging_module._configured = False
            logging_module._listener = None
        setup_logging()
        first = len(root.handlers)
        setup_logging()
        second = len(root.handlers)
        setup_logging()
        third = len(root.handlers)
        assert first == second == third
    finally:
        root.handlers = original
        with _LOGGING_LOCK:
            logging_module._LOGGING_CONFIGURED = False
            logging_module._configured = False
            logging_module._listener = None
