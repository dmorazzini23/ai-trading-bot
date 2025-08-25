import logging

import ai_trading.logging as L


def test_setup_logging_idempotent():
    """Multiple setup calls should not grow root handler count."""
    root = logging.getLogger()
    original_handlers = root.handlers.copy()
    try:
        root.handlers.clear()
        with L._LOGGING_LOCK:
            L._LOGGING_CONFIGURED = False
            L._configured = False
            if L._listener is not None:
                try:
                    L._listener.stop()
                except Exception:
                    pass
            L._listener = None
            L._log_queue = None
        logger1 = L.setup_logging()
        first = len(root.handlers)
        logger2 = L.setup_logging()
        second = len(root.handlers)
        logger3 = L.setup_logging()
        third = len(root.handlers)
        assert first == second == third
        assert logger1 is logger2 is logger3
    finally:
        root.handlers = original_handlers
        with L._LOGGING_LOCK:
            L._LOGGING_CONFIGURED = False
            L._configured = False
            if L._listener is not None:
                try:
                    L._listener.stop()
                except Exception:
                    pass
            L._listener = None
            L._log_queue = None

