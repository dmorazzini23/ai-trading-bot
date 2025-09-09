import logging
import os

import ai_trading.logging as L


def test_configure_logging_logs_once(capsys):
    root = logging.getLogger()
    original_handlers = root.handlers.copy()
    try:
        root.handlers.clear()
        os.environ['PYTEST_RUNNING'] = '1'
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
            L._loggers.clear()
        L.configure_logging()
        L.configure_logging()
        out = capsys.readouterr().out
        assert out.count('Logging configured successfully - no duplicates possible') == 1
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
            L._loggers.clear()
        os.environ.pop('PYTEST_RUNNING', None)
