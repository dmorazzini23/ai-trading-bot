import importlib
import logging
from unittest import mock

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


def test_logging_reload_with_null_settings(monkeypatch):
    """Reload module when settings resolve to ``None`` without error."""

    monkeypatch.setenv("PYTEST_RUNNING", "1")

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

        with mock.patch("ai_trading.config.get_settings", return_value=None):
            reloaded = importlib.reload(L)
            logger = reloaded.setup_logging()
            assert isinstance(logger, logging.Logger)

            handlers = [h for h in logging.getLogger().handlers if getattr(h, "formatter", None)]
            assert handlers, "expected at least one configured handler"
            assert any(isinstance(h.formatter, reloaded.JSONFormatter) for h in handlers)

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
        importlib.reload(L)

