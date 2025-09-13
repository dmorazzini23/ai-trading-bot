import logging

import ai_trading.logging as logger  # Use centralized logging module


def test_setup_logging_with_file(monkeypatch, tmp_path):
    """File handler is added when log_file is provided."""
    root = logging.getLogger()
    original_handlers = root.handlers.copy()
    orig_configured = logger._configured
    orig_logging_configured = logger._LOGGING_CONFIGURED
    try:
        logger._configured = False
        logger._LOGGING_CONFIGURED = True
        fake = logging.NullHandler()

        def fake_makedirs(path, exist_ok=False):
            pass

        calls = []

        def fake_get_handler(*args, **kwargs):
            calls.append((args, kwargs))
            return fake

        monkeypatch.setattr(logger.os, "makedirs", fake_makedirs)
        monkeypatch.setattr(logger, "get_rotating_handler", fake_get_handler)

        log_file = tmp_path / "x" / "app.log"
        logger.setup_logging(log_file=str(log_file))
        assert calls
    finally:
        root.handlers = original_handlers
        logger._configured = orig_configured
        logger._LOGGING_CONFIGURED = orig_logging_configured
        logger._listener = None
        logger._log_queue = None
