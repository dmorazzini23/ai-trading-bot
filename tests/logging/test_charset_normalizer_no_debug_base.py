"""Ensure charset_normalizer debug logs are suppressed using base setup_logging."""

import logging

import ai_trading.logging as base_logger


def _reset_logging_state() -> None:
    """Reset global logging state for isolated tests."""
    base_logger._configured = False
    base_logger._LOGGING_CONFIGURED = False
    base_logger._listener = None
    base_logger._log_queue = None
    logging.getLogger().handlers.clear()


def test_charset_normalizer_no_debug_with_base(monkeypatch, caplog) -> None:
    """Calling ai_trading.logging.setup_logging should silence charset_normalizer debug."""
    _reset_logging_state()
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.delenv("LOG_QUIET_LIBRARIES", raising=False)

    base_logger.setup_logging()

    with caplog.at_level(logging.DEBUG):
        logging.getLogger("charset_normalizer").debug("noisy debug")

    assert "noisy debug" not in caplog.text
    _reset_logging_state()
