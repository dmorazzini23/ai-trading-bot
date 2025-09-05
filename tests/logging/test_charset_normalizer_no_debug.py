"""Validate that charset_normalizer does not emit debug logs once configured."""

import logging

import ai_trading.logging as base_logger
import ai_trading.logging.setup as log_setup


def _reset_logging_state() -> None:
    """Reset global logging state for isolated tests."""
    base_logger._configured = False
    base_logger._LOGGING_CONFIGURED = False
    base_logger._listener = None
    base_logger._log_queue = None
    logging.getLogger().handlers.clear()


def test_charset_normalizer_no_debug(monkeypatch, caplog) -> None:
    """After setup, charset_normalizer debug logs should be suppressed."""
    _reset_logging_state()
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.delenv("LOG_QUIET_LIBRARIES", raising=False)

    log_setup.setup_logging()

    with caplog.at_level(logging.DEBUG):
        logging.getLogger("charset_normalizer").debug("noisy debug")

    assert "noisy debug" not in caplog.text
    _reset_logging_state()

