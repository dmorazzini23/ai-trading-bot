"""Verify charset_normalizer logs are suppressed with default setup."""
import logging

import ai_trading.logging as base_logger
import ai_trading.logging.setup as log_setup


def _reset_logging_state() -> None:
    base_logger._configured = False
    base_logger._LOGGING_CONFIGURED = False
    base_logger._listener = None
    base_logger._log_queue = None
    logging.getLogger().handlers.clear()


def test_charset_normalizer_default_suppressed(monkeypatch, caplog) -> None:
    _reset_logging_state()
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.delenv("LOG_QUIET_LIBRARIES", raising=False)

    log_setup.setup_logging()

    noisy = logging.getLogger("charset_normalizer")
    assert noisy.getEffectiveLevel() == logging.WARNING
    with caplog.at_level(logging.DEBUG):
        noisy.debug("noisy debug")
    assert "noisy debug" not in caplog.text
    _reset_logging_state()
