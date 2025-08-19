"""Regression tests for QueueListener logging setup idempotency."""

from ai_trading.logging import get_logger, setup_logging


def test_setup_logging_idempotent_no_runtime_error():
    """Multiple calls to setup_logging should not raise errors."""
    setup_logging()
    setup_logging()


def test_logger_emits_after_setup():
    """Logger should emit records after setup."""
    setup_logging()
    log = get_logger(__name__)
    log.info("queue-listener-ok")

