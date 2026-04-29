"""Regression tests for QueueListener logging setup idempotency."""

import logging

import ai_trading.logging as logging_module
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


def test_asyncio_task_capture_disabled_for_runtime_stability():
    """Task-name capture is disabled to avoid Python 3.12 startup crashes."""
    assert getattr(logging, "logAsyncioTasks", None) is False


def test_bounded_queue_handler_counts_drops():
    q = logging_module.queue.Queue(maxsize=1)
    handler = logging_module.BoundedQueueHandler(q)
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "msg", (), None)
    before = logging_module.get_logging_queue_stats()["dropped"]

    handler.enqueue(record)
    handler.enqueue(record)

    stats = logging_module.get_logging_queue_stats()
    assert q.qsize() == 1
    assert stats["dropped"] == before + 1
    assert stats["backpressure_events"] >= 1
