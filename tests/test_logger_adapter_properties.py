from __future__ import annotations

import logging

from ai_trading.logging import get_logger


def test_get_logger_handlers_is_mutable_and_propagate_toggle() -> None:
    """Handlers list is mutable and propagate flag can be toggled."""
    logger = get_logger("ai_trading.tests.adapter_props")
    original_handlers = list(logger.handlers)
    try:
        assert isinstance(logger.handlers, list)
        sentinel = logging.NullHandler()
        logger.handlers.append(sentinel)
        assert sentinel in logger.handlers
        logger.propagate = False
        assert logger.propagate is False
        logger.propagate = True
        assert logger.propagate is True
    finally:
        logger.handlers = original_handlers
        logger.propagate = True
