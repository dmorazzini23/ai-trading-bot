from __future__ import annotations

import inspect
import logging
import pytest

import ai_trading.logging as L


def test_validate_logging_setup_single_handler():
    """Ensure validate_logging_setup() deduplicates handlers."""
    if not hasattr(L, "validate_logging_setup"):
        pytest.skip(
            "validate_logging_setup() not present in ai_trading.logging; add it or adjust test."
        )

    sig = inspect.signature(L.validate_logging_setup)
    if len(sig.parameters) != 1:
        pytest.skip(
            "validate_logging_setup() does not accept a logger argument; dedup test skipped."
        )

    # Use a dedicated logger name to avoid global handlers
    logger = logging.getLogger("ai_trading.tests.single_handler")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)

    # Add two duplicate StreamHandlers deliberately
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.StreamHandler())
    pre_count = sum(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    assert pre_count == 2, "Fixture assumption failed: expected two handlers"

    count = L.validate_logging_setup(logger)  # expected to dedupe
    post_count = sum(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    assert post_count == 1, f"Expected a single StreamHandler, found {post_count}"
    if isinstance(count, int):
        assert count == len(logger.handlers)

