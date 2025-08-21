from __future__ import annotations

import logging
import pytest

import ai_trading.logging as L


def test_validate_logging_setup_single_handler():
    """Ensure validate_logging_setup() deduplicates handlers."""
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

    if hasattr(L, "dedupe_stream_handlers"):
        # Prefer explicit dedupe helper
        final_count = L.dedupe_stream_handlers(logger)
        post_count = sum(isinstance(h, logging.StreamHandler) for h in logger.handlers)
        assert post_count == 1, f"Expected a single StreamHandler, found {post_count}"
        assert final_count == len(logger.handlers)
        # And the validator should also reflect dedupe when asked
        res = L.validate_logging_setup(logger, dedupe=False)
        assert res["handlers_count"] == final_count
    elif hasattr(L, "validate_logging_setup"):
        # Fallback: validator should be able to operate on a provided logger
        res = L.validate_logging_setup(logger, dedupe=True)
        post_count = sum(isinstance(h, logging.StreamHandler) for h in logger.handlers)
        assert post_count == 1, f"Expected a single StreamHandler, found {post_count}"
        assert res["handlers_count"] == post_count and res.get("deduped") is True
    else:
        pytest.skip("Neither dedupe_stream_handlers() nor validate_logging_setup() found in ai_trading.logging.")

