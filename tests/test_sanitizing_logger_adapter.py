from __future__ import annotations

import logging

from ai_trading.logging import SanitizingLoggerAdapter


class _CaptureHandler(logging.Handler):
    """Capture the last log record for assertions."""

    def __init__(self):
        super().__init__()
        self.last: logging.LogRecord | None = None

    def emit(self, record: logging.LogRecord) -> None:
        self.last = record


def test_extra_key_collisions_are_prefixed():
    """Ensure reserved keys in extra dict are prefixed, preserving core fields."""
    # Fresh, isolated logger
    logger = logging.getLogger("ai_trading.tests.collisions")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)

    adapter = SanitizingLoggerAdapter(logger, {})
    handler = _CaptureHandler()
    logger.addHandler(handler)
    try:
        adapter.info(
            "hello",
            extra={
                "levelname": "XLEVEL",  # reserved key collision
                "message": "XMSG",      # reserved key collision (message)
                "feed": "iex",          # a normal key for contrast
            },
        )
    finally:
        logger.removeHandler(handler)

    rec = handler.last
    assert rec, "No log record captured"

    # Standard fields must remain intact
    assert rec.levelname == "INFO"
    assert rec.getMessage() == "hello"

    # The extra values should appear under the exact 'x_' prefix your adapter uses
    d = rec.__dict__
    assert d.get("x_levelname") == "XLEVEL", (
        f"Expected x_levelname='XLEVEL', got: {d.get('x_levelname')} (keys={list(d.keys())})"
    )
    assert d.get("x_message") == "XMSG", (
        f"Expected x_message='XMSG', got: {d.get('x_message')} (keys={list(d.keys())})"
    )

