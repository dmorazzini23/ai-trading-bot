from __future__ import annotations

import logging

from ai_trading.logging import SanitizingLoggerAdapter


class _CaptureHandler(logging.Handler):
    """Capture the last log record for assertions."""

    def __init__(self):
        super().__init__()
        self.last: logging.LogRecord | None = None

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
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
                "msg": "XMSG",         # reserved key collision (message)
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

    # The extra values should appear under a *prefixed* key, not the reserved names
    d = rec.__dict__
    has_level_extra = any(
        k != "levelname" and "levelname" in k and d[k] == "XLEVEL" for k in d
    )
    has_message_extra = any(
        k != "msg" and "msg" in k and d[k] == "XMSG" for k in d
    )
    assert has_level_extra, f"Collision key 'levelname' was not prefixed: keys={list(d.keys())}"
    assert has_message_extra, f"Collision key 'msg' was not prefixed: keys={list(d.keys())}"

