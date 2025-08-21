from __future__ import annotations

import logging
from datetime import date

from ai_trading.data.bars import (
    _format_fallback_payload,
    _log_fallback_window_debug,
)
from ai_trading.data.timeutils import nyse_session_utc


def test_fallback_payload_is_canonical(caplog):
    d = date(2025, 8, 20)
    start_u, end_u = nyse_session_utc(d)
    payload = _format_fallback_payload("1Min", "iex", start_u, end_u)
    assert isinstance(payload, list) and len(payload) == 4
    tf, feed, s, e = payload
    assert tf == "1Min"
    assert feed in {"iex", "sip"}
    assert s.endswith("+00:00") or s.endswith("Z")
    assert e.endswith("+00:00") or e.endswith("Z")
    logger = logging.getLogger("test_fallback_payload")
    with caplog.at_level("DEBUG"):
        _log_fallback_window_debug(logger, d, start_u, end_u)
    records = [r for r in caplog.records if r.message == "DATA_FALLBACK_WINDOW_DEBUG"]
    assert len(records) == 1

