from __future__ import annotations

from datetime import date

from ai_trading import data_fetcher as df
from ai_trading.data.timeutils import nyse_session_utc


def test_canonical_helpers() -> None:
    assert df._to_timeframe_str("1 minute") == "1Min"
    assert df._to_timeframe_str("1Min") == "1Min"
    assert df._to_timeframe_str("day") == "1Day"

    class Weird:
        def __str__(self) -> str:  # AI-AGENT-REF: simulate odd repr
            return "<function _noop>"

    assert df._to_timeframe_str(Weird()) in {"1Day", "1Min"}
    assert df._to_feed_str("IEX") == "iex"
    assert df._to_feed_str("sip") == "sip"
    assert df._to_feed_str(object()) in {"sip", "iex"}


def test_fallback_payload_is_canonical_df() -> None:
    d = date(2025, 8, 20)
    start_u, end_u = nyse_session_utc(d)
    payload = df._format_fallback_payload_df("1Min", "iex", start_u, end_u)
    assert isinstance(payload, list) and len(payload) == 4
    tf, feed, s, e = payload
    assert tf == "1Min"
    assert feed in {"iex", "sip"}
    assert s.endswith("+00:00") or s.endswith("Z")
    assert e.endswith("+00:00") or e.endswith("Z")

