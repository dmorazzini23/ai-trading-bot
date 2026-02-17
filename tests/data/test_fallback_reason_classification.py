from __future__ import annotations

from ai_trading.data import fetch


def test_classify_configured_daily_source_reason() -> None:
    reason, details = fetch._classify_fallback_reason(
        "configured_daily_source",
        {"symbol": "AAPL", "timeframe": "1Day"},
    )
    assert reason == "configured_source_override"
    assert details["symbol"] == "AAPL"
