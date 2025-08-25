"""Ensure yfinance fallback sets auto_adjust and TzCache."""
from tests.optdeps import require
require("pandas")

import sys
import types
from datetime import UTC, datetime


def test_yfinance_auto_adjust_and_cache(monkeypatch):
    calls = {"auto_adjust": None, "cache_called": False}

    fake = types.SimpleNamespace()

    def set_tz_cache_location(path):  # AI-AGENT-REF: track tz cache invocation
        calls["cache_called"] = True

    def download(*args, auto_adjust=None, **kwargs):  # AI-AGENT-REF: capture auto_adjust
        calls["auto_adjust"] = auto_adjust
        import pandas as pd

        return pd.DataFrame(
            {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [100]},
            index=pd.date_range(datetime(2025, 8, 1, tzinfo=UTC), periods=1, name="Date"),
        )

    fake.set_tz_cache_location = set_tz_cache_location
    fake.download = download
    monkeypatch.setitem(sys.modules, "yfinance", fake)

    from ai_trading.data_fetcher import _yahoo_get_bars

    _ = _yahoo_get_bars("SPY", datetime(2025, 8, 1, tzinfo=UTC), datetime(2025, 8, 2, tzinfo=UTC), "1Day")

    assert calls["auto_adjust"] is True
    assert calls["cache_called"] is True
