from __future__ import annotations

from tests.optdeps import require
require("pandas")

import pandas as pd
from ai_trading.data import bars as bars_mod

# AI-AGENT-REF: test central helpers
from ai_trading.logging.normalize import (
    canon_feed as _canon_feed,
)
from ai_trading.logging.normalize import (
    canon_timeframe as _canon_tf,
)


def _dummy_callable():  # AI-AGENT-REF: simulate stub attribute
    return None


def test_canonicalizers_map_weird_values_to_safe_defaults():
    assert _canon_tf(_dummy_callable) == "1Day"
    assert _canon_tf("1m") == "1Min"
    assert _canon_tf("1Min") == "1Min"
    assert _canon_tf("DAY") == "1Day"

    assert _canon_feed(_dummy_callable) == "sip"
    assert _canon_feed("IEX") == "iex"
    assert _canon_feed("sip") == "sip"


def test_safe_get_stock_bars_uses_canonicalized_values(monkeypatch):
    class Req:
        timeframe = _dummy_callable
        feed = _dummy_callable
        start = None
        end = None

    captured: dict[str, str] = {}

    def fake_http_get_bars(symbol, timeframe, start, end, *, feed=None):
        captured["timeframe"] = timeframe
        captured["feed"] = feed
        return pd.DataFrame()

    monkeypatch.setattr(bars_mod, "http_get_bars", fake_http_get_bars)

    class DummyClient:
        class _Resp:
            df = pd.DataFrame()

        def get_stock_bars(self, request):  # pragma: no cover - simple stub
            return self._Resp()

    bars_mod.safe_get_stock_bars(DummyClient(), Req(), symbol="SPY", context="DAILY")

    assert captured["timeframe"] in {"1Day", "1Min"}
    assert captured["feed"] in {"iex", "sip"}
