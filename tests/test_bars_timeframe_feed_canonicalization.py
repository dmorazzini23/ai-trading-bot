from __future__ import annotations


import pytest

pd = pytest.importorskip("pandas")
from ai_trading.data import bars as bars_mod

# AI-AGENT-REF: test central helpers
from ai_trading.logging.normalize import (
    canon_feed as _canon_feed,
)
from ai_trading.logging.normalize import (
    canon_symbol as _canon_symbol,
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

    assert _canon_symbol("brk-b") == "BRK.B"
    assert _canon_symbol("AAPL") == "AAPL"


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


def test_safe_get_stock_bars_normalizes_symbol(monkeypatch):
    class APIError(Exception):
        pass

    called: dict[str, str] = {}

    class Client:
        class _Resp:
            df = pd.DataFrame()

        def get_stock_bars(self, request):
            sym = request.symbol_or_symbols
            if isinstance(sym, list):
                sym = sym[0]
            if sym == "BRK-B":
                raise APIError("bad symbol")
            called["symbol"] = sym
            return self._Resp()

    req = bars_mod.StockBarsRequest(symbol_or_symbols="BRK-B", timeframe=bars_mod.TimeFrame.Day)
    bars_mod.safe_get_stock_bars(Client(), req, symbol="BRK-B", context="TEST")

    assert called["symbol"] == "BRK.B"
