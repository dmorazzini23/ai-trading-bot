from datetime import datetime, timedelta, timezone
import types
import pandas as pd
import ai_trading.core.bot_engine as be
import ai_trading.trade_logic as tl

def _mk_df():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="D"),
            "open": [1,2,3], "high":[1,2,3], "low":[1,2,3], "close":[1,2,3], "volume":[100,200,300]
        }
    )

def test_regime_batch(monkeypatch):
    ctx = types.SimpleNamespace(regime_symbols=["SPY","QQQ"], data_feed=None, regime_lookback_days=60)
    monkeypatch.setattr(be, "_fetch_regime_bars", lambda ctx, **kwargs: {"SPY": _mk_df(), "QQQ": _mk_df()})
    out = be._build_regime_dataset(ctx)
    assert "SPY" in out.columns and "QQQ" in out.columns

def test_pretrade_batch(monkeypatch):
    ctx = types.SimpleNamespace(lookback_start="2024-01-01", lookback_end="2024-02-01", data_feed=None, min_rows=2)
    monkeypatch.setattr(be, "_fetch_universe_bars_chunked", lambda **kwargs: {"AAPL": _mk_df()})
    res = be.pre_trade_health_check(ctx, ["AAPL"])
    assert res["checked"] == 1

def test_intraday_entries_and_exits(monkeypatch):
    ctx = types.SimpleNamespace(logger=types.SimpleNamespace(warning=lambda *a, **k: None), data_feed=None)
    
    # Mock the chunked fetch function directly in trade_logic module to avoid API calls
    def mock_fetch_intraday(*args, **kwargs):
        return {"AAPL": _mk_df(), "MSFT": _mk_df()}
    
    # Patch both the import and the function to ensure it's mocked
    monkeypatch.setattr("ai_trading.trade_logic._fetch_intraday_bars_chunked", mock_fetch_intraday)
    monkeypatch.setattr(tl, "_compute_entry_signal", lambda ctx, sym, df: {"buy": True})
    monkeypatch.setattr(tl, "_compute_exit_signal", lambda ctx, sym, df: {"sell": True})
    
    entries = tl.evaluate_entries(ctx, ["AAPL","MSFT"])
    exits = tl.evaluate_exits(ctx, {"AAPL": {}})
    assert "AAPL" in entries and "MSFT" in entries
    assert "AAPL" in exits