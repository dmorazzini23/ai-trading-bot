import types
import pandas as pd
import ai_trading.core.bot_engine as be

def _mk_df():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="D"),
            "open": [1,2,3], "high":[1,2,3], "low":[1,2,3], "close":[1,2,3], "volume":[100,200,300]
        }
    )

def test_daily_fallback_parallel(monkeypatch):
    # Force batch to return empty so we hit fallback path for all.
    ctx = types.SimpleNamespace()
    calls = {"single": []}
    monkeypatch.setattr(be, "get_bars_batch", lambda *a, **k: {})
    def fake_single(sym, *a, **k):
        calls["single"].append(sym)
        return _mk_df()
    monkeypatch.setattr(be, "get_bars", fake_single)
    out = be._fetch_universe_bars(ctx, ["A","B","C","D"], "1D", "2024-01-01", "2024-02-01", None)
    assert set(out.keys()) == {"A","B","C","D"}
    # We can't assert true parallelism, but we can ensure all fallbacks executed.
    assert set(calls["single"]) == {"A","B","C","D"}

def test_intraday_fallback_parallel(monkeypatch):
    ctx = types.SimpleNamespace()
    calls = {"single": []}
    monkeypatch.setattr(be, "get_minute_bars_batch", lambda *a, **k: {})
    def fake_single(sym, *a, **k):
        calls["single"].append(sym)
        return _mk_df()
    monkeypatch.setattr(be, "get_minute_bars", fake_single)
    out = be._fetch_intraday_bars_chunked(ctx, ["X","Y","Z"], "2024-01-01 09:30", "2024-01-01 10:30", None)
    assert set(out.keys()) == {"X","Y","Z"}
    assert set(calls["single"]) == {"X","Y","Z"}