from types import SimpleNamespace
from datetime import time
import threading
import pandas as pd
import pytest

from ai_trading.core import bot_engine, execution_flow

def test_execute_entry_uses_config_max_factor(monkeypatch):
    monkeypatch.setattr(bot_engine, "submit_order", lambda *a, **k: None)
    monkeypatch.setattr(bot_engine, "vwap_pegged_submit", lambda *a, **k: None)
    monkeypatch.setattr(bot_engine, "pov_submit", lambda *a, **k: None)
    monkeypatch.setattr(bot_engine, "POV_SLICE_PCT", 0)
    monkeypatch.setattr(bot_engine, "SLICE_THRESHOLD", 10)
    df = pd.DataFrame({"close": [100.0], "atr": [1.0]})
    monkeypatch.setattr(bot_engine, "fetch_minute_df_safe", lambda symbol: df)
    monkeypatch.setattr(bot_engine, "prepare_indicators", lambda raw: raw)
    monkeypatch.setattr(bot_engine, "get_latest_close", lambda df: float(df["close"].iloc[-1]))
    monkeypatch.setattr(bot_engine, "is_high_vol_regime", lambda: False)
    captured = {}
    def fake_scaled_atr_stop(entry, atr, now, mo, mc, max_factor, min_factor):
        captured["max_factor"] = max_factor
        return (entry - 1, entry + 1)
    monkeypatch.setattr(bot_engine, "scaled_atr_stop", fake_scaled_atr_stop)
    monkeypatch.setattr(bot_engine, "targets_lock", threading.Lock())
    monkeypatch.setenv("TAKE_PROFIT_FACTOR", "3.0")

    ctx = SimpleNamespace(
        api=SimpleNamespace(get_account=lambda: SimpleNamespace(buying_power="10000")),
        trade_logger=None,
        market_open=time(9, 30),
        market_close=time(16, 0),
        stop_targets={},
        take_profit_targets={},
    )

    execution_flow.execute_entry(ctx, "AAPL", 1, "buy")
    assert captured["max_factor"] == 3.0


def test_get_take_profit_factor_invalid(monkeypatch):
    monkeypatch.setenv("TAKE_PROFIT_FACTOR", "not-a-number")
    with pytest.raises(ValueError):
        bot_engine.get_take_profit_factor()
