from types import SimpleNamespace
from datetime import time
import threading
import pandas as pd

from ai_trading.core import bot_engine, execution_flow


def test_execute_entry_logs_trade(tmp_path, monkeypatch):
    """execute_entry should create trade log and append entry."""

    log_path = tmp_path / "trades.jsonl"
    monkeypatch.setattr(bot_engine, "TRADE_LOG_FILE", str(log_path))
    bot_engine._TRADE_LOGGER_SINGLETON = None

    # Stub out heavy dependencies
    monkeypatch.setattr(bot_engine, "submit_order", lambda *a, **k: None)
    monkeypatch.setattr(bot_engine, "vwap_pegged_submit", lambda *a, **k: None)
    monkeypatch.setattr(bot_engine, "pov_submit", lambda *a, **k: None)
    monkeypatch.setattr(bot_engine, "POV_SLICE_PCT", 0)
    monkeypatch.setattr(bot_engine, "SLICE_THRESHOLD", 10)

    df = pd.DataFrame({"close": [100.0], "atr": [1.0]})
    monkeypatch.setattr(bot_engine, "fetch_minute_df_safe", lambda symbol: df)
    monkeypatch.setattr(bot_engine, "prepare_indicators", lambda raw: raw)
    monkeypatch.setattr(bot_engine, "get_latest_close", lambda df: float(df["close"].iloc[-1]))
    monkeypatch.setattr(bot_engine, "TAKE_PROFIT_FACTOR", 1.0)
    monkeypatch.setattr(bot_engine, "is_high_vol_regime", lambda: False)
    monkeypatch.setattr(
        bot_engine,
        "scaled_atr_stop",
        lambda entry, atr, now, mo, mc, max_factor, min_factor: (entry - 1, entry + 1),
    )
    monkeypatch.setattr(bot_engine, "targets_lock", threading.Lock())

    ctx = SimpleNamespace(
        api=SimpleNamespace(get_account=lambda: SimpleNamespace(buying_power="10000")),
        trade_logger=None,
        market_open=time(9, 30),
        market_close=time(16, 0),
        stop_targets={},
        take_profit_targets={},
    )

    execution_flow.execute_entry(ctx, "AAPL", 1, "buy")

    lines = log_path.read_text().splitlines()
    assert len(lines) == 2
    assert "AAPL" in lines[1]
