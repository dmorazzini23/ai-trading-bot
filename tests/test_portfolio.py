import types
import pandas as pd
import bot_engine
import pytest


def test_short_close_queued(monkeypatch, caplog):
    state = bot_engine.BotState()
    state.position_cache = {"TSLA": -44}
    caplog.set_level("INFO")

    orders = []
    monkeypatch.setattr(bot_engine, "submit_order", lambda ctx, symbol, qty, side: orders.append((symbol, qty, side)))
    monkeypatch.setattr(bot_engine, "is_market_open", lambda: True)
    monkeypatch.setattr(bot_engine, "fetch_minute_df_safe", lambda s: pd.DataFrame({"close": [1]}, index=[pd.Timestamp("2023-01-01")]))
    monkeypatch.setattr(bot_engine, "_safe_trade", lambda *a, **k: None)
    monkeypatch.setattr(bot_engine.prediction_executor, "submit", lambda fn, s: types.SimpleNamespace(result=lambda: fn(s)))
    monkeypatch.setattr(bot_engine, "log_skip_cooldown", lambda *a, **k: None)
    bot_engine.skipped_duplicates = types.SimpleNamespace(inc=lambda: None)

    processed, _ = bot_engine._process_symbols(["TSLA"], 1000.0, None, True, True)
    assert processed == []
    assert orders == []
    assert any("SKIP_SHORT_CLOSE_QUEUED" in r.message for r in caplog.records)
