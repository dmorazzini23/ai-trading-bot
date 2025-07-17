import types
import pandas as pd
import bot_engine


def test_skip_logic(monkeypatch, caplog):
    state = bot_engine.BotState()
    state.position_cache = {"MSFT": 10, "TSLA": -10}
    caplog.set_level("INFO")

    orders = []
    monkeypatch.setattr(bot_engine, "submit_order", lambda ctx, symbol, qty, side: orders.append((symbol, qty, side)))
    monkeypatch.setattr(bot_engine, "is_market_open", lambda: True)
    monkeypatch.setattr(bot_engine, "fetch_minute_df_safe", lambda s: pd.DataFrame({"close": [1]}, index=[pd.Timestamp("2023-01-01")]))
    monkeypatch.setattr(bot_engine, "_safe_trade", lambda *a, **k: None)
    monkeypatch.setattr(bot_engine.prediction_executor, "submit", lambda fn, s: types.SimpleNamespace(result=lambda: fn(s)))
    monkeypatch.setattr(bot_engine, "log_skip_cooldown", lambda *a, **k: None)
    monkeypatch.setattr(bot_engine.skipped_duplicates, "inc", lambda: None)

    processed, _ = bot_engine._process_symbols(["MSFT", "TSLA"], 1000.0, None, True)
    assert processed == []
    assert ("TSLA", 10, "buy") in orders
    assert all(o[0] != "MSFT" for o in orders)
    msgs = [r.message for r in caplog.records]
    assert any("SHORT_CLOSE_QUEUED" in m and "TSLA" in m for m in msgs)
    assert any("SKIP_HELD_POSITION" in m and "MSFT" in m for m in msgs)
