import types

import pytest

pd = pytest.importorskip("pandas")
from ai_trading.core import bot_engine


def test_skip_logic(monkeypatch, caplog):
    state = bot_engine.BotState()
    state.position_cache = {"MSFT": 10, "TSLA": -10}
    bot_engine.state = state
    caplog.set_level("INFO")

    orders = []
    monkeypatch.setattr(bot_engine, "submit_order", lambda ctx, symbol, qty, side: orders.append((symbol, qty, side)))
    monkeypatch.setattr(bot_engine, "is_market_open", lambda: True)
    monkeypatch.setattr(bot_engine, "fetch_minute_df_safe", lambda s: pd.DataFrame({"close": [1]}, index=[pd.Timestamp("2023-01-01")]))
    monkeypatch.setattr(bot_engine, "_safe_trade", lambda *a, **k: None)
    monkeypatch.setattr(
        bot_engine,
        "prediction_executor",
        types.SimpleNamespace(submit=lambda fn, s: types.SimpleNamespace(result=lambda: fn(s))),
    )
    monkeypatch.setattr(bot_engine, "log_skip_cooldown", lambda *a, **k: None)
    monkeypatch.setattr(
        bot_engine,
        "skipped_duplicates",
        types.SimpleNamespace(inc=lambda: None),
        raising=False,
    )

    processed, _ = bot_engine._process_symbols(
        ["MSFT", "TSLA"], 1000.0, None, True, True
    )
    assert processed == []
    assert orders == []


def test_skip_duplicates(monkeypatch):
    """Symbols with existing positions are skipped when skip_duplicates=True."""

    state = bot_engine.BotState()
    state.position_cache = {"AAPL": 5}
    bot_engine.state = state

    orders: list = []
    monkeypatch.setattr(
        bot_engine, "submit_order", lambda ctx, symbol, qty, side: orders.append((symbol, qty, side))
    )
    monkeypatch.setattr(bot_engine, "is_market_open", lambda: True)
    monkeypatch.setattr(
        bot_engine,
        "fetch_minute_df_safe",
        lambda s: pd.DataFrame({"close": [1]}, index=[pd.Timestamp("2023-01-01")]),
    )
    monkeypatch.setattr(bot_engine, "_safe_trade", lambda *a, **k: None)
    monkeypatch.setattr(
        bot_engine,
        "prediction_executor",
        types.SimpleNamespace(submit=lambda fn, s: types.SimpleNamespace(result=lambda: fn(s))),
    )
    logs: list[tuple[str, str]] = []
    monkeypatch.setattr(
        bot_engine, "log_skip_cooldown", lambda symbol, reason: logs.append((symbol, reason))
    )
    inc_calls: list[int] = []
    monkeypatch.setattr(
        bot_engine,
        "skipped_duplicates",
        types.SimpleNamespace(inc=lambda: inc_calls.append(1)),
        raising=False,
    )

    processed, _ = bot_engine._process_symbols(
        ["AAPL"], 1000.0, None, True, False, True
    )
    assert processed == []
    assert orders == []
    assert logs == [("AAPL", "duplicate")]
    assert inc_calls == [1]
