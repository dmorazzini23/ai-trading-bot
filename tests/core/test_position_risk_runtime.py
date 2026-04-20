from types import SimpleNamespace

import pandas as pd

from ai_trading.core import bot_engine


def test_manage_position_risk_sanitizes_invalid_atr_and_vwap(monkeypatch):
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        bot_engine.utils,
        "get_rolling_atr",
        lambda _symbol: "bad",
        raising=False,
    )
    monkeypatch.setattr(
        bot_engine.utils,
        "get_current_vwap",
        lambda _symbol: object(),
        raising=False,
    )
    monkeypatch.setattr(
        bot_engine.utils,
        "get_volume_spike_factor",
        lambda _symbol: 10.0,
        raising=False,
    )
    monkeypatch.setattr(
        bot_engine.utils,
        "get_ml_confidence",
        lambda _symbol: 1.0,
        raising=False,
    )
    monkeypatch.setattr(
        bot_engine,
        "fetch_minute_df_safe",
        lambda _symbol: pd.DataFrame({"close": [100.0]}),
    )
    monkeypatch.setattr(
        bot_engine,
        "update_trailing_stop",
        lambda ctx, symbol, price, qty, atr: calls.setdefault(
            "trailing_stop",
            (ctx, symbol, price, qty, atr),
        ),
    )
    monkeypatch.setattr(bot_engine, "compute_kelly_scale", lambda atr, _edge: float(atr))
    monkeypatch.setattr(
        bot_engine,
        "adjust_position_size",
        lambda position, kelly: calls.setdefault("adjust", (position.symbol, kelly)),
    )
    monkeypatch.setattr(
        bot_engine,
        "pyramid_add_position",
        lambda ctx, symbol, fraction, side: calls.setdefault(
            "pyramid",
            (ctx, symbol, fraction, side),
        ),
    )

    position = SimpleNamespace(
        symbol="AAPL",
        qty="10",
        avg_entry_price="95.0",
        unrealized_plpc="0.03",
    )
    ctx = SimpleNamespace()

    bot_engine.manage_position_risk(ctx, position)

    assert calls["trailing_stop"] == (ctx, "AAPL", 100.0, 10, 0.0)
    assert calls["adjust"] == ("AAPL", 0.0)
    assert "pyramid" not in calls


def test_manage_position_risk_tolerates_missing_optional_utils_helpers(monkeypatch):
    calls: dict[str, object] = {}

    monkeypatch.delattr(bot_engine.utils, "get_volume_spike_factor", raising=False)
    monkeypatch.delattr(bot_engine.utils, "get_ml_confidence", raising=False)
    monkeypatch.setattr(
        bot_engine.utils,
        "get_rolling_atr",
        lambda _symbol: 1.0,
        raising=False,
    )
    monkeypatch.setattr(
        bot_engine.utils,
        "get_current_vwap",
        lambda _symbol: 90.0,
        raising=False,
    )
    monkeypatch.setattr(
        bot_engine,
        "fetch_minute_df_safe",
        lambda _symbol: pd.DataFrame({"close": [100.0]}),
    )
    monkeypatch.setattr(
        bot_engine,
        "update_trailing_stop",
        lambda ctx, symbol, price, qty, atr: calls.setdefault(
            "trailing_stop",
            (ctx, symbol, price, qty, atr),
        ),
    )
    monkeypatch.setattr(bot_engine, "compute_kelly_scale", lambda atr, _edge: float(atr))
    monkeypatch.setattr(
        bot_engine,
        "adjust_position_size",
        lambda position, kelly: calls.setdefault("adjust", (position.symbol, kelly)),
    )

    position = SimpleNamespace(
        symbol="AAPL",
        qty="10",
        avg_entry_price="95.0",
        unrealized_plpc="0.03",
    )
    ctx = SimpleNamespace()

    bot_engine.manage_position_risk(ctx, position)

    assert calls["trailing_stop"] == (ctx, "AAPL", 100.0, 10, 1.0)
    assert calls["adjust"] == ("AAPL", 1.0)
