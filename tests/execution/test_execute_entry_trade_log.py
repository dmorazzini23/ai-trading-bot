from types import SimpleNamespace
from datetime import time
import threading
import os

import pandas as pd
import pytest

from ai_trading.core import bot_engine, execution_flow
from ai_trading.execution import ExecutionEngine
from ai_trading.core.enums import OrderSide, OrderType


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
    monkeypatch.setattr(bot_engine, "get_take_profit_factor", lambda: 1.0)
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


def test_slippage_converts_market_to_limit(monkeypatch):
    os.environ["TESTING"] = "true"
    monkeypatch.setenv("MAX_SLIPPAGE_BPS", "10")
    monkeypatch.setenv("SLIPPAGE_LIMIT_TOLERANCE_BPS", "5")
    monkeypatch.setattr("ai_trading.execution.engine.hash", lambda x: 99, raising=False)
    engine = ExecutionEngine()
    oid = engine.execute_order("AAPL", OrderSide.BUY, 10, expected_price=100.0)
    order = engine.order_manager.orders[oid]
    assert order.order_type == OrderType.LIMIT
    expected_price = round(100.0 + (100.0 * 5 / 10000), 4)
    assert round(float(order.price), 4) == expected_price


def test_slippage_reduces_order_size(monkeypatch):
    os.environ["TESTING"] = "true"
    monkeypatch.setenv("MAX_SLIPPAGE_BPS", "10")
    monkeypatch.setattr("ai_trading.execution.engine.hash", lambda x: 99, raising=False)
    engine = ExecutionEngine()
    oid = engine.execute_order("AAPL", OrderSide.BUY, 10)
    order = engine.order_manager.orders[oid]
    assert order.quantity < 10
