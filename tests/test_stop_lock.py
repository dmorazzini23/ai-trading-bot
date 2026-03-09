import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from ai_trading.config.runtime import TradingConfig
from ai_trading.core import bot_engine


@pytest.fixture(autouse=True)
def _disable_slo_derisk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep stop-lock tests deterministic regardless of global SLO monitor state."""

    monkeypatch.setenv("AI_TRADING_DERISK_ON_SLO_BREACH_ENABLED", "0")


def test_stop_lock_blocks_reentry(monkeypatch):
    cfg = TradingConfig.from_env(allow_missing_drawdown=True)
    cfg.update(
        netting_enabled=True,
        data_contract_enabled=False,
        recon_enabled=False,
        ledger_enabled=False,
        rth_only=False,
        allow_extended=True,
        decision_log_path=None,
    )
    runtime = SimpleNamespace(cfg=cfg, tickers=["AAPL"], universe_tickers=["AAPL"], api=None)
    state = bot_engine.BotState()

    bar_ts = datetime.now(UTC)
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.0, 101.0],
            "volume": [1000, 1000],
        },
        index=pd.DatetimeIndex([bar_ts - timedelta(minutes=1), bar_ts], tz=UTC),
    )

    monkeypatch.setattr("ai_trading.data.fetch.get_bars_batch", lambda symbols, timeframe, start, end: {"AAPL": df})
    monkeypatch.setattr(bot_engine, "ensure_data_fetcher", lambda runtime: None)
    monkeypatch.setattr(bot_engine, "compute_current_positions", lambda runtime: {})

    records: list[dict] = []

    def _capture(record, path):
        records.append(record.to_dict())

    monkeypatch.setattr(bot_engine, "_write_decision_record", _capture)

    state.stop_lock["AAPL"] = {"bar_ts": bar_ts, "direction": "long"}

    bot_engine._run_netting_cycle(state, runtime, "loop", 0.0)
    assert records
    assert any("STOP_LOCK" in r.get("gates", []) for r in records)


def test_netting_cycle_loads_universe_when_runtime_symbols_missing(monkeypatch, caplog):
    cfg = TradingConfig.from_env(allow_missing_drawdown=True)
    cfg.update(
        netting_enabled=True,
        data_contract_enabled=False,
        recon_enabled=False,
        ledger_enabled=False,
        rth_only=False,
        allow_extended=True,
        decision_log_path=None,
    )
    runtime = SimpleNamespace(cfg=cfg, tickers=[], universe_tickers=[], api=None)
    state = bot_engine.BotState()

    bar_ts = datetime.now(UTC)
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.0, 101.0],
            "volume": [1000, 1000],
        },
        index=pd.DatetimeIndex([bar_ts - timedelta(minutes=1), bar_ts], tz=UTC),
    )

    monkeypatch.setattr(
        "ai_trading.data.fetch.get_bars_batch",
        lambda symbols, timeframe, start, end: {"AAPL": df},
    )
    monkeypatch.setattr(bot_engine, "ensure_data_fetcher", lambda runtime: None)
    monkeypatch.setattr(bot_engine, "compute_current_positions", lambda runtime: {})
    monkeypatch.setattr(bot_engine, "load_tickers", lambda path=bot_engine.TICKERS_FILE: ["AAPL"])

    records: list[dict] = []

    def _capture(record, path):
        records.append(record.to_dict())

    monkeypatch.setattr(bot_engine, "_write_decision_record", _capture)
    state.stop_lock["AAPL"] = {"bar_ts": bar_ts, "direction": "long"}

    with caplog.at_level(logging.WARNING):
        bot_engine._run_netting_cycle(state, runtime, "loop", 0.0)

    assert records
    assert runtime.tickers == ["AAPL"]
    assert runtime.universe_tickers == ["AAPL"]
    assert not any(record.getMessage() == "NETTING_NO_SYMBOLS" for record in caplog.records)


def test_netting_cycle_filters_pending_blocked_symbols(monkeypatch):
    cfg = TradingConfig.from_env(allow_missing_drawdown=True)
    cfg.update(
        netting_enabled=True,
        data_contract_enabled=False,
        recon_enabled=False,
        ledger_enabled=False,
        rth_only=False,
        allow_extended=True,
        decision_log_path=None,
    )
    runtime = SimpleNamespace(
        cfg=cfg,
        tickers=["AAPL", "MSFT"],
        universe_tickers=["AAPL", "MSFT"],
        api=None,
    )
    setattr(runtime, bot_engine._PENDING_ORDER_BLOCKED_SYMBOLS_ATTR, ("AAPL",))
    state = bot_engine.BotState()
    monkeypatch.setenv("AI_TRADING_PENDING_ORDERS_BLOCK_SCOPE", "symbol")

    bar_ts = datetime.now(UTC)
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.0, 101.0],
            "volume": [1000, 1000],
        },
        index=pd.DatetimeIndex([bar_ts - timedelta(minutes=1), bar_ts], tz=UTC),
    )

    seen_batches: list[list[str]] = []

    def _batch(symbols, timeframe, start, end):
        seen_batches.append([str(symbol).upper() for symbol in symbols])
        return {str(symbol).upper(): df for symbol in symbols}

    monkeypatch.setattr("ai_trading.data.fetch.get_bars_batch", _batch)
    monkeypatch.setattr(bot_engine, "ensure_data_fetcher", lambda runtime: None)
    monkeypatch.setattr(bot_engine, "compute_current_positions", lambda runtime: {})
    monkeypatch.setattr(bot_engine, "submit_order", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot_engine, "_write_decision_record", lambda record, path: None)

    bot_engine._run_netting_cycle(state, runtime, "loop", 0.0)

    assert seen_batches
    assert all("AAPL" not in batch for batch in seen_batches)
    assert all("MSFT" in batch for batch in seen_batches)


def test_netting_cycle_applies_execution_symbol_budget(monkeypatch):
    cfg = TradingConfig.from_env(allow_missing_drawdown=True)
    cfg.update(
        netting_enabled=True,
        data_contract_enabled=False,
        recon_enabled=False,
        ledger_enabled=False,
        rth_only=False,
        allow_extended=True,
        decision_log_path=None,
    )

    class _ExecEngine:
        def _resolve_order_submit_cap(self):
            return 1, "configured"

    runtime = SimpleNamespace(
        cfg=cfg,
        tickers=["AAPL", "MSFT", "GOOG"],
        universe_tickers=["AAPL", "MSFT", "GOOG"],
        api=None,
        execution_engine=_ExecEngine(),
    )
    state = bot_engine.BotState()
    real_get_env = bot_engine.get_env

    def _get_env(key, default=None, cast=None, **kwargs):
        overrides = {
            "AI_TRADING_EXEC_SYMBOLS_PER_ORDER": 1,
            "AI_TRADING_EXEC_SYMBOL_BUDGET_MIN": 1,
            "AI_TRADING_EXEC_SYMBOL_BUDGET_MAX": 1,
        }
        if key in overrides:
            value = overrides[key]
            if callable(cast):
                return cast(value)
            return value
        return real_get_env(key, default, cast=cast, **kwargs)

    monkeypatch.setattr(bot_engine, "get_env", _get_env)

    bar_ts = datetime.now(UTC)
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.0, 101.0],
            "volume": [1000, 1000],
        },
        index=pd.DatetimeIndex([bar_ts - timedelta(minutes=1), bar_ts], tz=UTC),
    )

    seen_batches: list[list[str]] = []

    def _batch(symbols, timeframe, start, end):
        seen_batches.append([str(symbol).upper() for symbol in symbols])
        return {str(symbol).upper(): df for symbol in symbols}

    monkeypatch.setattr("ai_trading.data.fetch.get_bars_batch", _batch)
    monkeypatch.setattr(bot_engine, "ensure_data_fetcher", lambda runtime: None)
    monkeypatch.setattr(bot_engine, "compute_current_positions", lambda runtime: {})
    monkeypatch.setattr(bot_engine, "submit_order", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot_engine, "_write_decision_record", lambda record, path: None)

    bot_engine._run_netting_cycle(state, runtime, "loop", 0.0)

    assert seen_batches
    assert all(len(batch) <= 1 for batch in seen_batches)


def test_symbol_budget_rotation_for_held_positions():
    state = bot_engine.BotState()
    symbols = ["AAPL", "MSFT", "GOOG"]
    positions = {"AAPL": 10.0, "MSFT": 10.0, "GOOG": 10.0}

    first_selected, first_cursor_start, held_kept = bot_engine._select_symbols_with_budget_rotation(
        symbols,
        positions,
        symbol_budget=1,
        state=state,
    )
    second_selected, second_cursor_start, _ = bot_engine._select_symbols_with_budget_rotation(
        symbols,
        positions,
        symbol_budget=1,
        state=state,
    )

    assert held_kept == 3
    assert first_cursor_start == 0
    assert second_cursor_start == 1
    assert first_selected == ["AAPL"]
    assert second_selected == ["MSFT"]
