from __future__ import annotations

import logging
import types

from ai_trading.core import bot_engine as bot
from tests.bot_engine.test_data_source_retry_logging import (
    DummyAPI,
    DummyLock,
    DummyRiskEngine,
)


def test_retry_final_skipped_on_single_success(monkeypatch, caplog):
    state = bot.BotState()
    runtime = types.SimpleNamespace(
        risk_engine=DummyRiskEngine(),
        api=DummyAPI(),
        execution_engine=None,
        data_fetcher=types.SimpleNamespace(_minute_timestamps={}),
        model=object(),
        tickers=["AAA", "BBB"],
        portfolio_weights={},
    )

    setattr(bot.CFG, "log_market_fetch", False)
    setattr(bot.CFG, "shadow_mode", False)

    dummy_lock = DummyLock()
    monkeypatch.setattr(bot, "portfolio_lock", dummy_lock, raising=False)
    import ai_trading.utils as utils_mod

    monkeypatch.setattr(utils_mod, "portfolio_lock", dummy_lock, raising=False)
    monkeypatch.setattr(bot.portfolio, "compute_portfolio_weights", lambda runtime, symbols: {})

    monkeypatch.setattr(bot, "_ALPACA_IMPORT_ERROR", None, raising=False)
    monkeypatch.setattr(bot, "_ensure_alpaca_classes", lambda: None)
    monkeypatch.setattr(bot, "_init_metrics", lambda: None)
    monkeypatch.setattr(bot, "_ensure_execution_engine", lambda runtime: None)
    monkeypatch.setattr(bot, "ensure_alpaca_attached", lambda runtime: None)
    monkeypatch.setattr(bot, "_validate_trading_api", lambda api: True)
    monkeypatch.setattr(bot, "check_pdt_rule", lambda runtime: False)
    monkeypatch.setattr(bot, "get_trade_cooldown_min", lambda: 0)
    monkeypatch.setattr(bot, "is_market_open", lambda: True)
    monkeypatch.setattr(bot, "get_verbose_logging", lambda: False)
    monkeypatch.setattr(bot, "list_open_orders", lambda api: [])
    monkeypatch.setattr(bot, "ensure_data_fetcher", lambda runtime: None)
    monkeypatch.setattr(bot, "get_trade_logger", lambda: None)
    monkeypatch.setattr(bot, "_get_runtime_context_or_none", lambda: None)
    monkeypatch.setattr(
        bot,
        "_prepare_run",
        lambda runtime, state, tickers: (1000.0, True, ["AAA", "BBB"]),
    )
    monkeypatch.setattr(bot, "run_multi_strategy", lambda runtime: None)
    monkeypatch.setattr(bot, "_send_heartbeat", lambda: None)
    monkeypatch.setattr(bot, "_log_loop_heartbeat", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot, "_check_runtime_stops", lambda runtime: None)
    monkeypatch.setattr(bot, "check_halt_flag", lambda runtime: False)
    monkeypatch.setattr(bot, "manage_position_risk", lambda runtime, pos: None)
    monkeypatch.setattr(bot.time, "sleep", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot, "MEMORY_OPTIMIZATION_AVAILABLE", False, raising=False)
    monkeypatch.setattr(bot, "get_strategies", lambda: [])
    monkeypatch.setattr(
        bot,
        "_process_symbols",
        lambda symbols, current_cash, alpha_model, regime_ok: (
            list(symbols),
            {symbol: 5 for symbol in symbols},
            len(list(symbols)),
        ),
    )

    caplog.set_level(logging.INFO)

    bot.run_all_trades_worker(state, runtime)

    assert all(record.getMessage() != "DATA_SOURCE_RETRY_FINAL" for record in caplog.records)
