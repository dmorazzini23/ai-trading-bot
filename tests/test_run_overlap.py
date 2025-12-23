import threading
import time
import types

from ai_trading.core import bot_engine


def test_run_all_trades_overlap(monkeypatch, caplog):
    monkeypatch.delenv("AI_TRADING_MODEL_PATH", raising=False)
    monkeypatch.delenv("AI_TRADING_MODEL_MODULE", raising=False)
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setattr(bot_engine, "_MODEL_CACHE", None, raising=False)
    monkeypatch.setattr(bot_engine, "_global_ctx", None, raising=False)
    monkeypatch.setattr(bot_engine, "_ctx", None, raising=False)
    monkeypatch.setattr(bot_engine, "ctx", None, raising=False)

    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "build_fetcher",
        lambda *_: types.SimpleNamespace(source="stub"),
    )

    state = bot_engine.BotState()
    runtime = bot_engine.get_ctx()
    caplog.set_level("INFO")

    monkeypatch.setattr(bot_engine, "is_market_open", lambda: True)
    monkeypatch.setattr(bot_engine, "check_pdt_rule", lambda ctx: False)
    monkeypatch.setattr(bot_engine, "_prepare_run", lambda ctx, st: (0.0, True, []))
    monkeypatch.setattr(bot_engine, "_process_symbols", lambda *a, **k: ([], {}, 0))
    monkeypatch.setattr(bot_engine, "_send_heartbeat", lambda: None)

    api_obj = runtime.api
    if api_obj is None:
        api_obj = types.SimpleNamespace()
        runtime.api = api_obj
    monkeypatch.setattr(
        api_obj,
        "get_account",
        lambda: types.SimpleNamespace(cash=0, equity=0),
        raising=False,
    )
    assert getattr(bot_engine._MODEL_CACHE, "is_placeholder_model", False)

    def slow_prepare(ctx, st):
        time.sleep(0.2)
        return (0.0, True, [])

    monkeypatch.setattr(bot_engine, "_prepare_run", slow_prepare)

    t = threading.Thread(target=bot_engine.run_all_trades_worker, args=(state, runtime))
    t.start()
    time.sleep(0.05)
    bot_engine.run_all_trades_worker(state, runtime)
    t.join()
    assert any("RUN_ALL_TRADES_SKIPPED_OVERLAP" in r.message for r in caplog.records)


def test_run_all_trades_missing_get_account(monkeypatch, caplog):
    monkeypatch.delenv("AI_TRADING_MODEL_PATH", raising=False)
    monkeypatch.delenv("AI_TRADING_MODEL_MODULE", raising=False)
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setattr(bot_engine, "_MODEL_CACHE", None, raising=False)
    monkeypatch.setattr(bot_engine, "_global_ctx", None, raising=False)
    monkeypatch.setattr(bot_engine, "_ctx", None, raising=False)
    monkeypatch.setattr(bot_engine, "ctx", None, raising=False)

    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "build_fetcher",
        lambda *_: types.SimpleNamespace(source="stub"),
    )

    runtime = bot_engine.get_ctx()
    runtime.api = types.SimpleNamespace(list_positions=lambda: [])
    assert not hasattr(runtime.api, "get_account")
    runtime.drawdown_circuit_breaker = types.SimpleNamespace(
        update_equity=lambda equity: True,
        get_status=lambda: {
            "current_drawdown": 0.0,
            "max_drawdown": 0.10,
            "trading_allowed": True,
            "peak_equity": 0.0,
        },
    )
    runtime.portfolio_weights = {}
    runtime.risk_engine = types.SimpleNamespace(
        wait_for_exposure_update=lambda timeout: None
    )

    caplog.set_level("INFO")

    monkeypatch.setattr(bot_engine, "ensure_alpaca_attached", lambda ctx: None)
    monkeypatch.setattr(bot_engine, "_validate_trading_api", lambda api: True)
    monkeypatch.setattr(bot_engine, "list_open_orders", lambda api: [])
    monkeypatch.setattr(bot_engine, "ensure_data_fetcher", lambda ctx: None)
    monkeypatch.setattr(bot_engine, "get_trade_logger", lambda: None)
    monkeypatch.setattr(bot_engine, "get_strategies", lambda: [])
    monkeypatch.setattr(bot_engine, "is_market_open", lambda: True)
    monkeypatch.setattr(bot_engine, "check_pdt_rule", lambda ctx: False)
    monkeypatch.setattr(bot_engine, "get_verbose_logging", lambda: False)
    monkeypatch.setattr(bot_engine, "utc_now_iso", lambda: "now")
    monkeypatch.setattr(bot_engine, "_ensure_execution_engine", lambda ctx: None)
    monkeypatch.setattr(bot_engine, "check_halt_flag", lambda ctx: True)
    monkeypatch.setattr(bot_engine, "manage_position_risk", lambda ctx, pos: None)
    monkeypatch.setattr(bot_engine, "_log_health_diagnostics", lambda *a, **k: None)
    monkeypatch.setattr(bot_engine, "_send_heartbeat", lambda: None)
    monkeypatch.setattr(bot_engine, "_prepare_run", lambda ctx, st, tickers=None: (0.0, True, []))

    state = bot_engine.BotState()

    bot_engine.run_all_trades_worker(state, runtime)

    assert any("HALT_SKIP_NEW_TRADES" in r.message for r in caplog.records)
