import threading
import types

from ai_trading.core import bot_engine


def test_run_all_trades_overlap(monkeypatch, caplog):
    from ai_trading.core import run_all_trades_worker as worker

    entered = threading.Event()
    release = threading.Event()
    errors = []
    calls = []
    state = bot_engine.BotState()
    runtime = types.SimpleNamespace(risk_engine=object())
    caplog.set_level("INFO")

    monkeypatch.setattr(bot_engine, "run_lock", threading.Lock())
    monkeypatch.setattr(
        worker, "prepare_run_all_trades_cycle",
        lambda *args, **kwargs: types.SimpleNamespace(
            ready=True, cfg_runtime=object(), loop_start=0.0,
            api=object(), previous_last_run_at=None,
        ),
    )
    monkeypatch.setattr(worker, "_finalize_run_all_trades_cycle", lambda **kwargs: None)

    def execute(**kwargs):
        calls.append(kwargs)
        entered.set()
        assert release.wait(timeout=10), "first cycle was not released"

    monkeypatch.setattr(worker, "execute_run_all_trades_cycle", execute)

    def run_first():
        try:
            worker.run_all_trades_worker_cycle(state, runtime)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_first)
    thread.start()
    try:
        assert entered.wait(timeout=5), "first cycle never acquired the lock"
        worker.run_all_trades_worker_cycle(state, runtime)
        assert len(calls) == 1
        assert any("RUN_ALL_TRADES_SKIPPED_OVERLAP" in r.message for r in caplog.records)
    finally:
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert not errors
    assert not bot_engine.run_lock.locked()


def test_run_all_trades_missing_get_account(monkeypatch, caplog):
    monkeypatch.delenv("AI_TRADING_MODEL_PATH", raising=False)
    monkeypatch.delenv("AI_TRADING_MODEL_MODULE", raising=False)
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setenv("AI_TRADING_NETTING_ENABLED", "0")
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
    runtime.cfg = types.SimpleNamespace(netting_enabled=False)
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
    monkeypatch.setattr(bot_engine, "_netting_pipeline_enabled", lambda runtime: False)
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

    assert any(
        ("HALT_SKIP_NEW_TRADES" in r.message) or ("CYCLE_GATES" in r.message)
        for r in caplog.records
    )
