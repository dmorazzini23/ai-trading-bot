from types import SimpleNamespace

from ai_trading import main
import ai_trading.core.bot_engine as bot_engine
from ai_trading.utils.prof import SoftBudget
from ai_trading.utils import prof


def test_run_cycle_respects_budget(monkeypatch):
    """Exercise a prepared cycle with isolated I/O and one deterministic clock."""
    import ai_trading.alpaca_api as api
    import ai_trading.core.runtime as runtime_module
    from ai_trading.data import fetch

    clock = [0]
    monkeypatch.setattr(prof, "_monotonic_ns", lambda: clock[0])
    monkeypatch.setenv("EXECUTION_MODE", "sim")
    monkeypatch.setenv("AI_TRADING_EXECUTION_MODE", "sim")
    monkeypatch.setenv("AI_TRADING_WARMUP_MODE", "0")
    monkeypatch.setattr(main, "should_stop", lambda: False)
    monkeypatch.setattr(main, "_is_market_open_base", lambda: True)
    monkeypatch.setattr(main, "_STARTUP_PENDING_RECONCILED", True)
    monkeypatch.setattr(main, "_set_execution_phase", lambda *a, **k: None)
    monkeypatch.setattr(api, "alpaca_get", lambda *a, **k: {})
    monkeypatch.setattr(api, "get_alpaca_service_status", lambda: {})
    monkeypatch.setattr(api, "is_alpaca_service_available", lambda: True)
    monkeypatch.setattr(api, "_set_alpaca_service_available", lambda value: None)
    monkeypatch.setattr(fetch, "is_primary_provider_enabled", lambda: True)
    monkeypatch.setattr(bot_engine, "get_trade_logger", lambda: None)
    monkeypatch.setattr(bot_engine, "get_ctx", lambda: SimpleNamespace())
    monkeypatch.setattr(bot_engine, "_failsoft_mode_active", lambda: False)
    runtime = SimpleNamespace(params=dict(runtime_module.REQUIRED_PARAM_DEFAULTS), state={})
    state = SimpleNamespace()
    monkeypatch.setattr(main, "_resolve_cached_context", lambda *a: (state, runtime, False))
    monkeypatch.setattr(runtime_module, "enhance_runtime_with_context", lambda r, c: r)
    calls = []

    def worker(actual_state, actual_runtime):
        calls.append((actual_state, actual_runtime))
        clock[0] += 250_000_000

    monkeypatch.setattr(bot_engine, "run_all_trades_worker", worker)
    budget = SoftBudget(500)
    main.run_cycle()
    assert calls == [(state, runtime)]
    assert budget.elapsed_ms() == 250
    assert not budget.over_budget()
    clock[0] += 250_000_000
    assert budget.over_budget()
