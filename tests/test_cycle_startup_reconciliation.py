"""Exercise the cycle's startup boundary without a broker or worker side effect."""
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest


@pytest.fixture
def cycle(monkeypatch):
    from ai_trading import main, alpaca_api, config
    from ai_trading.config import management
    from ai_trading.core import bot_engine, runtime as runtime_module
    from ai_trading.data import fetch

    cfg = SimpleNamespace(rth_only=False, allow_extended=True,
                          skip_compute_when_provider_disabled=False,
                          order_stale_cleanup_interval=120)
    state = SimpleNamespace(ctx=None)
    runtime = SimpleNamespace(api=object(), params=dict(runtime_module.REQUIRED_PARAM_DEFAULTS), state={})
    calls = {"worker": [], "cancel": [], "pending": []}
    monkeypatch.setenv("EXECUTION_MODE", "paper")
    monkeypatch.setenv("AI_TRADING_WARMUP_MODE", "0")
    monkeypatch.setattr(main, "should_stop", lambda: False)
    monkeypatch.setattr(main, "alpaca_credential_status", lambda: (True, True))
    monkeypatch.setattr(main, "get_trading_config", lambda: cfg)
    monkeypatch.setattr(management.TradingConfig, "from_env", lambda **kw: cfg)
    monkeypatch.setattr(config, "get_settings", lambda: SimpleNamespace(max_position_size=0.1))
    monkeypatch.setattr(alpaca_api, "alpaca_get", lambda *a, **kw: {})
    monkeypatch.setattr(alpaca_api, "is_alpaca_service_available", lambda: True)
    monkeypatch.setattr(alpaca_api, "get_alpaca_service_status", lambda: {})
    monkeypatch.setattr(alpaca_api, "_set_alpaca_service_available", lambda enabled: None)
    monkeypatch.setattr(fetch, "is_primary_provider_enabled", lambda: True)
    monkeypatch.setattr(bot_engine, "get_trade_logger", lambda: None)
    monkeypatch.setattr(bot_engine, "_failsoft_mode_active", lambda: False)
    monkeypatch.setattr(bot_engine, "get_ctx", lambda: "context")
    monkeypatch.setattr(bot_engine, "ensure_alpaca_attached", lambda rt: None)
    monkeypatch.setattr(bot_engine, "list_open_orders", lambda api: calls["pending"])
    monkeypatch.setattr(bot_engine, "get_confirmed_pending_orders", lambda *a, **kw: calls["pending"])
    monkeypatch.setattr(bot_engine, "cancel_all_open_orders", lambda rt: calls["cancel"].append(rt))
    monkeypatch.setattr(bot_engine, "run_all_trades_worker", lambda s, rt: calls["worker"].append((s, rt)))
    monkeypatch.setattr(runtime_module, "enhance_runtime_with_context", lambda rt, ctx: rt)
    monkeypatch.setattr(main, "_resolve_cached_context", lambda *a: (state, runtime, False))
    monkeypatch.setattr(main, "_STARTUP_PENDING_RECONCILED", False)
    return main, state, runtime, calls


@pytest.mark.parametrize("status,age,cancel", [
    ("new", 121, True), ("pending_new", 60, False),
    ("new", None, True), ("filled", 300, False),
])
def test_startup_only_cancels_stale_or_undated_pending_orders(cycle, status, age, cancel):
    main, state, runtime, calls = cycle
    calls["pending"] = [SimpleNamespace(id="pending", status=status,
        submitted_at=datetime.now(UTC) - timedelta(seconds=age) if age is not None else None)]
    main.run_cycle()
    assert bool(calls["cancel"]) is cancel
    assert calls["worker"] == [(state, runtime)]
    assert main._STARTUP_PENDING_RECONCILED is True
    assert runtime.state["startup_pending_reconciled"] is True
    assert state.ctx == "context"
    main.run_cycle()
    assert len(calls["cancel"]) == int(cancel)


def test_missing_broker_never_runs_worker_and_keeps_reconciliation_pending(cycle):
    main, _, runtime, calls = cycle
    runtime.api = None
    main.run_cycle()
    assert not calls["worker"]
    assert not main._STARTUP_PENDING_RECONCILED
    assert runtime.state["startup_pending_reconciled"] is False
    assert runtime.state["service_phase"] == "reconcile"


def test_pending_order_read_failure_blocks_worker_and_retries_next_cycle(cycle, monkeypatch):
    from ai_trading.core import bot_engine
    main, _, runtime, calls = cycle

    def fail_read(*args, **kwargs):
        raise TimeoutError("broker order read timed out")

    monkeypatch.setattr(bot_engine, "get_confirmed_pending_orders", fail_read)
    main.run_cycle()
    assert not calls["worker"]
    assert not main._STARTUP_PENDING_RECONCILED
    assert runtime.state["service_phase"] == "reconcile"
    monkeypatch.setattr(bot_engine, "get_confirmed_pending_orders", lambda *a, **kw: [])
    main.run_cycle()
    assert len(calls["worker"]) == 1
    assert main._STARTUP_PENDING_RECONCILED


def test_cancel_failure_is_reported_without_retrying_every_cycle(cycle, monkeypatch, caplog):
    from ai_trading.core import bot_engine
    main, _, _, calls = cycle
    calls["pending"] = [SimpleNamespace(id="pending", status="new")]
    def fail_cancel(rt):
        raise TimeoutError("broker cancel timeout")
    monkeypatch.setattr(bot_engine, "cancel_all_open_orders", fail_cancel)
    main.run_cycle()
    assert "PENDING_ORDERS_STARTUP_CLEANUP_FAILED" in caplog.text
    assert main._STARTUP_PENDING_RECONCILED
    assert len(calls["worker"]) == 1


@pytest.mark.parametrize("mode,credentials", [("disabled", (True, True)), ("paper", (False, True))])
def test_unavailable_authority_blocks_worker_before_context(cycle, monkeypatch, mode, credentials):
    main, _, _, calls = cycle
    monkeypatch.setenv("EXECUTION_MODE", mode)
    monkeypatch.setattr(main, "alpaca_credential_status", lambda: credentials)
    main.run_cycle()
    assert not calls["worker"]
    assert not main._STARTUP_PENDING_RECONCILED
