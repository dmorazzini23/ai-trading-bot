from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from ai_trading.config.runtime import TradingConfig
from ai_trading.core import bot_engine


def test_reconcile_halts(monkeypatch):
    cfg = TradingConfig.from_env(allow_missing_drawdown=True)
    cfg.update(recon_enabled=True)
    runtime = SimpleNamespace(cfg=cfg, api=None)
    state = bot_engine.BotState()
    state.position_cache = {"AAPL": 10}

    monkeypatch.setattr(
        "ai_trading.oms.reconcile.fetch_broker_positions",
        lambda api: {"AAPL": 0.0},
    )

    ok = bot_engine._run_reconciliation_if_due(state, runtime, cfg, datetime.now(UTC))
    assert ok is False
    assert state.recon_halt is True


def test_reconcile_exception_latches_halt_until_success(monkeypatch):
    cfg = TradingConfig.from_env(allow_missing_drawdown=True)
    cfg.update(recon_enabled=True, recon_interval_seconds=300)
    runtime = SimpleNamespace(cfg=cfg, api=None)
    state = bot_engine.BotState()
    state.position_cache = {"AAPL": 10}

    def _raise_error(_api):
        raise RuntimeError("broker unavailable")

    monkeypatch.setattr("ai_trading.oms.reconcile.fetch_broker_positions", _raise_error)
    first_now = datetime.now(UTC)
    ok_first = bot_engine._run_reconciliation_if_due(state, runtime, cfg, first_now)
    assert ok_first is False
    assert state.recon_halt is True
    assert str(state.halt_reason or "").strip()

    monkeypatch.setattr(
        "ai_trading.oms.reconcile.fetch_broker_positions",
        lambda _api: {"AAPL": 10.0},
    )
    ok_second = bot_engine._run_reconciliation_if_due(
        state,
        runtime,
        cfg,
        first_now + timedelta(seconds=1),
    )
    assert ok_second is False
    assert state.recon_halt is True

    ok_third = bot_engine._run_reconciliation_if_due(
        state,
        runtime,
        cfg,
        first_now + timedelta(seconds=301),
    )
    assert ok_third is True
    assert state.recon_halt is False
