from datetime import UTC, datetime
from types import SimpleNamespace

from ai_trading.config.runtime import TradingConfig
from ai_trading.core import bot_engine


def test_reconcile_halts(monkeypatch):
    cfg = TradingConfig.from_env(allow_missing_drawdown=True)
    cfg.update(recon_enabled=True)
    runtime = SimpleNamespace(cfg=cfg, api=None)
    state = bot_engine.BotState()
    state.position_cache = {"AAPL": 10.0}

    monkeypatch.setattr(
        "ai_trading.oms.reconcile.fetch_broker_positions",
        lambda api: {"AAPL": 0.0},
    )

    ok = bot_engine._run_reconciliation_if_due(state, runtime, cfg, datetime.now(UTC))
    assert ok is False
    assert state.recon_halt is True
