from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd

from ai_trading.config.runtime import TradingConfig
from ai_trading.core import bot_engine
from ai_trading.core.netting import NettedTarget


def test_netting_cycle_emits_decision_reject_reason_summary(monkeypatch, caplog) -> None:
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

    now = datetime.now(UTC)
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.0, 101.0],
            "volume": [1000, 1000],
        },
        index=pd.DatetimeIndex([now - timedelta(minutes=1), now], tz=UTC),
    )

    def _force_target(symbol, bar_ts, proposals, disagree_ratio):
        return NettedTarget(
            symbol=str(symbol),
            bar_ts=bar_ts,
            target_dollars=1000.0,
            target_shares=0.0,
            proposals=list(proposals),
        )

    monkeypatch.setenv("AI_TRADING_POLICY_STRICT_CONFIG_GOVERNANCE", "0")
    monkeypatch.setenv("AI_TRADING_KILL_SWITCH", "1")
    monkeypatch.setenv("AI_TRADING_DECISION_REJECT_SUMMARY_LOG_TTL_SEC", "0")
    monkeypatch.setattr("ai_trading.data.fetch.get_bars_batch", lambda symbols, timeframe, start, end: {"AAPL": df})
    monkeypatch.setattr("ai_trading.core.netting.net_targets_for_symbol", _force_target)
    monkeypatch.setattr(bot_engine, "ensure_data_fetcher", lambda runtime: None)
    monkeypatch.setattr(bot_engine, "compute_current_positions", lambda runtime: {})
    monkeypatch.setattr(bot_engine, "check_daily_loss", lambda runtime, state: False)
    monkeypatch.setattr(bot_engine, "check_weekly_loss", lambda runtime, state: False)
    monkeypatch.setattr(bot_engine, "_write_decision_record", lambda record, path: None)

    with caplog.at_level(logging.INFO):
        bot_engine._run_netting_cycle(state, runtime, "loop", 0.0)

    assert any(
        record.getMessage() == "DECISION_REJECT_REASON_SUMMARY"
        for record in caplog.records
    )
