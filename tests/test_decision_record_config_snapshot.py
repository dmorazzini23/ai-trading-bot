from __future__ import annotations

from datetime import UTC, datetime

from ai_trading.config.management import get_trading_config
from ai_trading.core import bot_engine
from ai_trading.core.netting import DecisionRecord, NettedTarget, SleeveProposal


def test_decision_record_config_snapshot_included() -> None:
    cfg = get_trading_config()
    state = bot_engine.BotState()
    snapshot = bot_engine._decision_record_config_snapshot(
        cfg=cfg,
        state=state,
        allocation_weights={"day": 0.4, "swing": 0.35, "longshort": 0.25},
        learned_overrides={"per_symbol_cost_buffer_bps": {"AAPL": 2.0}},
        sleeve_configs={
            "day": {
                "entry_threshold": 0.3,
                "exit_threshold": 0.2,
                "flip_threshold": 0.5,
            }
        },
        liquidity_regime="THIN",
    )
    proposal = SleeveProposal(
        symbol="AAPL",
        sleeve="day",
        bar_ts=datetime.now(UTC),
        target_dollars=1000.0,
        expected_edge_bps=20.0,
        expected_cost_bps=5.0,
        score=0.7,
        confidence=0.8,
    )
    net_target = NettedTarget(
        symbol="AAPL",
        bar_ts=datetime.now(UTC),
        target_dollars=1000.0,
        target_shares=10.0,
        proposals=[proposal],
    )
    record = DecisionRecord(
        symbol="AAPL",
        bar_ts=datetime.now(UTC),
        sleeves=[proposal],
        net_target=net_target,
        gates=["OK_TRADE"],
        config_snapshot=snapshot,
    )
    payload = record.to_dict()
    assert "config_snapshot" in payload
    assert payload["config_snapshot"]["allocation_weights"]["day"] == 0.4
    assert payload["config_snapshot"]["learned_overrides"]["per_symbol_cost_buffer_bps"]["AAPL"] == 2.0
    assert payload["config_snapshot"]["sleeve_configs"]["day"]["entry_threshold"] == 0.3
    assert payload["config_snapshot"]["liquidity_participation"]["max_participation_pct"] > 0
