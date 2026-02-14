from datetime import UTC, datetime

from ai_trading.core.netting import SleeveConfig, compute_sleeve_proposal


def test_cost_gate_blocks_trade():
    cfg = SleeveConfig(
        name="day",
        timeframe="5Min",
        enabled=True,
        entry_threshold=0.1,
        exit_threshold=0.05,
        flip_threshold=0.2,
        reentry_threshold=0.6,
        deadband_dollars=10.0,
        deadband_shares=1.0,
        turnover_cap_dollars=0.0,
        cost_k=2.0,
        edge_scale_bps=1.0,
        max_symbol_dollars=1000.0,
        max_gross_dollars=5000.0,
    )
    proposal = compute_sleeve_proposal(
        cfg=cfg,
        symbol="AAPL",
        bar_ts=datetime.now(UTC),
        score=0.2,
        confidence=0.5,
        current_pos=0.0,
        price=100.0,
        spread=1.0,
        vol=0.05,
        volume=1000.0,
    )
    assert proposal.blocked
    assert proposal.reason_code == "COST_GATE"
