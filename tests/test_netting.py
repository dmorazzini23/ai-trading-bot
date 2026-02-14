from datetime import UTC, datetime

from ai_trading.core.netting import SleeveProposal, NettedTarget, apply_global_caps, net_targets_for_symbol


def test_disagreement_damping_applied():
    bar_ts = datetime.now(UTC)
    proposals = [
        SleeveProposal(
            symbol="AAPL",
            sleeve="day",
            bar_ts=bar_ts,
            target_dollars=100.0,
            expected_edge_bps=10.0,
            expected_cost_bps=2.0,
            score=0.4,
            confidence=0.6,
        ),
        SleeveProposal(
            symbol="AAPL",
            sleeve="swing",
            bar_ts=bar_ts,
            target_dollars=-80.0,
            expected_edge_bps=8.0,
            expected_cost_bps=2.0,
            score=-0.3,
            confidence=0.5,
        ),
    ]
    net = net_targets_for_symbol("AAPL", bar_ts, proposals, disagree_threshold=0.35)
    assert "DISAGREEMENT_DAMPING" in net.reasons
    assert net.target_dollars != 20.0


def test_apply_global_caps():
    bar_ts = datetime.now(UTC)
    targets = {
        "AAPL": NettedTarget(symbol="AAPL", bar_ts=bar_ts, target_dollars=50000.0, target_shares=0.0),
        "MSFT": NettedTarget(symbol="MSFT", bar_ts=bar_ts, target_dollars=20000.0, target_shares=0.0),
    }
    reasons = apply_global_caps(targets, max_symbol_dollars=25000.0, max_gross_dollars=60000.0, max_net_dollars=40000.0)
    assert any("RISK_CAP_SYMBOL" in r for r in reasons)
    assert abs(targets["AAPL"].target_dollars) <= 25000.0
