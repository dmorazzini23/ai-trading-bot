"""Lineage-specific lot matching must retain account-level net quantity."""
import pytest

from ai_trading.tools import runtime_performance_report as rpt


def test_flat_broker_cannot_erase_unmatched_fill_inventory():
    positions, source = rpt._choose_reconciliation_positions(
        trade_positions={}, fill_positions={"AAPL": 3}, broker_positions={}, broker_available=True)
    assert positions == {"AAPL": 3}
    assert source == "fill_events"


@pytest.mark.parametrize("second_side,expected", [("buy", 4), ("sell", 0)])
def test_open_positions_sum_all_lineage_books(second_side, expected):
    rows = []
    for lineage, side in [("first", "buy"), ("second", second_side)]:
        for i, (direction, qty) in enumerate([(side, 3), ("sell" if side == "buy" else "buy", 1)]):
            rows.append(dict(symbol="AAPL", side=direction, qty=qty, price=100,
                             timestamp=f"2026-09-18T14:0{i}:00Z", correlation_id=lineage))
    events = rpt._extract_fill_events(rows)
    assert len(events) == 4
    closed, positions, lots = rpt._reconstruct_closed_trades(events)
    assert len(closed) == 2
    assert lots == 2
    assert positions == ({"AAPL": expected} if expected else {})
