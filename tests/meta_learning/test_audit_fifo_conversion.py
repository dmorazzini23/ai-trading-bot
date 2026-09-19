"""Audit fills must form quantity-conserving, directional FIFO observations."""
import pandas as pd
import pytest

from ai_trading.meta_learning import core


@pytest.mark.parametrize("fills,quantities,rewards,open_qty", [
    ([("buy", 2, 100), ("sell", 2, 102)], [2], [4], 0),
    ([("sell", 2, 102), ("buy", 2, 100)], [2], [4], 0),
    ([("buy", 3, 100), ("sell", 1, 102)], [1], [2], 2),
    ([("buy", 1, 100), ("buy", 2, 101), ("sell", 3, 102)], [1, 2], [2, 2], 0),
    ([("buy", 1, 100), ("sell", 3, 102), ("buy", 2, 101)], [1, 2], [2, 2], 0),
])
def test_audit_fifo_conserves_quantity(monkeypatch, fills, quantities, rewards, open_qty):
    monkeypatch.setattr(core, "pd", pd)
    rows = [[f"00000000-0000-0000-0000-{i:012d}", f"2026-09-18T14:0{i}:00Z",
             "AAPL", side, qty, price, "paper", "filled"]
            for i, (side, qty, price) in enumerate(fills)]
    result = core._convert_audit_to_meta_format(pd.DataFrame(rows))
    closed = result[result.exit_price != ""]
    opened = result[result.exit_price == ""]
    assert closed.qty.tolist() == quantities
    assert closed.reward.tolist() == rewards
    assert opened.qty.sum() == open_qty
    assert all(closed.entry_time < closed.exit_time)


@pytest.mark.parametrize("side,qty,price", [("invalid", 1, 100), ("buy", -1, 100),
    ("buy", float("nan"), 100), ("sell", 1, float("inf")), ("buy", 1, 0)])
def test_invalid_audit_fill_cannot_create_inventory(monkeypatch, side, qty, price):
    monkeypatch.setattr(core, "pd", pd)
    row = ["00000000-0000-0000-0000-000000000001", "2026-09-18T14:00Z",
           "AAPL", side, qty, price, "paper", "filled"]
    assert core._convert_audit_to_meta_format(pd.DataFrame([row])).empty


@pytest.mark.parametrize("status", ["new", "accepted", "partially_filled", "pending_new", "unknown"])
def test_order_acknowledgment_does_not_establish_executed_quantity(monkeypatch, status):
    monkeypatch.setattr(core, "pd", pd)
    row = ["00000000-0000-0000-0000-000000000001", "2026-09-18T14:00Z",
           "AAPL", "buy", 10, 100, "paper", status]
    assert core._convert_audit_to_meta_format(pd.DataFrame([row])).empty
