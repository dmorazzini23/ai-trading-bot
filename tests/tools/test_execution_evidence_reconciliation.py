from __future__ import annotations

import pytest

from ai_trading.tools.execution_evidence_reconciliation import reconcile_evidence


def _fill(fill_id, side, qty, price, timestamp, fee=None):
    return {"fill_id": fill_id, "order_id": fill_id, "symbol": "AAPL", "side": side, "fill_qty": qty, "fill_price": price, "ts": timestamp, "fee_amount": fee, "fee_source": "broker_payload" if fee is not None else "missing"}


def test_estimated_or_unverified_fees_cannot_certify_net_profit():
    for source in ["configured_estimate", "legacy_unverified", "missing"]:
        buy = {**_fill("buy", "buy", 1, 100, "2026-01-02T15:00:00Z", 0), "fee_source": source}
        sell = _fill("sell", "sell", 1, 110, "2026-01-02T16:00:00Z", 0)
        report = reconcile_evidence(decisions=[], orders=[], fills=[buy, sell], tca=[])
        assert report["entry_exit_links"][0]["net_pnl"] is None


def test_invalid_fill_cannot_disappear_into_an_otherwise_linked_report():
    fill = _fill("valid", "buy", 1, 100, "2026-01-02T15:00:00Z", 0)
    report = reconcile_evidence(decisions=[fill], orders=[fill], tca=[fill], fills=[fill, {**fill, "fill_id": "invalid", "fill_qty": -1}])
    assert report["accepted_unique_fills"] == 1
    assert report["counts"]["invalid_fill_rows"] == 1
    assert report["status"] == "evidence_gaps"


def test_complete_paper_session_and_missing_boundary_fail_closed():
    from ai_trading.tools.execution_evidence_reconciliation import reconcile_session

    boundaries = [{"account_id": "test", "trading_mode": "paper", "positions_complete": True, "positions": {}, "timestamp": ts} for ts in ["2026-01-02T14:30:00Z", "2026-01-02T21:00:00Z"]]
    fills = [{**_fill(identity, side, 1, price, ts, 0), "account_id": "test", "trading_mode": "paper", "decision_id": identity} for identity, side, price, ts in [("buy", "buy", 100, "2026-01-02T15:00:00Z"), ("sell", "sell", 101, "2026-01-02T20:00:00Z")]]
    orders = [{"order_id": row["order_id"], "filled_qty": 1, "ts": row["ts"]} for row in fills]
    inputs = dict(session_date="2026-01-02", account_id="test", decisions=[{"decision_id": "buy"}, {"decision_id": "sell"}], orders=orders, fills=fills, tca=fills)
    report = reconcile_session(**inputs, boundaries=boundaries)
    assert report["status"] == "paper_session_reconciled"
    assert report["position_reconciliation"]["status"] == "matched"
    report = reconcile_session(**inputs, boundaries=boundaries[:1])
    assert "complete_session_boundary_pair_missing" in report["session_gaps"]
    assert report["status"] == "evidence_gaps"


def test_partial_fifo_exit_keeps_unknown_fees_and_open_inventory_explicit():
    buy = _fill("buy", "buy", 10, 100, "2026-01-02T15:00:00Z")
    sell = _fill("sell", "sell", 4, 110, "2026-01-02T16:00:00Z")
    report = reconcile_evidence(decisions=[{"order": {"broker_order_id": "buy"}}], orders=[{"order_id": "buy"}, {"order_id": "sell"}], fills=[buy, buy, sell], tca=[])
    assert report["accepted_unique_fills"] == 2
    assert report["counts"]["duplicate_fill_rows"] == 1
    assert report["counts"]["order_matched"] == 2
    assert report["entry_exit_links"][0]["gross_pnl"] == "40"
    assert report["entry_exit_links"][0]["net_pnl"] is None
    assert report["unclosed_observed_buy_quantity"]["AAPL"] == "6"
    assert report["position_reconciliation"]["status"] == "unavailable"


def test_conflicting_fill_ids_are_quarantined_and_short_history_not_fabricated():
    buy = _fill("buy", "buy", 10, 100, "2026-01-02T15:00:00Z", 0)
    sell = _fill("sell", "sell", 4, 110, "2026-01-02T16:00:00Z", 0)
    report = reconcile_evidence(decisions=[], orders=[], fills=[buy, {**buy, "fill_qty": 20}, sell], tca=[])
    assert report["conflicting_fill_ids"] == 1
    assert report["entry_exit_links"] == []
    assert report["unmatched_sell_quantity"]["AAPL"] == "4"


def test_known_fees_are_allocated_to_closed_quantity():
    buy = _fill("buy", "buy", 10, 100, "2026-01-02T15:00:00Z", 2)
    sell = _fill("sell", "sell", 5, 110, "2026-01-02T16:00:00Z", 1)
    report = reconcile_evidence(decisions=[], orders=[], fills=[buy, sell], tca=[])
    assert report["entry_exit_links"][0]["net_pnl"] == "48.0"


def test_position_snapshots_reconcile_only_their_time_window_and_account():
    opening = {"account_id": "test", "trading_mode": "paper", "timestamp": "2026-01-02T15:00:00Z", "positions": {"AAPL": "2"}}
    closing = {**opening, "timestamp": "2026-01-02T17:00:00Z", "positions": {"AAPL": "7"}}
    fills = [_fill("prior", "buy", 2, 100, "2026-01-02T14:00:00Z", 0), _fill("current", "buy", 5, 100, "2026-01-02T16:00:00Z", 0)]
    report = reconcile_evidence(decisions=[], orders=[], fills=fills, tca=[], opening_positions=opening, closing_positions=closing)
    assert report["position_reconciliation"]["status"] == "matched"
    assert report["position_reconciliation"]["included_fills"] == 1
    report = reconcile_evidence(decisions=[], orders=[], fills=fills, tca=[], opening_positions=opening, closing_positions={**closing, "positions": {"AAPL": "6"}})
    assert report["position_reconciliation"]["differences"] == {"AAPL": "1"}
    with pytest.raises(ValueError, match="same account"):
        reconcile_evidence(decisions=[], orders=[], fills=fills, tca=[], opening_positions=opening, closing_positions={**closing, "account_id": "different"})


def test_order_cumulative_quantity_mismatch_and_stale_snapshot_are_distinct():
    fills = [_fill("a", "buy", 5, 100, "2026-01-02T16:00:00Z", 0), _fill("b", "buy", 4, 100, "2026-01-02T16:00:00Z", 0)]
    orders = [{"order_id": "a", "filled_qty": 6, "ts": "2026-01-02T16:01:00Z"}, {"order_id": "b", "filled_qty": 2, "ts": "2026-01-02T15:59:00Z"}]
    report = reconcile_evidence(decisions=[], orders=orders, fills=fills, tca=[])
    quantities = report["order_quantity_reconciliation"]
    assert quantities["counts"] == {"quantity_mismatch": 1, "order_snapshot_precedes_last_fill": 1}
    assert quantities["mismatches"][0]["observed_fill_qty"] == "5"
