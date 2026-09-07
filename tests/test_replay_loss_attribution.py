from __future__ import annotations

import pytest

from ai_trading.core.bot_engine import _replay_summary_metrics
from ai_trading.replay.loss_attribution import build_loss_attribution


def _result(qty: float, *, include_loss: bool, price: float = 100.0) -> dict:
    orders = [{"id": "one", "client_order_id": "one", "symbol": "WIN", "side": "buy", "qty": qty, "limit_price": 100.0}]
    events = [{"order_id": "one", "event_type": "fill", "symbol": "WIN", "side": "buy", "fill_qty": qty, "fill_price": price, "ts": "2026-01-05T15:00:00Z"}]
    if include_loss:
        orders.append({**orders[0], "id": "two", "client_order_id": "two", "symbol": "LOSS"})
        events.append({**events[0], "order_id": "two", "symbol": "LOSS"})
    return {"orders": orders, "events": events, "cap_adjustments": []}


def _report(base: dict, cand: dict) -> dict:
    rows = [{"symbol": symbol, "ts": "2026-01-05T15:05:00Z", "close": price} for symbol, price in (("WIN", 101.0), ("LOSS", 98.0))]
    return build_loss_attribution(
        baseline=base, candidate=cand,
        baseline_summary=_replay_summary_metrics(base, market_rows=rows),
        candidate_summary=_replay_summary_metrics(cand, market_rows=rows),
    )


def test_cap_selection_and_execution_drag_reconcile_governed_metric() -> None:
    report = _report(_result(10, include_loss=True), _result(5, include_loss=False, price=100.1))
    assert report["reconciled"] is True
    assert report["residual_bps"] == pytest.approx(0)
    parts = report["components"]
    assert parts["sizing_caps"]["contribution_bps"] == pytest.approx(150)
    assert parts["fills"]["contribution_bps"] == pytest.approx(0)
    assert parts["costs"]["contribution_bps"] < 0
    assert parts["exits"]["status"] == "not_identifiable"
    assert sum(report["symbol_contributions_bps"].values()) == pytest.approx(report["candidate_minus_baseline_bps"])


def test_quantity_reduction_has_dollar_effect_without_changing_unweighted_edge() -> None:
    report = _report(_result(10, include_loss=False), _result(5, include_loss=False))
    assert report["candidate_minus_baseline_bps"] == pytest.approx(0)
    weighted = report["quantity_weighted_markout"]
    assert weighted["direct_size_effect_holding_baseline_fills_dollars"] == pytest.approx(-5)
    assert weighted["remaining_fill_and_price_effect_dollars"] == pytest.approx(0)
    assert weighted["is_realized_pnl"] is False


def test_empty_fills_are_insufficient_evidence_not_success() -> None:
    report = _report({"orders": [], "events": []}, {"orders": [], "events": []})
    assert report["status"] == "insufficient_markout_evidence"
    assert report["quantity_weighted_markout"]["candidate_bps"] is None


def test_fill_participation_change_is_separate_from_order_caps() -> None:
    base = _result(10, include_loss=True)
    cand = _result(10, include_loss=True)
    cand["events"] = cand["events"][:1]
    report = _report(base, cand)
    assert report["components"]["sizing_caps"]["contribution_bps"] == pytest.approx(0)
    assert report["components"]["fills"]["contribution_bps"] == pytest.approx(150)
    assert report["reconciled"] is True
