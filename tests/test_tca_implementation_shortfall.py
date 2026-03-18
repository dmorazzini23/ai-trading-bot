from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ai_trading.analytics.tca import (
    ExecutionBenchmark,
    FillSummary,
    build_tca_record,
    implementation_shortfall_bps,
    reconcile_pending_tca_with_fill,
    resolve_pending_tca_from_fill,
)


def test_implementation_shortfall_buy_direction() -> None:
    # Buy worse fill than decision price should be positive cost bps
    value = implementation_shortfall_bps("buy", 100.0, 101.0, fees=0.0, qty=10)
    assert round(value, 6) == 100.0


def test_implementation_shortfall_sell_direction() -> None:
    # Sell worse fill than decision price should also be positive cost bps
    value = implementation_shortfall_bps("sell", 100.0, 99.0, fees=0.0, qty=10)
    assert round(value, 6) == 100.0


def test_tca_record_includes_canonical_price_fields() -> None:
    ts = datetime(2025, 1, 1, 14, 30, tzinfo=UTC)
    record = build_tca_record(
        client_order_id="cid-1",
        symbol="AAPL",
        side="buy",
        benchmark=ExecutionBenchmark(
            arrival_price=100.0,
            mid_at_arrival=100.2,
            decision_ts=ts,
            submit_ts=ts,
            first_fill_ts=ts,
        ),
        fill=FillSummary(fill_vwap=100.5, total_qty=10, fees=0.0, status="filled"),
    )
    assert record["decision_price"] == 100.0
    assert record["submit_price_reference"] == 100.2
    assert record["fill_price"] == 100.5


def test_resolve_pending_tca_from_fill_updates_pending_fields() -> None:
    submit_ts = datetime(2026, 3, 13, 15, 0, tzinfo=UTC)
    fill_ts = datetime(2026, 3, 13, 15, 0, 2, tzinfo=UTC)
    pending = {
        "client_order_id": "cid-1",
        "symbol": "AAPL",
        "side": "buy",
        "decision_price": 100.0,
        "qty": 10.0,
        "fees": 0.0,
        "pending_event": True,
        "benchmark": {"mid_at_arrival": 100.0, "submit_ts": submit_ts.isoformat()},
    }

    resolved = resolve_pending_tca_from_fill(
        pending_record=pending,
        fill_price=100.5,
        fill_qty=10.0,
        status="filled",
        fill_ts=fill_ts,
        source="unit_test",
    )

    assert resolved["pending_event"] is False
    assert resolved["pending_resolved"] is True
    assert resolved["fill_price"] == 100.5
    assert resolved["fill_vwap"] == 100.5
    assert resolved["resolved_fill_price"] == 100.5
    assert resolved["fill_latency_ms"] == 2000
    assert resolved["benchmark"]["first_fill_ts"] == fill_ts.isoformat()
    assert resolved["pending_resolved_source"] == "unit_test"


def test_reconcile_pending_tca_with_fill_appends_resolved_row(tmp_path: Path) -> None:
    path = tmp_path / "tca_records.jsonl"
    submit_ts = datetime(2026, 3, 13, 15, 0, tzinfo=UTC)
    pending = {
        "ts": submit_ts.isoformat(),
        "client_order_id": "cid-2",
        "order_id": "oid-2",
        "symbol": "MSFT",
        "side": "sell",
        "decision_price": 200.0,
        "qty": 5.0,
        "pending_event": True,
        "benchmark": {"mid_at_arrival": 200.0, "submit_ts": submit_ts.isoformat()},
    }
    path.write_text(json.dumps(pending) + "\n", encoding="utf-8")

    reconciled, reason = reconcile_pending_tca_with_fill(
        str(path),
        client_order_id="cid-2",
        order_id="oid-2",
        fill_price=199.0,
        fill_qty=5.0,
        status="filled",
        fill_ts=datetime(2026, 3, 13, 15, 0, 3, tzinfo=UTC),
        source="unit_test",
    )

    assert reconciled is True
    assert reason == "reconciled"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2
    resolved = rows[-1]
    assert resolved["pending_event"] is False
    assert resolved["status"] == "filled"
    assert resolved["resolved_fill_price"] == 199.0
    assert resolved["client_order_id"] == "cid-2"


def test_reconcile_pending_tca_with_fill_ignores_already_resolved(tmp_path: Path) -> None:
    path = tmp_path / "tca_records.jsonl"
    submit_ts = datetime(2026, 3, 13, 15, 0, tzinfo=UTC)
    pending = {
        "ts": submit_ts.isoformat(),
        "client_order_id": "cid-3",
        "order_id": "oid-3",
        "symbol": "NVDA",
        "side": "buy",
        "decision_price": 300.0,
        "qty": 1.0,
        "pending_event": True,
        "benchmark": {"mid_at_arrival": 300.0, "submit_ts": submit_ts.isoformat()},
    }
    resolved = {
        **pending,
        "pending_event": False,
        "status": "filled",
        "fill_price": 301.0,
        "fill_vwap": 301.0,
    }
    path.write_text(
        json.dumps(pending) + "\n" + json.dumps(resolved) + "\n",
        encoding="utf-8",
    )

    reconciled, reason = reconcile_pending_tca_with_fill(
        str(path),
        client_order_id="cid-3",
        order_id="oid-3",
        fill_price=302.0,
        fill_qty=1.0,
        status="filled",
    )

    assert reconciled is False
    assert reason == "already_resolved"
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2
