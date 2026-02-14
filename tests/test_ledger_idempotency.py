from datetime import UTC, datetime

from ai_trading.oms.ledger import LedgerEntry, OrderLedger, deterministic_client_order_id


def test_ledger_idempotency(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = OrderLedger(str(path), lookback_hours=24.0)
    client_order_id = deterministic_client_order_id(
        salt="seed",
        symbol="AAPL",
        bar_ts="2024-01-01T10:00:00Z",
        side="buy",
        qty=10,
        limit_price=100.0,
    )
    entry = LedgerEntry(
        client_order_id=client_order_id,
        symbol="AAPL",
        bar_ts="2024-01-01T10:00:00Z",
        qty=10,
        side="buy",
        limit_price=100.0,
        ts=datetime.now(UTC).isoformat(),
        broker_order_id="oid",
        status="submitted",
    )
    ledger.record(entry)
    reloaded = OrderLedger(str(path), lookback_hours=24.0)
    assert reloaded.seen_client_order_id(client_order_id)
