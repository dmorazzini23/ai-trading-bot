from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json

from ai_trading.data import reference_reconcile


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def test_reference_reconcile_writes_and_dedupes(tmp_path, monkeypatch):
    decisions_path = tmp_path / "dual_feed_decisions.jsonl"
    reconcile_path = tmp_path / "reference_reconciliation.jsonl"
    reliability_path = tmp_path / "feed_reliability_scores.json"
    decision_ts = datetime.now(UTC) - timedelta(minutes=30)
    decision = {
        "decision_id": "dec-1",
        "decision_ts": decision_ts.isoformat(),
        "symbol": "AAPL",
        "outcome": "submitted",
        "side": "buy",
        "quantity": 10,
        "execution_feed": "iex",
        "reference_feed": "delayed_sip",
        "execution_price": 101.0,
        "execution_bid": 100.9,
        "execution_ask": 101.1,
        "execution_volume": 4000,
        "context": {
            "signal_side": "buy",
            "signal_strength": 0.82,
            "signal_confidence": 0.64,
            "signal_weight": 0.9,
        },
    }
    _write_jsonl(decisions_path, [decision])

    monkeypatch.setenv("AI_TRADING_DUAL_FEED_DECISIONS_PATH", str(decisions_path))
    monkeypatch.setenv("AI_TRADING_REFERENCE_RECONCILIATION_PATH", str(reconcile_path))
    monkeypatch.setenv("AI_TRADING_FEED_RELIABILITY_PATH", str(reliability_path))
    monkeypatch.setenv("AI_TRADING_FEED_RELIABILITY_MIN_SAMPLES", "1")

    monkeypatch.setattr(
        reference_reconcile,
        "fetch_reference_minute_bar_snapshot",
        lambda *_args, **_kwargs: {
            "feed": "delayed_sip",
            "price": 100.0,
            "volume": 10000.0,
            "bar_ts": decision_ts.isoformat(),
        },
    )

    first = reference_reconcile.run_reference_reconciliation_once(max_rows=50, min_lag_minutes=10)
    assert first["processed"] == 1
    assert first["written"] == 1
    lines = reconcile_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["decision_id"] == "dec-1"
    assert payload["outcome"] == "submitted"
    assert payload["reference_price"] == 100.0
    assert payload["metrics"]["price_drift_bps"] is not None
    assert payload["signal_agreement"] is False
    assert payload["signal_disagreement"] is True
    assert payload["signal_metrics"]["signal_side"] == "buy"
    assert payload["signal_metrics"]["signal_weight"] == 0.9
    assert first["reliability_path"] == str(reliability_path)
    reliability_payload = json.loads(reliability_path.read_text(encoding="utf-8"))
    assert reliability_payload["lookback_rows"] == 1
    assert "AAPL" in reliability_payload["scores"]

    second = reference_reconcile.run_reference_reconciliation_once(max_rows=50, min_lag_minutes=10)
    assert second["processed"] == 0
    assert second["written"] == 0
