import json
from pathlib import Path
import sys

import pytest

from ai_trading.tools import paper_evidence_review as review


@pytest.mark.parametrize("recorded_at, account, decision_ts, expected", [
    ("2026-10-08T19:55:25Z", "test", "2026-10-08T19:55:07Z", 1),
    ("2026-10-08T19:55:35Z", "test", "2026-10-08T19:55:07Z", 0),
    ("2026-10-08T19:55:25Z", "other", "2026-10-08T19:55:07Z", 0),
    ("2026-10-08T19:55:25Z", "test", None, 0),
])
def test_daily_report_reads_causal_operational_decision_without_manufacturing_acceptance(tmp_path: Path, monkeypatch, recorded_at, account, decision_ts, expected):
    fill = {"ts": "2026-10-08T19:55:31Z", "fill_id": "fill", "order_id": "order",
            "client_order_id": "eod-2026-10-08-MSFT-sell", "decision_trace_id": "trace",
            "symbol": "MSFT", "side": "sell", "fill_qty": 1, "fill_price": 522.47,
            "account_id": "test", "trading_mode": "paper"}
    boundaries = [{"timestamp": ts, "account_id": "test", "trading_mode": "paper", "positions_complete": True,
                   "positions": {"MSFT": qty}} for ts, qty in [("2026-10-08T13:29:00Z", 1), ("2026-10-08T20:01:00Z", 0)]]
    rows = {"decision_records": [], "order_events": [{**fill, "filled_qty": 1}],
            "fill_events": [fill], "tca_records": [{**fill, "pending_event": False}],
            "broker_position_boundaries": boundaries}
    for name, data in rows.items():
        (tmp_path / (name+".jsonl")).write_text("".join(json.dumps(r)+"\n" for r in data))
    original = (tmp_path / "decision_records.jsonl").read_bytes()
    decision = {"decision_trace_id": "trace", "symbol": "MSFT", "side": "sell", "account_id": account,
                "decision_ts": decision_ts, "recorded_at": recorded_at, "final_accepted": False,
                "reported_gate_reasons": ["OPERATIONAL_EXIT_REQUESTED"]}
    funnel = {"status": "local_diagnostic_complete", "database": "read-only.db", "decisions": [decision], "gaps": []}
    monkeypatch.setattr(review, "session_diagnostic", lambda *_:funnel)
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["paper_evidence_review", "--runtime-dir", str(tmp_path),
                                    "--replay-report", str(tmp_path/"missing-replay.json"),
                                    "--session", "2026-10-08", "--output", str(output)])
    review.main()
    r = json.loads(output.read_text())
    assert r["session_audit"]["counts"].get("decision_matched", 0) == expected
    assert r["session_audit"]["counts"]["fees_missing"] == 1
    assert r["status"] == "evidence_pending" and r["promotion_authority"] is False and r["orders_sent"] == 0
    assert r["sources"]["oms_decisions"]["read_only"] is True
    assert decision["final_accepted"] is False
    assert (tmp_path / "decision_records.jsonl").read_bytes() == original
