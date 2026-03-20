from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ai_trading.tools.compact_tca_pending import main


def test_compact_tca_pending_tool_compacts_matches(monkeypatch, tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    tca_path = runtime_dir / "tca_records.jsonl"
    fill_events_path = runtime_dir / "fill_events.jsonl"
    ts = datetime(2026, 3, 17, 10, 0, tzinfo=UTC).isoformat()

    pending = {
        "ts": ts,
        "client_order_id": "cid-tool-1",
        "order_id": "oid-tool-1",
        "symbol": "AAPL",
        "side": "buy",
        "qty": 10.0,
        "pending_event": True,
    }
    resolved = {
        "ts": ts,
        "client_order_id": "cid-tool-1",
        "order_id": "oid-tool-1",
        "symbol": "AAPL",
        "side": "buy",
        "pending_event": False,
        "status": "filled",
        "fill_price": 101.0,
        "fill_vwap": 101.0,
    }
    tca_path.write_text(json.dumps(pending) + "\n" + json.dumps(resolved) + "\n", encoding="utf-8")
    fill_events_path.write_text("", encoding="utf-8")

    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AI_TRADING_TCA_PATH", "runtime/tca_records.jsonl")
    monkeypatch.setenv("AI_TRADING_FILL_EVENTS_PATH", "runtime/fill_events.jsonl")
    monkeypatch.setattr(
        "sys.argv",
        ["compact_tca_pending"],
    )

    exit_code = main()

    assert exit_code == 0
    rows = [
        json.loads(line)
        for line in tca_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    compacted = [row for row in rows if row.get("pending_compacted") is True]
    assert compacted
    assert compacted[0]["pending_event"] is False
