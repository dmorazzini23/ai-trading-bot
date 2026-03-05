from __future__ import annotations

import json
from pathlib import Path

from ai_trading.core import bot_engine


def test_gate_effectiveness_analytics_writes_rollups(
    tmp_path: Path,
    monkeypatch,
) -> None:
    log_path = tmp_path / "gate_effectiveness.jsonl"
    summary_path = tmp_path / "gate_effectiveness_summary.json"
    monkeypatch.setenv("AI_TRADING_GATE_EFFECTIVENESS_ANALYTICS_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_GATE_EFFECTIVENESS_LOG_PATH", str(log_path))
    monkeypatch.setenv("AI_TRADING_GATE_EFFECTIVENESS_SUMMARY_PATH", str(summary_path))

    bot_engine._update_gate_effectiveness_analytics(
        decision_gate_counts={"OK_TRADE": 2, "AUTH_HALT": 3},
        decision_records_total=5,
        accepted_decisions=2,
    )

    assert log_path.exists()
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["total_records"] == 5
    assert summary["total_accepted_records"] == 2
    assert summary["gate_totals"]["AUTH_HALT"] == 3
