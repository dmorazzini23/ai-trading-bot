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


def test_gate_effectiveness_analytics_excludes_global_halt_noise_and_writes_attribution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    log_path = tmp_path / "gate_effectiveness.jsonl"
    summary_path = tmp_path / "gate_effectiveness_summary.json"
    monkeypatch.setenv("AI_TRADING_GATE_EFFECTIVENESS_ANALYTICS_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_GATE_EFFECTIVENESS_LOG_PATH", str(log_path))
    monkeypatch.setenv("AI_TRADING_GATE_EFFECTIVENESS_SUMMARY_PATH", str(summary_path))
    monkeypatch.setenv("AI_TRADING_GATE_EFFECTIVENESS_EXCLUDE_GLOBAL_HALTS", "1")
    monkeypatch.setenv("AI_TRADING_GATE_EFFECTIVENESS_INCLUDE_ATTRIBUTION", "1")

    bot_engine._update_gate_effectiveness_analytics(
        decision_gate_counts={"OK_TRADE": 1, "AUTH_HALT": 4},
        decision_records_total=5,
        accepted_decisions=1,
        decision_observations=[
            {
                "symbol": "ALL",
                "gates": ["AUTH_HALT"],
                "accepted": False,
                "regime": "UNKNOWN",
                "expected_net_edge_bps": 0.0,
                "edge_proxy_bps": 0.0,
            },
            {
                "symbol": "AAPL",
                "gates": ["OK_TRADE", "VOL_TARGET_SCALE"],
                "accepted": True,
                "regime": "THIN",
                "expected_net_edge_bps": 10.0,
                "edge_proxy_bps": 7.5,
            },
        ],
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["total_records"] == 1
    assert summary["total_accepted_records"] == 1
    assert summary["excluded_records_total"] == 1
    assert summary["gate_totals"]["OK_TRADE"] == 1
    assert "AUTH_HALT" not in summary["gate_totals"]
    assert summary["excluded_gate_totals"]["AUTH_HALT"] == 1
    assert summary["symbol_attribution"]["AAPL"]["count"] == 1.0
    assert summary["regime_attribution"]["THIN"]["accepted_records"] == 1.0
    assert summary["gate_attribution"]["OK_TRADE"]["edge_proxy_bps_sum"] == 7.5
