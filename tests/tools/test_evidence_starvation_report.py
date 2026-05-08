from __future__ import annotations

import json
from pathlib import Path

from ai_trading.tools import evidence_starvation_report as report_tool


def test_evidence_starvation_hard_safety_blocker_wins() -> None:
    payload = report_tool.build_evidence_starvation_report(
        report_date="2026-05-05",
        executable_symbols=["AAPL", "AMZN"],
        fills=[],
        runtime_gonogo={"gate_passed": False, "failed_checks": ["win_rate"]},
        sample_target=150,
    )

    assert payload["status"] == "blocked_by_safety"
    assert payload["recommendation"] == "stay_observe_due_to_hard_safety"
    assert payload["hard_safety_blockers"] == ["win_rate"]


def test_evidence_starvation_recommends_diagnostic_sampling_when_fill_rate_too_low() -> None:
    payload = report_tool.build_evidence_starvation_report(
        report_date="2026-05-05",
        executable_symbols=["AAPL", "AMZN"],
        fills=[{"ts": "2026-05-05T14:00:00Z", "symbol": "AAPL"}],
        runtime_gonogo={"gate_passed": True},
        sample_target=150,
        min_daily_fills=3,
    )

    assert payload["status"] == "starved"
    assert payload["recommendation"] == "widen_paper_diagnostic_sampling"


def test_evidence_starvation_flags_symbol_concentration() -> None:
    candidates = [
        {"ts": "2026-05-05T14:00:00Z", "symbol": "AAPL", "status": "submitted"}
        for _ in range(20)
    ]
    payload = report_tool.build_evidence_starvation_report(
        report_date="2026-05-05",
        executable_symbols=["AAPL", "AMZN"],
        candidates=candidates,
        fills=[{"ts": "2026-05-05T14:00:00Z", "symbol": "AAPL"} for _ in range(5)],
        runtime_gonogo={"gate_passed": True},
        sample_target=150,
        min_daily_fills=3,
    )

    assert payload["status"] == "symbol_starved"
    assert payload["symbols_without_candidates"] == ["AMZN"]
    assert payload["recommendation"] == "add_shadow_symbols"


def test_evidence_starvation_cli_writes_latest(tmp_path: Path) -> None:
    fills = tmp_path / "fills.jsonl"
    out = tmp_path / "starvation.json"
    latest = tmp_path / "latest.json"
    fills.write_text(
        json.dumps({"ts": "2026-05-05T14:00:00Z", "symbol": "AAPL"}) + "\n",
        encoding="utf-8",
    )

    rc = report_tool.main(
        [
            "--report-date",
            "2026-05-05",
            "--executable-symbols",
            "AAPL,AMZN",
            "--fills-jsonl",
            str(fills),
            "--sample-target",
            "10",
            "--output-json",
            str(out),
            "--latest-json",
            str(latest),
        ]
    )

    assert rc == 0
    assert json.loads(out.read_text(encoding="utf-8"))["status"] == "starved"
    assert latest.is_file()
