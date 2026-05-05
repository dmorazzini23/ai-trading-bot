from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ai_trading.tools import research_completion_notify


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload)), encoding="utf-8")


def test_research_completion_payload_reads_latest_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "reports"
    _write_json(
        root / "latest" / "daily_research_latest.json",
        {
            "status": "complete",
            "paths": {"report": "/runtime/research/research_automation_report.json"},
            "steps": [
                {
                    "name": "live_capital_readiness",
                    "output_path": str(root / "daily" / "run" / "live_capital_readiness.json"),
                }
            ],
            "step_results": [
                {"name": "live_cost_model", "status": "passed"},
                {"name": "multi_horizon_lightweight", "status": "skipped"},
            ],
        },
    )
    _write_json(
        root / "latest" / "daily_operator_summary.json",
        {"operator_action": "review_summary_and_generated_artifacts", "blocked_reasons": []},
    )
    _write_json(
        root / "latest" / "daily_readiness_latest.json",
        {"recommended_next_session_mode": "paper_trade", "trade_allowed": True},
    )
    _write_json(root / "daily" / "run" / "live_capital_readiness.json", {"status": "blocked"})

    payload = research_completion_notify.build_research_completion_payload(
        cadence="daily",
        workflow="daily",
        exit_code=0,
        report_root=root,
        channel="#all-beatwallstreet",
    )

    assert payload["channel"] == "#all-beatwallstreet"
    assert payload["text"].startswith("ai-trading research daily finished: complete")
    field_text = "\n".join(
        field["text"]
        for block in payload["blocks"]
        if block.get("type") == "section"
        for field in block.get("fields", [])
    )
    assert "paper_trade" in field_text
    assert "blocked" in field_text
    assert "multi_horizon_lightweight" in field_text


def test_research_completion_notify_dry_run_does_not_post(tmp_path: Path, capsys) -> None:
    rc = research_completion_notify.main(
        [
            "--cadence",
            "daily",
            "--report-root",
            str(tmp_path / "reports"),
            "--channel",
            "#all-beatwallstreet",
            "--dry-run",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sent"] is False
    assert payload["reason"] == "dry_run"
    assert payload["payload"]["channel"] == "#all-beatwallstreet"


def test_research_completion_notify_posts_when_webhook_present(monkeypatch, tmp_path: Path):
    calls: list[tuple[str, Mapping[str, Any], float]] = []

    def _fake_post(webhook_url: str, payload: Mapping[str, Any], timeout_s: float) -> int:
        calls.append((webhook_url, payload, timeout_s))
        return 200

    monkeypatch.setattr(research_completion_notify, "_post_slack_message", _fake_post)

    rc = research_completion_notify.main(
        [
            "--cadence",
            "weekly",
            "--report-root",
            str(tmp_path / "reports"),
            "--webhook-url",
            "https://hooks.slack.test/research",
            "--channel",
            "#all-beatwallstreet",
            "--timeout-s",
            "1.5",
        ]
    )

    assert rc == 0
    assert calls
    assert calls[0][0] == "https://hooks.slack.test/research"
    assert calls[0][1]["channel"] == "#all-beatwallstreet"
    assert calls[0][2] == 1.5
