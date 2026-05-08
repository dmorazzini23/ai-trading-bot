from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def test_research_automation_script_delegates_plan_only(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    report_root = tmp_path / "reports"
    env = {
        **os.environ,
        "AI_TRADING_RESEARCH_REPORT_ROOT": str(report_root),
        "AI_TRADING_RESEARCH_PLAN_ONLY": "1",
        "AI_TRADING_RESEARCH_LOCK_DIR": str(tmp_path),
    }

    result = subprocess.run(
        [str(root / "scripts" / "run_research_automation.sh"), "daily"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    payload = json.loads(
        (report_root / "latest" / "daily_research_automation_latest.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["status"] == "planned"
    assert payload["cadence"] == "daily"


def test_research_automation_script_dry_run_skips_completion_notification(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    report_root = tmp_path / "reports"
    env = {
        **os.environ,
        "AI_TRADING_RESEARCH_REPORT_ROOT": str(report_root),
        "AI_TRADING_RESEARCH_DRY_RUN": "1",
        "AI_TRADING_RESEARCH_NOTIFY_SLACK": "1",
        "AI_TRADING_RESEARCH_LOCK_DIR": str(tmp_path),
        "AI_TRADING_RESEARCH_SLACK_WEBHOOK_URL": "https://hooks.slack.test/research",
    }

    result = subprocess.run(
        [str(root / "scripts" / "run_research_automation.sh"), "daily"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "skipping Slack/OpenClaw completion notification" in result.stderr
    assert '"reason": "dry_run"' not in result.stdout
    payload = json.loads(
        (report_root / "latest" / "daily_research_automation_latest.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["status"] == "dry_run"
