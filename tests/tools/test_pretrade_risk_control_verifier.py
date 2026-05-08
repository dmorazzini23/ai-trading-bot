from __future__ import annotations

import json
from pathlib import Path

from ai_trading.tools import pretrade_risk_control_verifier as verifier


def _passing_controls() -> dict[str, dict[str, object]]:
    return {
        control: {"enabled": True, "fail_closed": True, "status": "passed"}
        for control in verifier.DEFAULT_REQUIRED_CONTROLS
    }


def test_pretrade_verifier_fails_closed_for_missing_and_unknown_controls() -> None:
    controls = _passing_controls()
    controls.pop("kill_switch")
    controls["mystery_override"] = {"enabled": True, "fail_closed": True, "status": "passed"}

    payload = verifier.build_pretrade_risk_control_verification(
        report_date="2026-05-05",
        controls_config={"controls": controls},
    )

    assert payload["status"] == "fail_closed"
    assert payload["fail_closed"] is True
    assert {item["kind"] for item in payload["violations"]} >= {
        "missing_required_control",
        "unknown_control",
    }


def test_pretrade_verifier_requires_intent_control_evidence() -> None:
    controls = _passing_controls()
    intents = [
        {
            "intent_id": "intent-1",
            "controls": {
                "kill_switch": "passed",
                "buying_power": "passed",
            },
        }
    ]

    payload = verifier.build_pretrade_risk_control_verification(
        report_date="2026-05-05",
        controls_config={"controls": controls},
        intents=intents,
    )

    assert payload["status"] == "fail_closed"
    assert "intent_required_control_missing" in {item["kind"] for item in payload["violations"]}


def test_pretrade_verifier_cli_writes_artifacts_and_returns_failure(tmp_path: Path) -> None:
    controls_path = tmp_path / "controls.json"
    output_path = tmp_path / "pretrade.json"
    latest_path = tmp_path / "latest.json"
    controls_path.write_text(json.dumps({"controls": {"kill_switch": {"status": "unknown"}}}), encoding="utf-8")

    rc = verifier.main(
        [
            "--report-date",
            "2026-05-05",
            "--controls-json",
            str(controls_path),
            "--output-json",
            str(output_path),
            "--latest-json",
            str(latest_path),
        ]
    )

    assert rc == 2
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "fail_closed"
    assert latest_path.is_file()
