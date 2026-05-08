from __future__ import annotations

import json
from pathlib import Path

from ai_trading.tools import adversarial_failure_simulation as simulation
from ai_trading.tools import pretrade_risk_control_verifier as verifier


def _passing_controls() -> dict[str, dict[str, object]]:
    return {
        control: {"enabled": True, "fail_closed": True, "status": "passed"}
        for control in verifier.DEFAULT_REQUIRED_CONTROLS
    }


def test_adversarial_simulation_confirms_fail_closed_semantics() -> None:
    payload = simulation.build_adversarial_failure_simulation(
        report_date="2026-05-05",
        controls_config={"controls": _passing_controls()},
        scenarios=("missing_required_control", "unknown_control", "missing_intent_evidence"),
    )

    assert payload["status"] == "passed"
    assert payload["non_live"] is True
    assert {item["observed_status"] for item in payload["scenario_results"]} == {"fail_closed"}


def test_adversarial_simulation_treats_surveillance_breach_as_blocking() -> None:
    payload = simulation.build_adversarial_failure_simulation(
        report_date="2026-05-05",
        controls_config={"controls": _passing_controls()},
        surveillance_report={"status": "control_breach", "summary": {"findings": 2}},
        scenarios=("surveillance_breach_blocks",),
    )

    assert payload["status"] == "passed"
    assert payload["scenario_results"][0]["observed_status"] == "fail_closed"


def test_adversarial_simulation_cli_writes_artifacts(tmp_path: Path) -> None:
    controls = tmp_path / "controls.json"
    output = tmp_path / "simulation.json"
    latest = tmp_path / "latest.json"
    controls.write_text(json.dumps({"controls": _passing_controls()}), encoding="utf-8")

    rc = simulation.main(
        [
            "--report-date",
            "2026-05-05",
            "--controls-json",
            str(controls),
            "--scenario",
            "missing_required_control",
            "--scenario",
            "unknown_control",
            "--output-json",
            str(output),
            "--latest-json",
            str(latest),
        ]
    )

    assert rc == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "passed"
    assert latest.is_file()
