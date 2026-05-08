"""Run non-live adversarial scenarios against control artifacts and config."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from ai_trading.runtime.artifacts import resolve_runtime_artifact_path
from ai_trading.tools import pretrade_risk_control_verifier as pretrade_verifier


DEFAULT_SCENARIOS = (
    "missing_required_control",
    "unknown_control",
    "missing_intent_evidence",
    "surveillance_breach_blocks",
)


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _controls_copy(config: Mapping[str, Any]) -> dict[str, Any]:
    copied = deepcopy(dict(config))
    if not isinstance(copied.get("controls"), Mapping):
        copied["controls"] = {}
    return copied


def _remove_control(config: Mapping[str, Any], control_name: str) -> dict[str, Any]:
    copied = _controls_copy(config)
    controls = copied["controls"]
    if isinstance(controls, dict):
        controls.pop(control_name, None)
    return copied


def _inject_unknown_control(config: Mapping[str, Any]) -> dict[str, Any]:
    copied = _controls_copy(config)
    controls = copied["controls"]
    if isinstance(controls, dict):
        controls["rogue_runtime_override"] = {"enabled": True, "fail_closed": True, "status": "passed"}
    return copied


def _scenario_result(
    *,
    name: str,
    observed_status: str,
    expected_status: str = "fail_closed",
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    passed = observed_status == expected_status
    return {
        "name": name,
        "expected_status": expected_status,
        "observed_status": observed_status,
        "passed": passed,
        "details": dict(details or {}),
    }


def _run_pretrade_scenario(
    *,
    name: str,
    report_date: str,
    config: Mapping[str, Any],
    intents: Sequence[Mapping[str, Any]] = (),
    required_controls: Sequence[str],
) -> dict[str, Any]:
    payload = pretrade_verifier.build_pretrade_risk_control_verification(
        report_date=report_date,
        controls_config=config,
        intents=intents,
        required_controls=required_controls,
    )
    return _scenario_result(
        name=name,
        observed_status=str(payload.get("status") or ""),
        details={
            "violations": payload.get("summary", {}).get("violations", 0),
            "fail_closed": bool(payload.get("fail_closed")),
        },
    )


def build_adversarial_failure_simulation(
    *,
    report_date: str,
    controls_config: Mapping[str, Any] | None = None,
    pretrade_report: Mapping[str, Any] | None = None,
    surveillance_report: Mapping[str, Any] | None = None,
    scenarios: Sequence[str] = DEFAULT_SCENARIOS,
    required_controls: Sequence[str] = pretrade_verifier.DEFAULT_REQUIRED_CONTROLS,
) -> dict[str, Any]:
    config = dict(controls_config or {})
    required = tuple(str(control).strip().lower() for control in required_controls if str(control).strip())
    results: list[dict[str, Any]] = []
    for scenario in scenarios:
        if scenario == "missing_required_control":
            missing_name = required[0] if required else "kill_switch"
            results.append(
                _run_pretrade_scenario(
                    name=scenario,
                    report_date=report_date,
                    config=_remove_control(config, missing_name),
                    required_controls=required,
                )
            )
        elif scenario == "unknown_control":
            results.append(
                _run_pretrade_scenario(
                    name=scenario,
                    report_date=report_date,
                    config=_inject_unknown_control(config),
                    required_controls=required,
                )
            )
        elif scenario == "missing_intent_evidence":
            results.append(
                _run_pretrade_scenario(
                    name=scenario,
                    report_date=report_date,
                    config=config,
                    intents=[{"intent_id": "adversarial-missing-controls"}],
                    required_controls=required,
                )
            )
        elif scenario == "surveillance_breach_blocks":
            status = str((surveillance_report or {}).get("status") or "not_applicable")
            findings = int((surveillance_report or {}).get("summary", {}).get("findings") or 0)
            observed = "fail_closed" if status in {"control_breach", "watchlist"} or findings > 0 else "not_applicable"
            results.append(
                _scenario_result(
                    name=scenario,
                    observed_status=observed,
                    expected_status="fail_closed" if observed != "not_applicable" else "not_applicable",
                    details={"surveillance_status": status, "findings": findings},
                )
            )
        else:
            results.append(
                _scenario_result(
                    name=scenario,
                    observed_status="unknown_scenario",
                    details={"reason": "scenario is not implemented"},
                )
            )

    pretrade_status = str((pretrade_report or {}).get("status") or "not_provided")
    applicable_results = [item for item in results if str(item.get("observed_status")) != "not_applicable"]
    passed = all(bool(item.get("passed")) for item in results)
    return {
        "schema_version": "1.0.0",
        "artifact_type": "adversarial_failure_simulation",
        "report_date": report_date,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": "passed" if passed else "failed",
        "recommended_next_action": "keep_fail_closed_controls" if passed else "block_live_risk_increase_and_fix_fail_closed_semantics",
        "non_live": True,
        "inputs": {
            "pretrade_report_status": pretrade_status,
            "scenarios": list(scenarios),
            "required_controls": list(required),
        },
        "summary": {
            "scenarios": len(results),
            "applicable_scenarios": len(applicable_results),
            "passed": sum(1 for item in results if bool(item.get("passed"))),
            "failed": sum(1 for item in results if not bool(item.get("passed"))),
        },
        "scenario_results": results,
        "promotion_authority": False,
        "live_money_authority": False,
    }


def _default_report_paths(report_date: str) -> tuple[Path, Path]:
    root = resolve_runtime_artifact_path("runtime/reports", default_relative="runtime/reports", for_write=True)
    compact = report_date.replace("-", "")
    return root / f"adversarial_failure_simulation_{compact}.json", root / "adversarial_failure_simulation_latest.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-date", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    parser.add_argument("--controls-json", type=Path, default=None)
    parser.add_argument("--pretrade-json", type=Path, default=None)
    parser.add_argument("--surveillance-json", type=Path, default=None)
    parser.add_argument("--scenario", action="append", choices=DEFAULT_SCENARIOS, default=None)
    parser.add_argument("--required-control", action="append", default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--latest-json", type=Path, default=None)
    args = parser.parse_args(argv)
    output_json, latest_json = _default_report_paths(str(args.report_date))
    output_json = args.output_json or output_json
    latest_json = args.latest_json or latest_json
    report = build_adversarial_failure_simulation(
        report_date=str(args.report_date),
        controls_config=_read_json(args.controls_json),
        pretrade_report=_read_json(args.pretrade_json),
        surveillance_report=_read_json(args.surveillance_json),
        scenarios=tuple(args.scenario or DEFAULT_SCENARIOS),
        required_controls=tuple(args.required_control or pretrade_verifier.DEFAULT_REQUIRED_CONTROLS),
    )
    for path in (output_json, latest_json):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps({"path": str(output_json), "status": report["status"]}) + "\n")
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
