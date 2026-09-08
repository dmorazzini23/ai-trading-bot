"""Verify completed scheduled research and a specified observed paper session."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ai_trading.logging import get_logger
from ai_trading.runtime.atomic_io import atomic_write_text
from ai_trading.tools.execution_evidence_reconciliation import _read, reconcile_session
from ai_trading.utils.market_calendar import session_info


def verify_run(report: dict[str, Any], *, run_date: date) -> dict[str, Any]:
    try:
        generated = datetime.fromisoformat(str(report.get("generated_at") or "").replace("Z", "+00:00"))
        fresh = generated.tzinfo is not None and generated.astimezone(ZoneInfo("America/New_York")).date() == run_date
    except ValueError:
        fresh = False
    required = ("training_accelerator_daily", "regime_champion_models", "broker_accounting_evidence", "paper_evidence_review")
    results = {row["name"]: row for row in report.get("step_results", [])}
    steps = []
    for name in required:
        row = results.get(name, {})
        path = Path(row["output_path"]) if row.get("output_path") else None
        artifact = json.loads(path.read_text()) if path is not None and path.is_file() else {}
        steps.append({"name": name, "execution_status": row.get("status", "missing"), "returncode": row.get("returncode"), "artifact_status": artifact.get("status"), "artifact_exists": bool(artifact), "artifact_path": str(path) if path else None, "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path is not None and path.is_file() else None})
        if name == "training_accelerator_daily":
            steps[-1]["training_data_verified"] = artifact.get("data_provenance", {}).get("quality_passed") is True
            steps[-1]["training_blocked_reasons"] = artifact.get("blocked_reasons", [])
    executed = fresh and report.get("status") in {"complete", "blocked"} and all(row["execution_status"] in {"passed", "blocked"} and row["returncode"] in {0, 2} and row["artifact_exists"] for row in steps)
    return {"status": "workflow_executed" if executed else "awaiting_or_incomplete_run", "expected_run_date": run_date.isoformat(), "generated_at": report.get("generated_at"), "fresh_run": fresh, "workflow_status": report.get("status"), "steps": steps, "interpretation": "Executed blocked gates are reported separately from strategy eligibility."}


def verify_operator(operator: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    evidence = operator.get("health_report_summary", {}).get("paper_execution_evidence", {})
    try:
        generated = datetime.fromisoformat(str(operator.get("generated_at") or "").replace("Z", "+00:00"))
        run_generated = datetime.fromisoformat(str(run.get("generated_at") or "").replace("Z", "+00:00"))
        current = generated.tzinfo is not None and run_generated.tzinfo is not None and generated >= run_generated and generated.astimezone(ZoneInfo("America/New_York")).date().isoformat() == run["expected_run_date"] and operator.get("status") == run.get("workflow_status")
    except ValueError:
        current = False
    ready = current and bool(evidence.get("broker_accounting")) and "completion_gaps" in evidence and run["status"] == "workflow_executed"
    return {"status": "operator_report_verified" if ready else "operator_report_pending", "generated_at": operator.get("generated_at"), "evidence_status": evidence.get("status"), "completion_gaps": evidence.get("completion_gaps", [])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--session", type=date.fromisoformat, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    now = datetime.now(UTC)
    path = args.runtime_dir / "research_reports/latest/daily_research_automation_latest.json"
    run = verify_run(json.loads(path.read_text()) if path.exists() else {}, run_date=now.astimezone(ZoneInfo("America/New_York")).date())
    operator_path = path.with_name("daily_operator_summary.json")
    operator = verify_operator(json.loads(operator_path.read_text()) if operator_path.exists() else {}, run)
    session = session_info(args.session)
    if session is None:
        raise ValueError("requested date is not a trading session")
    audit: dict[str, Any] = {"status": "awaiting_session_close", "session_date": args.session.isoformat(), "verify_after_utc": (session.end_utc + timedelta(minutes=15)).isoformat()}
    if now >= session.end_utc + timedelta(minutes=15):
        inputs, sources = {}, {}
        for name, filename in {"decisions": "decision_records.jsonl", "orders": "order_events.jsonl", "fills": "fill_events.jsonl", "tca": "tca_records.jsonl", "boundaries": "broker_position_boundaries.jsonl"}.items():
            source = args.runtime_dir / filename
            inputs[name], _, sources[name] = _read(source) if source.exists() else ([], [], {"missing": True})
        accounts = {row["account_id"] for row in inputs["boundaries"] if row.get("account_id") and row.get("trading_mode") == "paper"}
        audit = reconcile_session(session_date=args.session.isoformat(), account_id=next(iter(accounts)), **inputs) if len(accounts) == 1 else {"status": "evidence_gaps", "session_gaps": ["unique_paper_account_required"]}
        audit["sources"] = sources
        source_gaps = [name for name, source in sources.items() if source.get("missing") or source.get("invalid_rows", 0) or source.get("stable_during_read") is False]
        if source_gaps:
            audit["status"] = "evidence_gaps"
            audit["input_evidence_gaps"] = source_gaps
        audit["no_recorded_executions"] = audit.get("accepted_unique_fills") == 0
    report = {"generated_at": now.isoformat(), "scheduled_run": run, "operator_report": operator, "paper_session": audit, "status": "verified" if run["status"] == "workflow_executed" and operator["status"] == "operator_report_verified" and audit["status"] == "paper_session_reconciled" else "evidence_pending", "orders_sent": 0, "alerts_sent": 0, "promotion_authority": False}
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    atomic_write_text(args.output_dir / f"verification_{now:%Y%m%dT%H%M%SZ}.json", encoded)
    atomic_write_text(args.output_dir / "latest.json", encoded)
    get_logger(__name__).info("SCHEDULED_EVIDENCE_VERIFICATION", extra={"status": report["status"], "workflow_status": run["status"], "session_status": audit["status"]})


if __name__ == "__main__":
    main()
