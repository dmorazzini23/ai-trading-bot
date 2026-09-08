import json
from datetime import UTC, date, datetime

from ai_trading.tools.scheduled_evidence_verification import verify_operator, verify_run
from ai_trading.tools.broker_accounting_evidence import daily_fee_totals


def test_actual_run_required_and_blocked_gates_are_explicit(tmp_path):
    output = tmp_path / "artifact.json"
    output.write_text(json.dumps({"status": "blocked"}))
    rows = [{"name": name, "status": "passed", "returncode": 0, "output_path": str(output)} for name in ("training_accelerator_daily", "regime_champion_models", "broker_accounting_evidence", "paper_evidence_review")]
    report = {"generated_at": "2026-09-07T21:00:00Z", "status": "blocked", "step_results": rows}
    assert verify_run(report, run_date=date(2026, 9, 7))["status"] == "workflow_executed"
    assert verify_run(report, run_date=date(2026, 9, 8))["status"] == "awaiting_or_incomplete_run"
    report["status"] = "planned"
    assert verify_run(report, run_date=date(2026, 9, 7))["status"] != "workflow_executed"
    report["status"] = "complete"
    rows[0]["status"] = "skipped"
    assert verify_run(report, run_date=date(2026, 9, 7))["status"] != "workflow_executed"
    assert verify_run({}, run_date=date(2026, 9, 7))["fresh_run"] is False


def test_operator_report_needs_current_accounting_diagnostics():
    run = {"expected_run_date": "2026-09-07", "status": "workflow_executed", "workflow_status": "blocked", "generated_at": "2026-09-07T20:59:00Z"}
    operator = {"generated_at": "2026-09-07T21:00:00Z", "status": "blocked", "health_report_summary": {"paper_execution_evidence": {"completion_gaps": [], "broker_accounting": {"status": "accounting_compared"}}}}
    assert verify_operator(operator, run)["status"] == "operator_report_verified"
    assert verify_operator({}, run)["status"] == "operator_report_pending"
    operator["generated_at"] = "2026-09-07T20:58:00Z"
    assert verify_operator(operator, run)["status"] == "operator_report_pending"


def test_accounting_date_fees_keep_currency_and_credit_separate():
    rows = [{"date": "2026-09-04", "currency": currency, "net_amount": amount} for currency, amount in [("USD", "-0.03"), ("USD", "0.01"), ("EUR", "-1"), ("USD", None)]]
    result = daily_fee_totals(rows)
    usd = next(row for row in result if row["currency"] == "USD")
    assert usd["charges"] == "0.03"
    assert usd["credits"] == "0.01"
    assert usd["net_charges"] == "0.02"
    assert usd["invalid_amounts"] == 1
    assert usd["per_fill_allocation"] is False
    assert daily_fee_totals([]) == []


def test_cli_reports_no_recorded_executions_without_claiming_session_validated(tmp_path, monkeypatch):
    from ai_trading.tools.scheduled_evidence_verification import main
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 7, 23, tzinfo=UTC)
    monkeypatch.setattr("ai_trading.tools.scheduled_evidence_verification.datetime", Clock)
    for name in ("decision_records", "order_events", "fill_events", "tca_records"):
        (tmp_path / f"{name}.jsonl").write_text("")
    boundaries = [{"account_id": "paper", "trading_mode": "paper", "positions_complete": True, "positions": {}, "timestamp": ts} for ts in ("2026-09-04T13:30:00Z", "2026-09-04T20:00:00Z")]
    (tmp_path / "broker_position_boundaries.jsonl").write_text("\n".join(json.dumps(row) for row in boundaries))
    monkeypatch.setattr("sys.argv", ["verify", "--runtime-dir", str(tmp_path), "--session", "2026-09-04", "--output-dir", str(tmp_path / "out")])
    main()
    report = json.loads((tmp_path / "out/latest.json").read_text())
    assert report["status"] == "evidence_pending"
    assert report["paper_session"]["no_recorded_executions"] is True
    assert "no_execution_samples" in report["paper_session"]["session_gaps"]
    assert report["alerts_sent"] == report["orders_sent"] == 0
