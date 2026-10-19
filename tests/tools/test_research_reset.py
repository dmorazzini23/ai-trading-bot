from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ai_trading.tools import research_automation as automation
from ai_trading.tools import research_reset as reset
from ai_trading.tools.experiment_ledger import claim_campaign_trial, finish_campaign_trial, register_campaign

NOW = datetime(2026, 9, 9, 1, tzinfo=UTC)


def test_policy_expiration_requires_explicit_review(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"enabled": True, "start_date": "2026-09-08", "review_date": "2026-10-08", "resume_policy": "explicit_review_required"}))
    assert reset.reset_policy(path, NOW)["phase"] == "evidence_reset"
    assert reset.reset_policy(path, datetime(2027, 1, 1, tzinfo=UTC))["phase"] == "review_due"
    assert reset.reset_policy(path, datetime(2026, 9, 7, tzinfo=UTC)) is None


@pytest.mark.parametrize("cadence", ["daily", "weekly", "monthly", "weekend-saturday", "weekend-sunday"])
def test_reset_plans_only_collect_evidence(cadence, tmp_path, monkeypatch):
    monkeypatch.setattr(automation, "_now", lambda: NOW)
    config = automation._build_config(automation._build_parser().parse_args([cadence, "--report-root", str(tmp_path), "--plan-only"]))
    steps, blocked = automation.build_research_steps(config)
    assert blocked == []
    assert [step.name for step in steps] == ["broker_accounting_evidence", "paper_evidence_review", "research_reset_scorecard"]
    assert all(step.required for step in steps)
    assert not any("train" in arg or "notify" in arg for step in steps for arg in step.command)
    monkeypatch.setattr(automation, "_RESET_POLICY_PATH", tmp_path / "missing.json")
    steps, blocked = automation.build_research_steps(config)
    assert steps == []
    assert blocked == ["research_reset_policy_invalid:FileNotFoundError"]


def complete_inputs():
    decision = {"symbol": "SPY", "correlation_id": "d", "metrics": {"decision_ts": "2026-09-08T14:00:00Z", "decision_ts_basis": "explicit", "source_timestamp": "2026-09-08T13:59:00Z", "quote_timestamp": "2026-09-08T13:59:59.500Z", "raw_quote_bid": 100, "raw_quote_ask": 100.01}}
    order = {"symbol": "SPY", "side": "buy", "correlation_id": "d", "order_id": "o", "account_id": "paper", "trading_mode": "paper", "ts": "2026-09-08T14:00:00.1Z"}
    fill = {"symbol": "SPY", "side": "buy", "correlation_id": "d", "order_id": "o", "fill_id": "f", "fill_qty": 1, "fill_price": 100.01, "account_id": "paper", "trading_mode": "paper", "ts": "2026-09-08T14:00:00.2Z", "fee_amount": .01, "fee_source": "broker_activity", "fee_basis": "per_fill_total", "fee_currency": "USD"}
    return [decision], [order], [fill]


def test_chain_requires_causal_quotes_and_verified_fees():
    decisions, orders, fills = complete_inputs()
    assert reset.execution_chain(decisions, orders, fills)["complete_chains"] == 1
    decisions[0]["metrics"]["quote_timestamp"] = "2026-09-08T14:00:00.1Z"
    fills[0]["fee_source"] = "configured_estimate"
    result = reset.execution_chain(decisions, orders, fills)
    assert result["complete_chains"] == 0
    assert result["gap_counts"]["verified_total_fees_missing"] == 1
    assert result["gap_counts"]["causal_quote_unverified"] == 1


def durable_exit(recorded_at="2026-09-08T14:00:00.05Z"):
    return {"symbol": "SPY", "side": "buy", "account_id": "paper", "decision_trace_id": "eod-1",
            "evidence_source": "oms_decision_event", "recorded_at": recorded_at,
            "metrics": {"decision_ts": "2026-09-08T14:00:00Z", "decision_ts_basis": "explicit"}}


def eod_fill_inputs():
    _, orders, fills = complete_inputs()
    for row in (orders[0], fills[0]):
        row.pop("correlation_id")
        row["decision_trace_id"] = "eod-1"
    return orders, fills


def test_chain_links_causal_durable_decision_only():
    orders, fills = eod_fill_inputs()
    linked = reset.execution_chain([], orders, fills, durable=[durable_exit()])["gap_counts"]
    assert "decision_missing_or_ambiguous" not in linked
    assert "durable_decision_not_causal" not in linked
    late = reset.execution_chain([], orders, fills, durable=[durable_exit("2026-09-08T14:00:01Z")])
    assert late["gap_counts"]["durable_decision_not_causal"] == 1
    assert late["complete_chains"] == 0


def test_file_decision_takes_precedence_over_durable_decision():
    decisions, orders, fills = complete_inputs()
    fills[0]["decision_trace_id"] = "eod-1"
    result = reset.execution_chain(decisions, orders, fills, durable=[durable_exit()])
    assert result["complete_chains"] == 1


def test_scorecard_uses_durable_exit_decisions_in_window(tmp_path, monkeypatch):
    now = datetime(2026, 9, 8, 21, tzinfo=UTC)
    orders, fills = eod_fill_inputs()
    file_decision = {**complete_inputs()[0][0], "ts": "2026-09-08T14:00:00Z"}
    for name, rows in (("decision_records", [file_decision]), ("order_events", orders), ("fill_events", fills)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    durable = {key: value for key, value in durable_exit().items() if key not in {"metrics", "evidence_source"}}
    durable["decision_ts"] = durable_exit()["metrics"]["decision_ts"]
    stale = {**durable, "decision_trace_id": "old", "recorded_at": "2026-08-01T14:00:00Z"}
    sessions = []

    def diagnostic(session, database):
        sessions.append(session)
        return {"status": "local_diagnostic_complete", "decisions": [durable, stale] if session == "2026-09-08" else []}

    monkeypatch.setattr(reset, "session_diagnostic", diagnostic)
    report = reset.build_scorecard(tmp_path, tmp_path / "a.json", tmp_path / "p.json", [], now, oms_database=tmp_path / "oms.db")
    assert "decision_missing_or_ambiguous" not in report["execution_chain"]["gap_counts"]
    assert report["sources"]["oms_decisions"]["rows_in_window"] == 1
    assert sessions[0] == "2026-09-01" and sessions[-1] == "2026-09-08"
    assert report["quotes"]["rows_read"] == 1
    without = reset.build_scorecard(tmp_path, tmp_path / "a.json", tmp_path / "p.json", [], now)
    assert without["execution_chain"]["gap_counts"]["decision_missing_or_ambiguous"] == 1


def test_chain_rejects_duplicate_conflicts_wrong_account_and_missing_ids():
    decisions, orders, fills = complete_inputs()
    duplicate = deepcopy(fills[0])
    duplicate["fill_price"] = 999
    orders[0]["account_id"] = "other"
    result = reset.execution_chain(decisions, orders, [*fills, duplicate, {}])
    assert result["complete_chains"] == 0
    assert result["gap_counts"]["conflicting_fill"] == 1
    assert result["gap_counts"]["causal_account_order_unverified"] == 1
    assert result["gap_counts"]["fill_id_missing"] == 1
    assert reset.execution_chain([], [], [])["complete_chain_fraction"] is None


def test_correlation_alone_cannot_match_another_order():
    decisions, orders, fills = complete_inputs()
    orders[0]["order_id"] = "different-order-same-decision"
    result = reset.execution_chain(decisions, orders, fills)
    assert result["complete_chains"] == 0
    assert result["gap_counts"]["causal_account_order_unverified"] == 1


def test_opposite_order_side_does_not_complete_chain():
    decisions, orders, fills = complete_inputs()
    orders[0]["side"] = "sell"
    assert reset.execution_chain(decisions, orders, fills)["complete_chains"] == 0


@pytest.mark.parametrize("field,value", [
    ("account_id", "other-account"),
    ("trading_mode", "live"),
    ("side", "sell"),
])
def test_decision_identity_conflicts_do_not_complete_chain(field, value):
    decisions, orders, fills = complete_inputs()
    decisions[0][field] = value

    result = reset.execution_chain(decisions, orders, fills)

    assert result["status"] == "evidence_pending"
    assert result["gap_counts"]["decision_identity_mismatch"] == 1


def test_window_counts_exclusions_and_hashes_source(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('\n'.join([json.dumps({"ts": ts}) for ts in ("2026-09-08T14:00:00Z", "2024-01-01T00:00:00Z", "2027-01-01T00:00:00Z", "2026-09-08T00:00:00")] + ['bad', '[]']))
    rows, source = reset.read_window(path, NOW)
    assert len(rows) == 1
    assert source["counts"] == {"rows_read": 6, "before_window": 1, "future_timestamp": 1, "invalid_timestamp": 1, "malformed": 2}
    assert source["sha256"]
    assert source["stable_during_read"]


def test_campaign_scorecard_verifies_ledger_and_does_not_count_snapshots(tmp_path, monkeypatch):
    from ai_trading.tools import experiment_ledger
    monkeypatch.setattr(experiment_ledger, "_utc_now", lambda: NOW)
    path = tmp_path / "campaign.json"
    contract = {"campaign_id": "fixed", "max_trials": 1, "hypotheses": ["fixed"], "development_start": "2024-01-01", "development_end": "2025-12-31", "holdout_start": "2026-09-09", "holdout_end": "2026-12-08"}
    register_campaign(path, contract)
    claim_campaign_trial(path, hypothesis_id="fixed", evidence_signature="hash", evaluation_start="2024-01-01", evaluation_end="2025-12-31", quality_passed=True)
    report = tmp_path / "report.json"
    report.write_text('{"decision": "retired", "decision_periods": 4, "block_bootstrap_95_bps": {"vs_cash": [1, 1]}}')
    finish_campaign_trial(path, evidence_signature="hash", decision="retired", report_path=report)
    results, counts = reset.campaign_results([path, path], NOW)
    assert len(results) == 1
    assert counts["concluded_last_seven_days"] == 1
    assert results[0]["uncertainty"] is None
    assert results[0]["uncertainty_status"] == "unavailable_fewer_than_two_blocks"
    report.write_text('{"decision": "eligible_for_untouched_test_planning"}')
    _, counts = reset.campaign_results([path], NOW)
    assert counts.get("concluded_total", 0) == 0


def test_missing_sources_do_not_claim_profitability(tmp_path):
    result = reset.build_scorecard(tmp_path, tmp_path / "account.json", tmp_path / "paper.json", [], NOW)
    assert result["status"] == "evidence_pending"
    assert result["execution_chain"]["complete_chain_fraction"] is None
    assert result["untouched_net_performance"]["status"] == "unavailable"
    assert result["promotion_authority"] is False
    assert result["orders_sent"] == 0


def test_scorecard_prioritizes_known_sampling_gap_and_exposes_fee_schema(tmp_path):
    sampling = tmp_path / "sampling.json"
    sampling.write_text(json.dumps({"generated_at": NOW.isoformat(), "status": "blocked_sampling_support", "possible_common_periods": 4, "trial_claimed": False}))
    accounting = tmp_path / "accounting.json"
    accounting.write_text(json.dumps({"generated_at": NOW.isoformat(), "fee_record_coverage": {"fee_rows_with_execution_reference": 0}}))
    report = reset.build_scorecard(tmp_path, accounting, tmp_path / "paper.json", [], NOW, sampling)
    assert report["highest_value_blocker"] == "sampling_support_insufficient"
    assert report["sampling_support"]["possible_common_periods"] == 4
    assert report["accounting"]["fee_record_coverage"]["fee_rows_with_execution_reference"] == 0
    sampling.unlink()
    report = reset.build_scorecard(tmp_path, accounting, tmp_path / "paper.json", [], NOW, sampling)
    assert report["highest_value_blocker"] == "sampling_support_audit_missing_or_stale"


def test_scorecard_reports_broker_portfolio_history_next_to_internal_fills(tmp_path):
    from zoneinfo import ZoneInfo

    from ai_trading.analytics import portfolio_history

    now = datetime(2026, 10, 19, 3, tzinfo=UTC)  # Sunday night: no window edge cuts a session
    fills = [
        {"ts": "2026-10-15T14:00:00+00:00", "symbol": "AAPL", "side": "buy", "fill_qty": 2, "fill_price": 100.0},
        {"ts": "2026-10-15T15:00:00+00:00", "symbol": "AAPL", "side": "sell", "fill_qty": 2, "fill_price": 101.5},
        {"ts": "2026-10-16T15:00:00+00:00", "symbol": "MSFT", "side": "buy", "fill_qty": 1, "fill_price": 50.0},
        {"ts": "2026-10-16T16:00:00+00:00", "symbol": "MSFT", "side": "sell", "fill_qty": 1, "fill_price": 50.76},
    ]
    (tmp_path / "fill_events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in fills))
    close = lambda day: int(datetime.fromisoformat(f"{day}T20:00:00").replace(tzinfo=ZoneInfo("America/New_York")).timestamp())  # noqa: E731
    days = ("2026-10-09", "2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15", "2026-10-16")
    history = portfolio_history.normalize_portfolio_history(
        {"timestamp": [close(day) for day in days], "equity": [81661.54, 81662.82, 81660.62, 81659.90, 81665.36, 81666.08], "profit_loss": [4.24, 1.28, -2.20, -0.72, 5.46, 0.72], "profit_loss_pct": [0.0001, 0.0, 0.0, 0.0, 0.0001, 0.0], "base_value": 81662.2},
        period="1M", timeframe="1D", fetched_at=now.isoformat(),
    )

    report = reset.build_scorecard(tmp_path, tmp_path / "a.json", tmp_path / "p.json", [], now, broker_portfolio_history=history)

    broker = report["broker_portfolio"]
    assert broker["available"] is True
    assert broker["reported_only"] is True
    assert broker["evidence_class"] == "broker_reported_portfolio_history"
    assert broker["internal_evidence_class"] == "internal_same_day_fill_round_trips"
    assert [row["date"] for row in broker["daily"]] == list(days[1:])  # window opens Sunday the 11th
    daily = {row["date"]: row for row in broker["daily"]}
    assert daily["2026-10-15"]["broker_daily_pnl"] == 5.46
    assert daily["2026-10-15"]["internal_same_day_fill_pnl"] == 3.0
    assert daily["2026-10-15"]["difference_usd"] == pytest.approx(2.46)
    assert daily["2026-10-16"]["internal_same_day_fill_pnl"] == pytest.approx(0.76)
    assert daily["2026-10-12"]["internal_same_day_fill_pnl"] == 0.0
    assert broker["broker_weekly_equity_change"] == pytest.approx(81666.08 - 81661.54)
    assert broker["internal_weekly_same_day_fill_pnl"] == pytest.approx(3.76)
    assert broker["weekly_difference_usd"] == pytest.approx((1.28 - 2.20 - 0.72 + 5.46 + 0.72) - 3.76)
    # Reported only: the broker series never becomes a blocker.
    assert not any("broker_portfolio" in blocker for blocker in report["blockers"])
    without = reset.build_scorecard(tmp_path, tmp_path / "a.json", tmp_path / "p.json", [], now)
    assert without["blockers"] == report["blockers"]
    assert without["broker_portfolio"]["available"] is False
    assert without["broker_portfolio"]["reason"] == "not_requested"
    json.dumps(report, allow_nan=False)


def test_scorecard_marks_window_edge_sessions_partial(tmp_path):
    now = datetime(2026, 10, 16, 18, tzinfo=UTC)  # Friday 14:00 New York: both edges are mid-session
    report = reset.build_scorecard(tmp_path, tmp_path / "a.json", tmp_path / "p.json", [], now)
    assert report["broker_portfolio"]["internal_available"] is False  # no fill file
    assert reset._window_edge_partial(now) is True
    assert reset._window_edge_partial(datetime(2026, 10, 19, 3, tzinfo=UTC)) is False
