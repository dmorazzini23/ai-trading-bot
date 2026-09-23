import json
import sys
from types import SimpleNamespace

import pytest

from ai_trading.tools.broker_accounting_evidence import capture, main, reconcile, reconcile_account_equity
from ai_trading.tools.candidate_abstention_review import review_candidate


def test_activity_capture_paginates_read_only_and_detects_truncation():
    class Client:
        def get_account(self):
            return SimpleNamespace(id="paper")
        def get(self, path, data):
            assert path == "/account/activities"
            assert data["direction"] == "asc"
            return [] if "page_token" in data else [{"id": str(i)} for i in range(100)]
    assert capture(Client(), after="2026-09-01", max_pages=1)["pagination_complete"] is False
    assert capture(Client(), after="2026-09-01")["pagination_complete"] is True


def test_activity_capture_records_a_time_stamped_broker_account_boundary():
    class Client:
        def get_account(self):
            return SimpleNamespace(id="paper", currency="USD", cash="975.00",
                                   equity="1010.00", portfolio_value="1010.00")

        def get(self, _path, *, data):
            return []

    snapshot = capture(Client(), after="2026-09-01T00:00:00Z")
    boundary = snapshot["account_boundary"]
    assert boundary["account_id"] == "paper"
    assert boundary["cash"] == "975.00"
    assert boundary["equity"] == "1010.00"
    assert boundary["timestamp"] <= snapshot["fetched_at"]


def _equity_fixture():
    opening = {"account_id": "paper", "trading_mode": "paper", "currency": "USD",
               "timestamp": "2026-09-22T13:00:00Z", "cash": "1000.00", "equity": "1000.00"}
    closing = {**opening, "timestamp": "2026-09-22T20:00:00Z",
               "cash": "909.99", "equity": "1010.00"}
    activities = {
        "account_id": "paper", "trading_mode": "paper", "pagination_complete": True,
        "after": "2026-09-22T12:59:00Z", "fetched_at": "2026-09-22T20:01:00Z",
        "activities": [
            {"id": "trade", "activity_type": "FILL", "order_id": "order", "symbol": "AAPL",
             "side": "buy", "qty": "1", "price": "100", "transaction_time": "2026-09-22T14:00:00Z"},
            {"id": "dividend", "activity_type": "DIV", "currency": "USD",
             "net_amount": "10", "executed_at": "2026-09-22T18:00:00Z"},
            {"id": "fee", "activity_type": "FEE", "currency": "USD",
             "net_amount": "-0.01", "executed_at": "2026-09-22T19:00:00Z"},
        ],
    }
    return activities, opening, closing


def test_equity_audit_reconciles_cash_and_reports_equity_without_claiming_net_strategy_return():
    activities, opening, closing = _equity_fixture()
    result = reconcile_account_equity(activities, opening, closing)
    assert result["status"] == "cash_reconciled"
    assert result["execution_cash_effect_usd"] == "-100"
    assert result["other_booked_cash_effect_usd"] == "9.99"
    assert result["booked_fee_cash_effect_usd"] == "-0.01"
    assert result["equity_change_usd"] == "10.00"
    assert result["performance_authority"] is False
    assert result["fee_allocation"] == "unknown_per_execution"


def test_equity_audit_leaves_date_only_fee_and_unknown_cash_movement_unverified():
    activities, opening, closing = _equity_fixture()
    activities["activities"][2].pop("executed_at")
    activities["activities"][2]["date"] = "2026-09-22"
    result = reconcile_account_equity(activities, opening, closing)
    assert result["status"] == "unverified"
    assert result["unresolved_activities"] == [
        {"activity_id": "fee", "reason": "effective_instant_unavailable"}
    ]
    activities["activities"].append({"id": "split", "activity_type": "SPLIT",
                                      "executed_at": "2026-09-22T18:30:00Z"})
    assert {row["activity_id"] for row in reconcile_account_equity(activities, opening, closing)["unresolved_activities"]} == {"fee", "split"}
    activities["activities"][2]["date"] = "2026-09-21"
    assert reconcile_account_equity(activities, opening, closing)["status"] == "unverified"


def test_equity_audit_does_not_hide_one_cent_cash_difference():
    activities, opening, closing = _equity_fixture()
    closing["cash"] = "910.00"
    result = reconcile_account_equity(activities, opening, closing)
    assert result["status"] == "unverified"
    assert result["cash_difference_usd"] == "0.01"


def test_equity_audit_refuses_pending_activity_or_mode_conflict():
    activities, opening, closing = _equity_fixture()
    activities["activities"][2]["status"] = "pending"
    assert reconcile_account_equity(activities, opening, closing)["unresolved_activities"] == [
        {"activity_id": "fee", "reason": "activity_not_executed"}
    ]
    activities["activities"][2]["trading_mode"] = "live"
    with pytest.raises(ValueError, match="trading-mode conflict"):
        reconcile_account_equity(activities, opening, closing)


def test_equity_audit_rejects_incomplete_or_conflicting_source_boundaries():
    activities, opening, closing = _equity_fixture()
    activities["pagination_complete"] = False
    with pytest.raises(ValueError, match="complete same-account"):
        reconcile_account_equity(activities, opening, closing)
    activities["pagination_complete"] = True
    closing["account_id"] = "other"
    with pytest.raises(ValueError, match="complete same-account"):
        reconcile_account_equity(activities, opening, closing)
    closing["account_id"] = "paper"
    activities["activities"].append({"id": "trade", "activity_type": "FILL", "qty": "999"})
    with pytest.raises(ValueError, match="conflicting activity identity"):
        reconcile_account_equity(activities, opening, closing)


def test_equity_cli_writes_separate_audit_without_changing_source_files(tmp_path, monkeypatch):
    activities, opening, closing = _equity_fixture()
    paths = {name: tmp_path / f"{name}.json" for name in ("snapshot", "opening", "closing")}
    paths["snapshot"].write_text(json.dumps(activities))
    paths["opening"].write_text(json.dumps({"account_boundary": opening}))
    paths["closing"].write_text(json.dumps({"account_boundary": closing}))
    fills = tmp_path / "fills.jsonl"
    fills.write_text("")
    report = tmp_path / "accounting.json"
    equity = tmp_path / "equity.json"
    original = paths["snapshot"].read_bytes()
    monkeypatch.setattr(sys, "argv", ["broker_accounting_evidence", "--fills", str(fills),
                                  "--snapshot", str(paths["snapshot"]), "--output", str(report),
                                  "--opening-account", str(paths["opening"]),
                                  "--closing-account", str(paths["closing"]),
                                  "--equity-output", str(equity)])
    main()
    assert json.loads(equity.read_text())["status"] == "cash_reconciled"
    assert paths["snapshot"].read_bytes() == original
    assert report.exists()


def test_accounting_preserves_unallocated_fees_and_does_not_infer_zero():
    snapshot = {"account_id": "paper", "trading_mode": "paper", "pagination_complete": True, "after": "2026-09-01", "activities": [{"id": "a", "activity_type": "FILL", "order_id": "o", "qty": "2"}, {"id": "fee", "activity_type": "FEE", "net_amount": "-0.02"}]}
    fills = [{"account_id": "paper", "trading_mode": "paper", "fill_id": "f", "order_id": "o", "fill_qty": 2, "ts": "2026-09-02T15:00:00Z"}]
    result = reconcile(snapshot, fills)
    assert result["order_quantity_comparison"][0]["status"] == "quantity_matched"
    assert result["fee_coverage"] == {"total_fee_unknown": 1}
    assert result["fee_activities"][0]["allocation_status"].startswith("unallocated")
    fills[0].pop("account_id")
    result = reconcile(snapshot, fills)
    assert result["recovered_order_account_links"] == 1
    assert result["order_quantity_counts"] == {"quantity_matched": 1}
    assert "account_id" not in fills[0]
    fills[0]["account_id"] = "different"
    assert reconcile(snapshot, fills)["recovered_order_account_links"] == 0
    snapshot["activities"] = snapshot["activities"][:1]
    assert reconcile(snapshot, fills)["net_fee_validation"] != "validated"
    snapshot["pagination_complete"] = False
    assert reconcile(snapshot, fills)["status"] == "accounting_incomplete"


def test_accounting_flags_aggregated_local_fill_despite_matching_order_quantity():
    snapshot = {
        "account_id": "paper", "trading_mode": "paper", "pagination_complete": True,
        "after": "2026-09-22T13:00:00Z",
        "activities": [
            {"id": "broker-1", "activity_type": "FILL", "order_id": "exit", "qty": "1"},
            {"id": "broker-2", "activity_type": "FILL", "order_id": "exit", "qty": "1"},
        ],
    }
    fills = [{
        "account_id": "paper", "trading_mode": "paper", "fill_id": "aggregate",
        "order_id": "exit", "fill_qty": 2, "ts": "2026-09-22T19:55:15Z",
    }]
    comparison = reconcile(snapshot, fills)["order_quantity_comparison"][0]
    assert comparison["status"] == "quantity_matched"
    assert comparison["broker_execution_count"] == 2
    assert comparison["recorded_fill_count"] == 1
    assert comparison["execution_count_status"] == "count_mismatch"


def test_retirement_uses_development_evidence_only():
    fold = {"selected_candidates": 0, "abstention_diagnostics": {"scope": "inner_validation_only", "status": "abstained", "tested_percentiles": [{"mean_net_markout_bps": -5, "mean_gross_markout_bps": 1, "rejection_reasons": ["costs_exceed_positive_gross_edge"]}]}}
    report = review_candidate({"walk_forward": {"folds": [fold]}, "holdout_evaluation": {"mean_net_markout_bps": 100}})
    assert report["status"] == "retired_candidate"
    assert report["holdout_used_for_decision"] is False
    assert report["threshold_rejection_counts"] == {"costs_exceed_positive_gross_edge": 1}
    fold["abstention_diagnostics"]["scope"] = "holdout"
    assert review_candidate({"walk_forward": {"folds": [fold]}})["status"] == "review_required"
    assert review_candidate({})["status"] == "review_required"
def test_fee_schema_audit_does_not_allocate_account_charges():
    from ai_trading.tools.broker_accounting_evidence import fee_record_coverage
    snapshot = {"pagination_complete": True, "activities": [{"activity_type": "FILL", "id": "fill", "order_id": "order", "price": "100", "qty": "1"}, {"activity_type": "FEE", "id": "fee", "net_amount": "-.01", "date": "2026-09-08", "currency": "USD"}]}
    result = fee_record_coverage(snapshot)
    assert result["fill_activity_rows"] == result["fee_activity_rows"] == 1
    assert result["fill_rows_with_fee_amount"] == 0
    assert result["fee_rows_with_execution_reference"] == 0
    assert result["status"] == "no_execution_linkage_in_fee_records"
    snapshot["activities"][1]["order_id"] = "order"
    assert fee_record_coverage(snapshot)["status"] == "references_present_completeness_unverified"
    assert fee_record_coverage({})["fill_rows_with_fee_amount"] == 0
