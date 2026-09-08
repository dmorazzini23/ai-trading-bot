from types import SimpleNamespace

from ai_trading.tools.broker_accounting_evidence import capture, reconcile
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


def test_retirement_uses_development_evidence_only():
    fold = {"selected_candidates": 0, "abstention_diagnostics": {"scope": "inner_validation_only", "status": "abstained", "tested_percentiles": [{"mean_net_markout_bps": -5, "mean_gross_markout_bps": 1, "rejection_reasons": ["costs_exceed_positive_gross_edge"]}]}}
    report = review_candidate({"walk_forward": {"folds": [fold]}, "holdout_evaluation": {"mean_net_markout_bps": 100}})
    assert report["status"] == "retired_candidate"
    assert report["holdout_used_for_decision"] is False
    assert report["threshold_rejection_counts"] == {"costs_exceed_positive_gross_edge": 1}
    fold["abstention_diagnostics"]["scope"] = "holdout"
    assert review_candidate({"walk_forward": {"folds": [fold]}})["status"] == "review_required"
    assert review_candidate({})["status"] == "review_required"
