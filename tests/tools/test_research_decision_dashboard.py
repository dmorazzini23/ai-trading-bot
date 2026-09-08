import numpy as np
import pandas as pd
import pytest

from ai_trading.tools.research_decision_dashboard import candidate_review, evaluate_portfolio, render


def test_funnel_reconciles_without_conflating_execution_and_training():
    row = {"model_name": "test", "walk_forward": {"folds": [{"fold_index": 1, "opportunity_funnel": {"outer_test_rows": 100, "finite_scores": 95, "selected_by_frozen_threshold": 10}}]}}
    result = candidate_review(row)
    assert result["opportunity_funnel"][0]["invalid_scores"] == 5
    assert result["opportunity_funnel"][0]["threshold_rejections"] == 85
    assert result["execution_funnel"]["status"] == "separate_evidence_required"
    assert all(r["status"] == "missing_or_unpaired" for r in result["baselines"])


def test_baselines_require_identical_fold_and_opportunity_coverage():
    folds = [{"fold_index": i, "opportunities": 100, "net_edge_per_opportunity_bps": 2} for i in (1, 2)]
    baseline = [{**row, "net_edge_per_opportunity_bps": 0} for row in folds]
    row = {"walk_forward": {"controlled_comparisons": {"scope": "development_outer_folds_only", "selection_scope": "nested_inner_validation_only", "metric": "equal_notional_markout_per_common_opportunity_not_portfolio_return", "variants": [{"name": "candidate", "folds": folds}, {"name": "cash", "folds": baseline}]}}}
    result = candidate_review(row)["baselines"][0]
    assert result["candidate_minus_baseline_bps"] == 2
    baseline[0]["opportunities"] = 99
    assert candidate_review(row)["baselines"][0]["status"] == "missing_or_unpaired"


def test_fixed_capital_portfolio_costs_and_compounding():
    rows = [{"session": str(day.date()), "strategy": name, "gross_return": gain, "turnover": 2, "gross_exposure": 1} for day in pd.date_range("2026-01-01", periods=20) for name, gain in (("existing", 0.01), ("candidate", -0.005))]
    frame = pd.DataFrame(rows)
    report = evaluate_portfolio(frame, candidate="candidate", costs=(0, 10))
    assert report["status"] == "evaluated"
    assert report["scenarios"][0]["after"]["total_return"] == pytest.approx((1.01**20 + .995**20) / 2 - 1)
    assert report["scenarios"][1]["after"]["total_return"] < report["scenarios"][0]["after"]["total_return"]
    assert report["concurrent_exposure_fraction"] == 1
    with pytest.raises(ValueError, match="coverage"):
        evaluate_portfolio(frame.iloc[1:], candidate="candidate")
    with pytest.raises(ValueError):
        evaluate_portfolio(pd.concat([frame, frame.iloc[:1]]), candidate="candidate")
    frame.loc[0, "gross_return"] = np.nan
    with pytest.raises(ValueError):
        evaluate_portfolio(frame, candidate="candidate")


def test_html_escapes_source_content():
    candidate = candidate_review({"model_name": "<script>alert(1)</script>"})
    page = render({"candidates": [candidate], "portfolio": {}, "experiment_lifecycle": {}, "sources": []})
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page
