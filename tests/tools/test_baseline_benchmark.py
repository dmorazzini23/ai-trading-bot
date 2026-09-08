import numpy as np
import pandas as pd
import pytest

from ai_trading.tools.baseline_benchmark import benchmark


def fixture():
    timestamps = pd.date_range("2026-04-01", periods=11, tz="UTC")
    data = pd.DataFrame({"timestamp": timestamps, "gross_long_bps": [1.] * 10 + [10000.], "round_trip_cost_bps": 6., "net_long_bps": [-5.] * 10 + [9994.], "sma_spread": [1., -1.] * 5 + [1.]})
    folds = [{"fold_index": i + 1, "test_start": str(timestamps[2*i]), "test_end": str(timestamps[2*i+1]), "test_rows": 2, "total_post_cost_net_edge_bps": 0, "selected_candidates": 0} for i in range(5)]
    return data, {"walk_forward": {"folds": folds}}


def test_fixed_baselines_use_saved_windows_costs_and_do_not_evaluate_holdout():
    data, candidate = fixture()
    report = benchmark(data, candidate)
    decisions = {row["name"]: row for row in report["decisions"]}
    assert decisions["always_long"]["net_edge_per_opportunity_bps"] == -5
    assert decisions["momentum"]["net_edge_per_opportunity_bps"] == -2.5
    assert decisions["cash"]["net_edge_per_opportunity_bps"] == 0
    assert decisions["always_long"]["decision"] == "fails_benchmark_criterion"
    assert not report["holdout_evaluated"]
    assert report["models_fitted"] == 0
    assert report["controlled_comparisons"]["scope"] == "development_outer_folds_only"


def test_mismatched_coverage_overlapping_folds_and_bad_costs_fail_closed():
    data, candidate = fixture()
    with pytest.raises(ValueError, match="coverage"):
        benchmark(data.iloc[1:], candidate)
    candidate["walk_forward"]["folds"][1]["test_start"] = candidate["walk_forward"]["folds"][0]["test_end"]
    with pytest.raises(ValueError, match="overlap"):
        benchmark(data, candidate)
    data, candidate = fixture()
    data.loc[0, "net_long_bps"] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        benchmark(data, candidate)
