from __future__ import annotations

from ai_trading.execution.cost_model import CostModel


def test_cost_model_calibration_updates_params_with_sufficient_samples() -> None:
    model = CostModel()
    records = []
    for idx in range(120):
        records.append(
            {
                "is_bps": 4.0 + (idx % 11),
                "spread_paid_bps": 1.0 + (idx % 5) * 0.4,
                "volatility_pct": 0.01 + (idx % 3) * 0.002,
            }
        )

    updated = model.calibrate(records, min_samples=80, quantile=0.55, outlier_bps=120.0)
    assert updated.sample_count >= 80
    assert updated.version.startswith("calibrated-")

    estimate = model.estimate_cost_bps(
        spread_bps=10.0,
        volatility_pct=0.02,
        participation_rate=0.03,
        tca_cost_bps=7.0,
    )
    assert updated.min_bps <= estimate <= updated.max_bps


def test_cost_model_round_trip_persistence(tmp_path) -> None:
    target = tmp_path / "cost_model.json"
    model = CostModel()
    model.calibrate(
        [{"is_bps": 6.0 + float(i % 4), "spread_paid_bps": 2.0} for i in range(90)],
        min_samples=20,
    )
    model.save(target)
    loaded = CostModel.load(target)
    assert loaded.params.version == model.params.version
    assert loaded.params.base_cost_bps == model.params.base_cost_bps

