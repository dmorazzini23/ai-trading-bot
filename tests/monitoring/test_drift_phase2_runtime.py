from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from ai_trading.monitoring import drift


def test_drift_metrics_and_signal_attribution_serialize_datetimes() -> None:
    now = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
    metrics = drift.DriftMetrics("alpha", 0.12, "medium", 1.0, 1.2, 0.1, 0.2, 50, now)
    attribution = drift.SignalAttribution(
        "signal",
        0.05,
        0.6,
        1.2,
        0.3,
        0.1,
        4,
        5.0,
        0.05,
        now,
        now,
    )

    assert metrics.to_dict()["calculated_at"] == now.isoformat()
    assert attribution.to_dict()["period_start"] == now.isoformat()
    assert attribution.to_dict()["period_end"] == now.isoformat()


def test_update_baseline_loads_and_monitors_feature_drift(tmp_path: Path) -> None:
    monitor = drift.DriftMonitor(str(tmp_path), drift.AlertThreshold(psi_low=0.01, psi_medium=0.05))
    baseline = pd.DataFrame({"alpha": np.linspace(0.0, 1.0, 100), "beta": np.linspace(1.0, 2.0, 100)})

    monitor.update_baseline(baseline)
    reloaded = drift.DriftMonitor(str(tmp_path), drift.AlertThreshold(psi_low=0.01, psi_medium=0.05))
    metrics = reloaded.monitor_feature_drift(
        pd.DataFrame(
            {
                "alpha": np.linspace(0.6, 1.0, 50),
                "beta": np.linspace(1.1, 2.1, 50),
                "missing": np.linspace(0.0, 1.0, 50),
            }
        )
    )

    assert (tmp_path / "feature_baseline.json").exists()
    assert {item.feature_name for item in metrics} == {"alpha", "beta"}
    assert all(item.sample_size == 50 for item in metrics)
    assert any(item.drift_level in {"medium", "high"} for item in metrics)


def test_calculate_psi_handles_empty_constant_and_shifted_distributions(tmp_path: Path) -> None:
    monitor = drift.DriftMonitor(str(tmp_path))

    assert monitor.calculate_psi(np.array([]), np.array([1.0])) == 0.0
    assert monitor.calculate_psi(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.0
    shifted = monitor.calculate_psi(np.linspace(0.0, 1.0, 100), np.linspace(0.4, 1.0, 100))
    assert shifted > 0.0


def test_signal_attribution_empty_and_non_empty_paths(tmp_path: Path) -> None:
    monitor = drift.DriftMonitor(str(tmp_path))
    empty = monitor.calculate_signal_attribution("empty", pd.Series(dtype=float))

    assert empty.trade_count == 0
    assert empty.period_return == 0.0

    returns = pd.Series(
        [0.02, -0.01, 0.03, -0.02, 0.01],
        index=pd.date_range("2026-04-01", periods=5, tz="UTC"),
    )
    attribution = monitor.calculate_signal_attribution("alpha", returns)

    assert attribution.period_return != 0.0
    assert attribution.hit_ratio == pytest.approx(0.6)
    assert attribution.trade_count >= 1
    assert attribution.max_drawdown >= 0.0

    monitor.save_attribution_history(attribution)
    assert (tmp_path / "signal_attribution_alpha.json").exists()


def test_drift_summary_and_global_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monitor = drift.DriftMonitor(str(tmp_path))
    monkeypatch.setattr(drift, "_global_drift_monitor", monitor)
    metrics = [
        drift.DriftMetrics("low", 0.01, "low", 0, 0, 1, 1, 10, datetime.now(UTC)),
        drift.DriftMetrics("medium", 0.15, "medium", 0, 1, 1, 2, 10, datetime.now(UTC)),
        drift.DriftMetrics("high", 0.3, "high", 0, 2, 1, 3, 10, datetime.now(UTC)),
    ]

    summary = monitor.get_drift_summary(metrics)

    assert monitor.get_drift_summary([]) == {}
    assert summary["total_features"] == 3
    assert summary["features_with_alerts"] == ["medium", "high"]
    assert drift.get_drift_monitor() is monitor

    monitor._baseline_stats["x"] = {"mean": 0.0, "std": 1.0, "count": 100}
    result = drift.monitor_drift(pd.DataFrame({"x": [-1.0, 0.0, 1.0]}))
    assert result[0].feature_name == "x"


def test_shadow_mode_evaluates_and_logs_common_predictions(tmp_path: Path) -> None:
    shadow = drift.ShadowMode(str(tmp_path))

    empty = shadow.evaluate_shadow_model("model", {"AAPL": 0.1}, {"MSFT": 0.2})
    evaluation = shadow.evaluate_shadow_model(
        "model",
        {"AAPL": 0.2, "MSFT": -0.1, "TSLA": 0.5},
        {"AAPL": 0.1, "MSFT": -0.2},
        market_data={"regime": "normal"},
    )

    assert empty == {}
    assert evaluation["num_predictions"] == 2
    assert evaluation["differences"]["AAPL"] == pytest.approx(0.1)
    assert evaluation["market_context"] == {"regime": "normal"}
    assert (tmp_path / "shadow_model.jsonl").exists()


def test_shadow_mode_global_helper(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    shadow = drift.ShadowMode(str(tmp_path))
    monkeypatch.setattr(drift, "_global_shadow_mode", shadow)

    assert drift.get_shadow_mode() is shadow
