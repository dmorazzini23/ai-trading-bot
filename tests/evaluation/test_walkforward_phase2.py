from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from ai_trading.evaluation import walkforward as wf


def _evaluator(tmp_path: Path) -> wf.WalkForwardEvaluator:
    return wf.WalkForwardEvaluator(
        mode="rolling",
        train_span=timedelta(days=3),
        test_span=timedelta(days=2),
        output_dir=str(tmp_path),
    )


def test_fold_metrics_cover_directional_hits_and_single_prediction(tmp_path: Path) -> None:
    evaluator = _evaluator(tmp_path)
    y_true = pd.Series([0.02, -0.01, 0.03], index=pd.date_range("2026-01-01", periods=3))

    metrics = evaluator._calculate_fold_metrics(
        y_true,
        np.array([0.01, -0.02, -0.01]),
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 4, tzinfo=UTC),
    )

    assert metrics["n_long_signals"] == 1
    assert metrics["n_short_signals"] == 2
    assert metrics["hit_rate_long"] == 1.0
    assert metrics["hit_rate_short"] == 0.5
    assert metrics["period_days"] == 3

    single = evaluator._calculate_fold_metrics(
        pd.Series([0.01]),
        np.array([0.02]),
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 2, tzinfo=UTC),
    )
    assert single["correlation"] == 0.0


def test_simulate_fold_trades_returns_fallback_metrics_on_bad_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _evaluator(tmp_path)
    monkeypatch.setattr(wf, "simulate_executed_trades", lambda **_kwargs: (_ for _ in ()).throw(TypeError("bad")))

    metrics = evaluator._simulate_fold_trades(y_true=pd.Series([0.01]), y_pred=np.array([0.02]))

    assert metrics == {
        "gross_return": 0.0,
        "net_return": 0.0,
        "cost_return": 0.0,
        "turnover_units": 0.0,
        "trade_count": 0.0,
        "signal_count": 0.0,
        "max_drawdown": 0.0,
        "hit_rate": 0.0,
    }


def test_aggregate_metrics_include_prediction_and_executed_trade_statistics(tmp_path: Path) -> None:
    evaluator = _evaluator(tmp_path)
    evaluator.fold_results = [
        {"metrics": {"period_days": 2}},
        {"metrics": {"period_days": 3}},
    ]
    evaluator.equity_curve = pd.Series([100.0, 105.0, 103.0], index=pd.date_range("2026-01-01", periods=3))
    evaluator.drawdown_series = evaluator._calculate_drawdown(evaluator.equity_curve)
    splits: list[dict[str, Any]] = [{"fold": 0}, {"fold": 1}]

    metrics = evaluator._calculate_aggregate_metrics(
        [0.02, -0.01, 0.03],
        [0.01, -0.02, 0.04],
        splits,
        fold_trade_metrics=[
            {"net_return": 0.02, "trade_count": 2, "turnover_units": 1.5, "cost_return": 0.001},
            {"net_return": -0.01, "trade_count": 1, "turnover_units": 0.5, "cost_return": 0.002},
        ],
    )

    assert metrics["total_predictions"] == 3
    assert metrics["evaluation_period_days"] == 5
    assert metrics["executed_trade_count"] == 3
    assert metrics["executed_turnover_units"] == 2.0
    assert metrics["executed_cost_return"] == pytest.approx(0.003)
    assert metrics["executed_total_return"] == pytest.approx((1.02 * 0.99) - 1.0)
    assert metrics["max_drawdown"] > 0


def test_run_single_fold_uses_feature_pipeline_and_trainer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = pd.date_range("2026-01-01", periods=8, tz=UTC)
    X = pd.DataFrame({"feature": np.arange(8, dtype=float)}, index=dates)
    y = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01, 0.02, 0.04, -0.03], index=dates)
    split = {
        "train_start": dates[0],
        "train_end": dates[4],
        "test_start": dates[4],
        "test_end": dates[7],
    }

    class _Pipeline:
        def transform(self, frame: Any) -> Any:
            return frame

    class _Trainer:
        best_params = {"alpha": 1.0}

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.model = SimpleModel()

        def train(self, X_train: Any, y_train: Any, **kwargs: Any) -> None:
            assert len(X_train) == 4
            assert len(y_train) == 4
            assert kwargs["feature_pipeline"] is not None

    class SimpleModel:
        def predict(self, frame: Any) -> np.ndarray:
            return np.array([0.01, -0.01, 0.02][: len(frame)])

    monkeypatch.setattr(wf, "create_feature_pipeline", lambda **_kwargs: _Pipeline())
    monkeypatch.setattr(wf, "_get_ml_trainer", lambda: _Trainer)
    evaluator = _evaluator(tmp_path)

    result = evaluator._run_single_fold(
        X,
        y,
        split,
        model_type="ridge",
        feature_pipeline_params={"include_regime": False},
        fold_idx=2,
    )

    assert result["fold"] == 2
    assert result["train_samples"] == 4
    assert result["test_samples"] == 3
    assert result["predictions"] == [0.01, -0.01, 0.02]
    assert result["model_params"] == {"alpha": 1.0}
    assert "trade_metrics" in result


def test_run_single_fold_predicts_raw_features_for_pipeline_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = pd.date_range("2026-01-01", periods=8, tz=UTC)
    X = pd.DataFrame({"feature": np.arange(8, dtype=float)}, index=dates)
    y = pd.Series(np.linspace(-0.02, 0.02, len(dates)), index=dates)
    split = {
        "train_start": dates[0],
        "train_end": dates[4],
        "test_start": dates[4],
        "test_end": dates[7],
    }

    class _Pipeline:
        def transform(self, _frame: Any) -> Any:
            raise AssertionError("pipeline model should receive raw X_test")

    class _PipelineModel:
        def predict(self, frame: Any) -> np.ndarray:
            assert isinstance(frame, pd.DataFrame)
            return np.asarray(frame["feature"], dtype=float)

    class _Trainer:
        best_params: dict[str, float] = {}

        def __init__(self, **_kwargs: Any) -> None:
            self.model = _PipelineModel()

        def train(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    monkeypatch.setattr(wf, "create_feature_pipeline", lambda **_kwargs: _Pipeline())
    monkeypatch.setattr(wf, "_get_ml_trainer", lambda: _Trainer)
    monkeypatch.setattr(wf, "_is_sklearn_pipeline", lambda _model: True)

    result = _evaluator(tmp_path)._run_single_fold(
        X,
        y,
        split,
        model_type="ridge",
        feature_pipeline_params={"include_regime": False},
        fold_idx=0,
    )

    assert result["predictions"] == [4.0, 5.0, 6.0]


def test_save_results_writes_csv_and_json_when_plotting_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator = _evaluator(tmp_path)
    evaluator.fold_results = [
        {
            "fold": 0,
            "train_start": "2026-01-01",
            "test_start": "2026-01-04",
            "train_samples": 3,
            "test_samples": 2,
            "metrics": {"mse": 0.1},
        }
    ]
    evaluator.aggregate_results = {"net_sharpe": 1.2}
    evaluator.equity_curve = pd.Series([100.0, 101.0], index=pd.date_range("2026-01-01", periods=2))
    monkeypatch.setattr(wf, "matplotlib_available", False)
    monkeypatch.setattr(wf, "_ensure_matplotlib", lambda: None)

    evaluator._save_results()

    assert list(tmp_path.glob("walkforward_folds_*.csv"))
    assert list(tmp_path.glob("walkforward_aggregate_*.json"))
    assert list(tmp_path.glob("equity_curve_*.csv"))
    assert not list(tmp_path.glob("walkforward_plots_*.png"))
