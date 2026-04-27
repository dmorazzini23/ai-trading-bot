from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from ai_trading.evaluation import walkforward as wf


class _Trainer:
    instances: list["_Trainer"] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.best_params = {"alpha": 1.0}
        self.model = SimpleNamespace(predict=lambda X: np.asarray(X.iloc[:, 0], dtype=float) * 0.5)
        _Trainer.instances.append(self)

    def train(self, X, y, optimize_hyperparams: bool = False, feature_pipeline=None) -> None:
        self.train_shape = X.shape
        self.optimize_hyperparams = optimize_hyperparams
        self.feature_pipeline = feature_pipeline


class _Pipeline:
    def transform(self, X):
        return X


def test_walkforward_run_single_fold_aggregate_save_and_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    dates = pd.date_range("2026-01-01", periods=12, freq="D", tz=UTC)
    data = pd.DataFrame({"feature": np.linspace(-0.02, 0.03, len(dates)), "target": np.linspace(-0.01, 0.02, len(dates))}, index=dates)
    split = {
        "train_start": dates[0],
        "train_end": dates[6],
        "test_start": dates[6],
        "test_end": dates[-1] + timedelta(days=1),
    }

    monkeypatch.setattr(wf, "walkforward_splits", lambda **_kwargs: [split])
    monkeypatch.setattr(wf, "_get_ml_trainer", lambda: _Trainer)
    monkeypatch.setattr(wf, "create_feature_pipeline", lambda **_kwargs: _Pipeline())
    monkeypatch.setattr(
        wf,
        "simulate_executed_trades",
        lambda **_kwargs: {
            "gross_return": 0.02,
            "net_return": 0.01,
            "cost_return": 0.001,
            "turnover_units": 2.0,
            "trade_count": 3.0,
            "signal_count": 4.0,
            "max_drawdown": 0.01,
            "hit_rate": 0.75,
        },
    )
    monkeypatch.setattr(wf, "_ensure_matplotlib", lambda: None)
    monkeypatch.setattr(wf, "matplotlib_available", False)

    evaluator = wf.WalkForwardEvaluator(mode="anchored", train_span=5, test_span=2, output_dir=str(tmp_path))
    results = evaluator.run_walkforward(
        data=data,
        target_col="target",
        feature_cols=["feature"],
        feature_pipeline_params={"scaler": "none"},
        save_results=True,
    )

    assert results["n_folds"] == 1
    assert results["final_equity"] == pytest.approx(101.0)
    assert results["aggregate_metrics"]["executed_trade_count"] == 3
    assert evaluator.drawdown_series.notna().all()
    assert list(tmp_path.glob("walkforward_folds_*.csv"))
    assert list(tmp_path.glob("walkforward_aggregate_*.json"))
    assert list(tmp_path.glob("equity_curve_*.csv"))

    monkeypatch.setattr(
        wf.WalkForwardEvaluator,
        "run_walkforward",
        lambda self, **_kwargs: {"n_folds": 1, "total_return": 0.01, "max_drawdown": 0.0, "aggregate_metrics": {"net_sharpe": 1.2}},
    )
    smoke = wf.run_walkforward_smoke_test()
    assert smoke["aggregate_metrics"]["net_sharpe"] == 1.2


def test_walkforward_error_and_metric_branches(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    evaluator = wf.WalkForwardEvaluator(output_dir=str(tmp_path))
    dates = pd.date_range("2026-01-01", periods=4, freq="D", tz=UTC)
    X = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]}, index=dates)
    y = pd.Series([1.0, -1.0, 1.0, -1.0], index=dates)
    empty_split = {
        "train_start": dates[0],
        "train_end": dates[0],
        "test_start": dates[2],
        "test_end": dates[3],
    }
    assert evaluator._run_single_fold(X, y, empty_split, "ridge", None, 0) == {"error": "insufficient_data"}  # noqa: SLF001

    metrics = evaluator._calculate_fold_metrics(  # noqa: SLF001
        y,
        np.array([1.0, -0.5, 0.5, -1.0]),
        dates[0].to_pydatetime(),
        dates[-1].to_pydatetime(),
    )
    assert metrics["directional_accuracy"] == 1.0
    assert metrics["n_long_signals"] == 2
    assert metrics["n_short_signals"] == 2

    monkeypatch.setattr(wf, "simulate_executed_trades", lambda **_kwargs: (_ for _ in ()).throw(ValueError("bad sim")))
    fallback = evaluator._simulate_fold_trades(y_true=y, y_pred=np.array([1.0, -1.0]))  # noqa: SLF001
    assert fallback["trade_count"] == 0.0

    evaluator.equity_curve = pd.Series([100.0, 80.0, 120.0], index=dates[:3])
    evaluator.drawdown_series = evaluator._calculate_drawdown(evaluator.equity_curve)  # noqa: SLF001
    evaluator.fold_results = [{"metrics": {"period_days": 2}}]
    aggregate = evaluator._calculate_aggregate_metrics(  # noqa: SLF001
        [0.1, -0.2, 0.3],
        [0.1, -0.1, 0.2],
        [{"test_start": dates[0], "test_end": dates[1]}],
        fold_trade_metrics=[{"net_return": 0.01, "trade_count": 2, "turnover_units": 1.5, "cost_return": 0.001}],
    )
    assert aggregate["total_predictions"] == 3
    assert aggregate["executed_trade_count"] == 2
    assert evaluator._calculate_aggregate_metrics([], [], [], fold_trade_metrics=[]) == {}  # noqa: SLF001
