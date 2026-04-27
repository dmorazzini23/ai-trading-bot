from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from ai_trading.evaluation import walkforward as wf


def _evaluator(tmp_path: Path, **kwargs: Any) -> wf.WalkForwardEvaluator:
    params = {
        "mode": "rolling",
        "train_span": timedelta(days=3),
        "test_span": timedelta(days=2),
        "output_dir": str(tmp_path),
    }
    params.update(kwargs)
    return wf.WalkForwardEvaluator(**params)


def _market_frame(periods: int = 10) -> Any:
    dates = pd.date_range("2026-01-01", periods=periods, freq="D", tz=UTC)
    return pd.DataFrame(
        {
            "feature": np.linspace(1.0, float(periods), periods),
            "target": np.linspace(-0.02, 0.03, periods),
        },
        index=dates,
    )


class _BoundaryTrainer:
    instances: list["_BoundaryTrainer"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.best_params = {"deterministic": True}
        self.model = self
        _BoundaryTrainer.instances.append(self)

    def train(
        self,
        X_train: Any,
        y_train: Any,
        *,
        optimize_hyperparams: bool,
        feature_pipeline: Any,
    ) -> None:
        self.train_index = list(X_train.index)
        self.y_index = list(y_train.index)
        self.optimize_hyperparams = optimize_hyperparams
        self.feature_pipeline = feature_pipeline

    def predict(self, X_test: Any) -> np.ndarray:
        return np.asarray(X_test["feature"], dtype=float) / 100.0


def test_run_walkforward_respects_fold_boundaries_and_one_day_embargo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _market_frame()
    _BoundaryTrainer.instances.clear()
    monkeypatch.setattr(wf, "_get_ml_trainer", lambda: _BoundaryTrainer)
    monkeypatch.setattr(
        wf,
        "simulate_executed_trades",
        lambda **_kwargs: {
            "gross_return": 0.02,
            "net_return": 0.01,
            "cost_return": 0.001,
            "turnover_units": 1.0,
            "trade_count": 2.0,
            "signal_count": 2.0,
            "max_drawdown": 0.0,
            "hit_rate": 1.0,
        },
    )
    evaluator = _evaluator(tmp_path, embargo_pct=1 / 3)

    results = evaluator.run_walkforward(data, target_col="target", save_results=False)

    assert results["n_folds"] == 3
    assert results["final_equity"] == pytest.approx(100.0 * 1.01**3)
    assert results["aggregate_metrics"]["executed_trade_count"] == 6
    assert [fold["train_samples"] for fold in results["fold_results"]] == [3, 3, 3]
    assert [fold["test_samples"] for fold in results["fold_results"]] == [2, 2, 1]
    for fold in results["fold_results"]:
        train_end = pd.Timestamp(fold["train_end"])
        test_start = pd.Timestamp(fold["test_start"])
        assert test_start - train_end == pd.Timedelta(days=1)
    assert [trainer.kwargs["random_state"] for trainer in _BoundaryTrainer.instances] == [42, 43, 44]


@pytest.mark.parametrize(
    "kwargs,error_type",
    [
        ({"mode": None}, TypeError),
        ({"mode": "sideways"}, ValueError),
        ({"train_span": 0}, ValueError),
        ({"test_span": timedelta(0)}, ValueError),
        ({"train_span": object()}, TypeError),
        ({"embargo_pct": -0.01}, ValueError),
        ({"embargo_pct": 1.0}, ValueError),
        ({"embargo_pct": True}, TypeError),
    ],
)
def test_constructor_rejects_invalid_window_and_embargo_config(
    tmp_path: Path,
    kwargs: dict[str, Any],
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        _evaluator(tmp_path, **kwargs)


def test_run_walkforward_handles_empty_generated_folds_without_artifacts(tmp_path: Path) -> None:
    evaluator = _evaluator(tmp_path, train_span=timedelta(days=20), test_span=timedelta(days=5))

    results = evaluator.run_walkforward(_market_frame(periods=4), target_col="target", save_results=False)

    assert results["n_folds"] == 0
    assert results["aggregate_metrics"] == {}
    assert results["final_equity"] == 100.0
    assert results["total_return"] == 0.0
    assert results["max_drawdown"] == 0.0
    assert not list(tmp_path.glob("walkforward_*.csv"))
    assert not list(tmp_path.glob("walkforward_*.json"))
    assert not list(tmp_path.glob("equity_curve_*.csv"))


def test_invalid_fold_descriptor_returns_error_instead_of_keyerror(tmp_path: Path) -> None:
    data = _market_frame(periods=5)
    evaluator = _evaluator(tmp_path)

    result = evaluator._run_single_fold(
        data[["feature"]],
        data["target"],
        {"train_end": data.index[2], "test_start": data.index[3], "test_end": data.index[4]},
        model_type="ridge",
        feature_pipeline_params=None,
        fold_idx=7,
    )

    assert result["fold"] == 7
    assert "train_start" in result["error"]


def test_metric_aggregation_covers_constant_predictions_and_no_trade_metrics(
    tmp_path: Path,
) -> None:
    evaluator = _evaluator(tmp_path)
    evaluator.fold_results = [{"metrics": {"period_days": 4}}]
    evaluator.equity_curve = pd.Series(
        [100.0, 100.0],
        index=[datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 5, tzinfo=UTC)],
    )
    evaluator.drawdown_series = evaluator._calculate_drawdown(evaluator.equity_curve)

    metrics = evaluator._calculate_aggregate_metrics(
        predictions_all=[0.01, 0.01],
        actual_all=[0.02, -0.02],
        splits=[{"fold": 0}],
        fold_trade_metrics=[],
    )

    assert metrics["prediction_sharpe"] == 0.0
    assert metrics["prediction_sortino"] == 0.0
    assert metrics["executed_total_return"] == 0.0
    assert metrics["net_sharpe"] == 0.0
    assert metrics["turnover_annual"] == pytest.approx(126.0)


def test_metric_aggregation_compounds_returns_and_sortino_branches(tmp_path: Path) -> None:
    evaluator = _evaluator(tmp_path)
    evaluator.fold_results = [
        {"metrics": {"period_days": 2}},
        {"metrics": {"period_days": 2}},
        {"metrics": {"period_days": 2}},
    ]
    evaluator.equity_curve = pd.Series(
        [100.0, 97.0, 96.03, 100.8315],
        index=pd.date_range("2026-01-01", periods=4, freq="D", tz=UTC),
    )
    evaluator.drawdown_series = evaluator._calculate_drawdown(evaluator.equity_curve)

    metrics = evaluator._calculate_aggregate_metrics(
        predictions_all=[-0.04, -0.01, 0.07],
        actual_all=[-0.03, -0.02, 0.06],
        splits=[{"fold": 0}, {"fold": 1}, {"fold": 2}],
        fold_trade_metrics=[
            {"net_return": -0.04, "trade_count": 1, "turnover_units": 0.5, "cost_return": 0.001},
            {"net_return": -0.01, "trade_count": 2, "turnover_units": 0.75, "cost_return": 0.002},
            {"net_return": 0.07, "trade_count": 3, "turnover_units": 1.25, "cost_return": 0.003},
        ],
    )

    assert metrics["prediction_sortino"] != metrics["prediction_sharpe"]
    assert metrics["sortino_ratio"] != metrics["net_sharpe"]
    assert metrics["executed_total_return"] == pytest.approx((0.96 * 0.99 * 1.07) - 1.0)
    assert metrics["executed_trade_count"] == 6
    assert metrics["executed_turnover_units"] == 2.5
    assert metrics["executed_cost_return"] == pytest.approx(0.006)


def test_fold_metric_and_drawdown_error_branches(tmp_path: Path) -> None:
    evaluator = _evaluator(tmp_path)
    metrics = evaluator._calculate_fold_metrics(
        pd.Series([0.01, 0.02]),
        np.array([0.01]),
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 2, tzinfo=UTC),
    )

    class BadEquity:
        def __len__(self) -> int:
            return 1

        def expanding(self) -> Any:
            raise TypeError("bad equity")

    assert metrics == {}
    assert evaluator._calculate_drawdown(BadEquity()).empty


def test_create_plots_writes_expected_report_with_fake_matplotlib(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Axis:
        def __init__(self) -> None:
            self.xaxis = SimpleNamespace(
                set_major_formatter=lambda _formatter: None,
                set_major_locator=lambda _locator: None,
                get_majorticklabels=lambda: [],
            )

        def plot(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def fill_between(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def set_title(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def set_ylabel(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def set_xlabel(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def grid(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def axhline(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def legend(self) -> None:
            return None

    class FakePlot:
        def __init__(self) -> None:
            self.saved: list[Path] = []

        def subplots(self, *_args: Any, **_kwargs: Any) -> Any:
            fig = SimpleNamespace(suptitle=lambda *_a, **_kw: None)
            return fig, ((Axis(), Axis()), (Axis(), Axis()))

        def setp(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def tight_layout(self) -> None:
            return None

        def savefig(self, path: str, *_args: Any, **_kwargs: Any) -> None:
            plot_path = Path(path)
            plot_path.write_text("plot", encoding="utf-8")
            self.saved.append(plot_path)

        def close(self) -> None:
            return None

    fake_plot = FakePlot()
    monkeypatch.setattr(wf, "matplotlib_available", True)
    monkeypatch.setattr(wf, "plt", fake_plot)
    monkeypatch.setattr(
        wf,
        "mdates",
        SimpleNamespace(
            DateFormatter=lambda _fmt: object(),
            MonthLocator=lambda interval: ("month", interval),
        ),
    )
    evaluator = _evaluator(tmp_path)
    evaluator.equity_curve = pd.Series(
        [100.0, 101.0],
        index=pd.date_range("2026-01-01", periods=2, freq="D", tz=UTC),
    )
    evaluator.drawdown_series = evaluator._calculate_drawdown(evaluator.equity_curve)
    evaluator.fold_results = [
        {"metrics": {"correlation": 0.5, "directional_accuracy": 0.75}},
        {"metrics": {"correlation": -0.25, "directional_accuracy": 0.25}},
    ]

    evaluator._create_plots("fixed")

    assert fake_plot.saved == [tmp_path / "walkforward_plots_fixed.png"]
    assert fake_plot.saved[0].read_text(encoding="utf-8") == "plot"


def test_ensure_matplotlib_disabled_uses_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    import ai_trading.config as config

    monkeypatch.setattr(wf, "matplotlib_available", False)
    monkeypatch.setattr(config, "get_settings", lambda: SimpleNamespace(enable_plotting=False))

    wf._ensure_matplotlib()

    assert wf.matplotlib_available is False


def test_run_walkforward_reraises_split_type_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _evaluator(tmp_path)
    monkeypatch.setattr(
        wf,
        "walkforward_splits",
        lambda **_kwargs: (_ for _ in ()).throw(TypeError("bad split")),
    )

    with pytest.raises(TypeError, match="bad split"):
        evaluator.run_walkforward(_market_frame(periods=6), target_col="target", save_results=False)


def test_smoke_test_builds_expected_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(self: wf.WalkForwardEvaluator, **kwargs: Any) -> dict[str, Any]:
        data = kwargs["data"]
        assert len(data) == 1461
        assert kwargs["target_col"] == "target"
        assert kwargs["feature_cols"] == ["feature_1", "feature_2", "feature_3"]
        assert kwargs["model_type"] == "ridge"
        assert kwargs["save_results"] is True
        assert {"price", "feature_1", "feature_2", "feature_3", "target"} <= set(data.columns)
        return {
            "n_folds": 2,
            "total_return": 0.02,
            "max_drawdown": 0.01,
            "aggregate_metrics": {"net_sharpe": 1.5},
        }

    monkeypatch.setattr(wf.WalkForwardEvaluator, "run_walkforward", fake_run)

    results = wf.run_walkforward_smoke_test()

    assert results["aggregate_metrics"]["net_sharpe"] == 1.5
