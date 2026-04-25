from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from ai_trading.training import train_ml
from ai_trading.training.train_ml import MLTrainer


class _Trial:
    def suggest_int(self, name: str, low: int, high: int) -> int:
        del high
        return low + (1 if name == "max_depth" else 0)

    def suggest_float(self, name: str, low: float, high: float, *, log: bool = False) -> float:
        del name, high, log
        return low

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        del name
        return choices[-1]


def test_default_and_suggested_params_cover_supported_model_types() -> None:
    ridge = MLTrainer(model_type="ridge", random_state=7)
    assert ridge._get_default_params() == {
        "alpha": 1.0,
        "fit_intercept": True,
        "solver": "auto",
        "random_state": 7,
    }
    assert ridge._suggest_params(_Trial()) == {
        "alpha": 0.001,
        "fit_intercept": False,
        "solver": "lsqr",
    }

    stacking = MLTrainer(model_type="stacking")
    assert stacking._get_default_params() == {"meta_label_threshold": None}
    assert stacking._suggest_params(_Trial()) == {}


def test_score_handles_scaled_predictions_nan_correlation_and_invalid_inputs() -> None:
    trainer = MLTrainer(model_type="ridge")

    assert trainer._calculate_score(pd.Series([1.0, -1.0]), np.array([100.0, -100.0])) == pytest.approx(1.0)
    assert trainer._calculate_score(pd.Series([0.01, -0.01]), np.array([-0.02, 0.02])) == pytest.approx(0.4)
    assert trainer._calculate_score(pd.Series([0.01]), np.array(["bad"], dtype=object)) == 0.0


def test_evaluate_cv_and_fit_final_model_capture_ridge_feature_importance() -> None:
    trainer = MLTrainer(model_type="ridge", cv_splits=2)
    X = pd.DataFrame(
        {
            "a": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
        }
    )
    y = pd.Series([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])

    class _Splitter:
        def split(self, *_args: Any, **_kwargs: Any) -> list[tuple[np.ndarray, np.ndarray]]:
            return [
                (np.array([0, 1, 2]), np.array([3, 4])),
                (np.array([0, 1, 2, 3]), np.array([4, 5])),
            ]

    params = trainer._get_default_params()
    cv_results = trainer._evaluate_cv(X, y, cast(Any, _Splitter()), params)
    trainer.model = trainer._create_model(params)
    trainer._fit_final_model(X, y)

    assert cv_results["n_splits"] == 2
    assert len(cv_results["fold_scores"]) == 2
    assert {detail["fold"] for detail in cv_results["fold_details"]} == {0, 1}
    assert set(trainer.feature_importance) == {"a", "b"}


def test_train_applies_feature_pipeline_and_returns_training_summary() -> None:
    trainer = MLTrainer(model_type="ridge", cv_splits=2)
    X = pd.DataFrame({"a": np.arange(12, dtype=float), "b": np.arange(12, dtype=float) * 2})
    y = pd.Series(np.arange(12, dtype=float) / 10.0)

    class _Pipeline:
        fitted = False

        def fit_transform(self, frame: Any, target: Any) -> Any:
            assert len(frame) == len(target)
            self.fitted = True
            return frame.assign(c=frame["a"] + frame["b"])

    pipeline = _Pipeline()
    result = trainer.train(X, y, optimize_hyperparams=False, feature_pipeline=pipeline)

    assert pipeline.fitted is True
    assert result["model_type"] == "ridge"
    assert result["feature_count"] == 3
    assert result["train_samples"] == 12
    assert result["cv_metrics"]["n_splits"] == 2


def test_save_model_rejects_paths_outside_allowed_dirs(tmp_path: Path) -> None:
    del tmp_path
    trainer = MLTrainer(model_type="ridge")
    outside = Path("/var/lib/ai-trading-not-allowed/model")

    with pytest.raises(RuntimeError, match="Model path not allowed"):
        trainer.save_model(str(outside))


def test_train_model_cli_normalizes_symbols_and_reports_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def _train_and_save(symbol: str, models_dir: Path) -> None:
        del models_dir
        calls.append(symbol)
        if symbol == "MSFT":
            raise RuntimeError("boom")

    monkeypatch.setattr("ai_trading.model_loader.train_and_save_model", _train_and_save)
    monkeypatch.setattr("ai_trading.paths.MODELS_DIR", Path("/tmp/models"))

    train_ml.train_model_cli([" aapl ", "AAPL", "", "msft"], model_type="ridge")

    assert calls == ["AAPL", "MSFT"]


def test_train_model_cli_rejects_empty_and_all_failed_symbol_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="symbol_list"):
        train_ml.train_model_cli([" ", ""], model_type="ridge")

    monkeypatch.setattr(
        "ai_trading.model_loader.train_and_save_model",
        lambda symbol, models_dir: (_ for _ in ()).throw(RuntimeError(f"{symbol} failed")),
    )
    monkeypatch.setattr("ai_trading.paths.MODELS_DIR", Path("/tmp/models"))

    with pytest.raises(RuntimeError, match="failed for all symbols"):
        train_ml.train_model_cli(["AAPL"], model_type="ridge")
