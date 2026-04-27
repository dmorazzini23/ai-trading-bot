from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from ai_trading.training import train_ml as tm


def test_ml_trainer_default_params_suggestions_scores_and_model_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = tm.MLTrainer(model_type="ridge", cv_splits=2)
    monkeypatch.setattr(tm, "LIGHTGBM_AVAILABLE", True)
    monkeypatch.setitem(__import__("sys").modules, "lightgbm", SimpleNamespace(LGBMRegressor=lambda **kwargs: ("lgbm", kwargs)))
    monkeypatch.setitem(__import__("sys").modules, "xgboost", SimpleNamespace(XGBRegressor=lambda **kwargs: ("xgb", kwargs)))

    class Trial:
        def suggest_int(self, name: str, low: int, high: int) -> int:
            return low

        def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
            return low

        def suggest_categorical(self, name: str, choices):
            return choices[0]

    assert tm.MLTrainer(model_type="lightgbm")._get_default_params()["n_estimators"] == 500  # noqa: SLF001
    assert tm.MLTrainer(model_type="xgboost")._get_default_params()["max_depth"] == 6  # noqa: SLF001
    assert trainer._get_default_params()["alpha"] == 1.0  # noqa: SLF001
    assert tm.MLTrainer(model_type="stacking")._get_default_params() == {"meta_label_threshold": None}  # noqa: SLF001
    assert tm.MLTrainer(model_type="unknown")._get_default_params() == {}  # noqa: SLF001

    assert trainer._suggest_params(Trial())["alpha"] == 0.001  # noqa: SLF001
    assert tm.MLTrainer(model_type="lightgbm")._suggest_params(Trial())["num_leaves"] == 10  # noqa: SLF001
    assert tm.MLTrainer(model_type="xgboost")._suggest_params(Trial())["min_child_weight"] == 1  # noqa: SLF001
    assert tm.MLTrainer(model_type="unknown")._suggest_params(Trial()) == {}  # noqa: SLF001

    score = trainer._calculate_score(pd.Series([0.1, -0.2, 0.3]), np.array([0.2, -0.1, 0.4]))  # noqa: SLF001
    scaled_score = trainer._calculate_score(pd.Series([10.0, -20.0]), np.array([20.0, -10.0]))  # noqa: SLF001
    assert score > 0.0
    assert scaled_score > 0.0
    assert trainer._calculate_score(pd.Series(["bad"]), np.array([1.0])) == 0.0  # noqa: SLF001

    assert tm.MLTrainer(model_type="lightgbm")._create_model({"a": 1})[0] == "lgbm"  # noqa: SLF001
    assert tm.MLTrainer(model_type="xgboost")._create_model({"a": 1})[0] == "xgb"  # noqa: SLF001
    assert type(trainer._create_model({"alpha": 1.0})).__name__ == "Ridge"  # noqa: SLF001
    assert type(tm.MLTrainer(model_type="stacking")._create_model({"meta_label_threshold": 0.1})).__name__ == "StackingMetaModel"  # noqa: SLF001
    with pytest.raises(ValueError, match="Unknown model type"):
        tm.MLTrainer(model_type="unknown")._create_model({})  # noqa: SLF001


def test_ml_trainer_cv_fit_save_load_and_cli(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    trainer = tm.MLTrainer(model_type="ridge", cv_splits=2)
    X = pd.DataFrame({"a": np.linspace(0.0, 1.0, 20), "b": np.linspace(1.0, 2.0, 20)})
    y = pd.Series(np.linspace(-0.1, 0.1, 20))

    class Splitter:
        def split(self, *_args, **_kwargs):
            yield np.arange(0, 10), np.arange(10, 15)
            yield np.arange(5, 15), np.arange(15, 20)

    cv = trainer._evaluate_cv(X, y, Splitter(), {"alpha": 1.0})  # noqa: SLF001
    assert cv["n_splits"] == 2
    assert len(cv["fold_details"]) == 2

    trainer.model = trainer._create_model({"alpha": 1.0})  # noqa: SLF001
    trainer._fit_final_model(X, y)  # noqa: SLF001
    assert set(trainer.feature_importance) == {"a", "b"}

    dumped: dict[str, object] = {}
    monkeypatch.setattr(tm.joblib, "dump", lambda model, path: dumped.setdefault("path", path))
    monkeypatch.setattr(tm, "write_artifact_manifest", lambda **kwargs: dumped.setdefault("manifest", kwargs))
    model_base = tmp_path / "model"
    trainer.best_params = {"alpha": 1.0}
    trainer.cv_results = {"mean_score": 0.5}
    trainer.save_model(str(model_base), {"extra": "yes"})
    assert (tmp_path / "model_meta.json").exists()

    monkeypatch.setattr(tm, "load_verified_joblib_artifact", lambda path: {"loaded": str(path)})
    loaded, metadata = tm.MLTrainer.load_model(str(model_base))
    assert loaded["loaded"].endswith("model.joblib")
    assert metadata["extra"] == "yes"

    tm.train_model_cli([" aapl ", "AAPL", ""], model_type="ridge", dry_run=True)
    with pytest.raises(ValueError, match="symbol_list"):
        tm.train_model_cli(["", " "], model_type="ridge", dry_run=True)

    trained: list[str] = []
    monkeypatch.setitem(
        __import__("sys").modules,
        "ai_trading.model_loader",
        SimpleNamespace(train_and_save_model=lambda symbol, _dir: trained.append(symbol)),
    )
    monkeypatch.setitem(__import__("sys").modules, "ai_trading.paths", SimpleNamespace(MODELS_DIR=tmp_path))
    tm.train_model_cli(["AAPL", "MSFT"], model_type="ridge", dry_run=False, wf_smoke=False)
    assert trained == ["AAPL", "MSFT"]
