from __future__ import annotations

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")
sklearn_base = pytest.importorskip("sklearn.base")
BaseEstimator = sklearn_base.BaseEstimator
RegressorMixin = sklearn_base.RegressorMixin

from ai_trading.training.stacking import StackingMetaModel


class _ConstantBase:
    def fit(self, X, y):
        del y
        self._value = float(len(X))
        return self

    def predict(self, X):
        return np.full(len(X), self._value, dtype=float)


class _RecordingBase(BaseEstimator, RegressorMixin):
    records: list[tuple[int, int]] = []

    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def fit(self, X, y):
        del y
        self._train_max = int(X.index.max())
        return self

    def predict(self, X):
        self.records.append((self._train_max, int(X.index.min())))
        return np.full(len(X), self.value, dtype=float)


def test_stacking_meta_model_uses_only_rows_with_oof_predictions() -> None:
    seen_meta_rows: list[int] = []

    class _Meta:
        def fit(self, X, y):
            seen_meta_rows.append(len(X))
            assert len(X) == len(y)
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=float)

    model = StackingMetaModel(cv_splits=2, embargo_pct=0.0, purge_pct=0.0)
    model._make_base_models = lambda: [_ConstantBase(), _ConstantBase()]  # noqa: SLF001
    model._make_pipeline = lambda *_args, **_kwargs: _Meta()  # noqa: SLF001

    X = pd.DataFrame({"feature": np.arange(30, dtype=float)})
    y = pd.Series(np.arange(30, dtype=float))

    model.fit(X, y)

    assert seen_meta_rows == [20]


def test_stacking_oof_splits_leave_next_bar_label_gap() -> None:
    _RecordingBase.records = []
    model = StackingMetaModel(cv_splits=2, embargo_pct=0.0, purge_pct=0.0)
    model._make_base_models = lambda: [_RecordingBase(), _RecordingBase()]  # noqa: SLF001

    X = pd.DataFrame({"feature": np.arange(30, dtype=float)})
    y = pd.Series(np.arange(30, dtype=float))

    model.fit(X, y)

    assert _RecordingBase.records
    assert all(train_max + 1 < test_min for train_max, test_min in _RecordingBase.records)


def test_stacking_final_base_learners_clone_and_preserve_params() -> None:
    originals = [_RecordingBase(value=7.0), _RecordingBase(value=11.0)]
    model = StackingMetaModel(cv_splits=2, embargo_pct=0.0, purge_pct=0.0)
    model._make_base_models = lambda: originals  # noqa: SLF001

    X = pd.DataFrame({"feature": np.arange(30, dtype=float)})
    y = pd.Series(np.arange(30, dtype=float))

    model.fit(X, y)

    assert model.base_models_[0] is not originals[0]
    assert model.base_models_[1] is not originals[1]
    assert model.base_models_[0].value == 7.0
    assert model.base_models_[1].value == 11.0


def test_stacking_meta_label_one_class_target_fits_explicit_gate() -> None:
    model = StackingMetaModel(
        cv_splits=2,
        embargo_pct=0.0,
        purge_pct=0.0,
        meta_label_threshold=100.0,
    )
    model._make_base_models = lambda: [_ConstantBase(), _ConstantBase()]  # noqa: SLF001

    X = pd.DataFrame({"feature": np.arange(30, dtype=float)})
    y = pd.Series(np.arange(30, dtype=float))

    model.fit(X, y)
    preds = model.predict(X.iloc[:3])

    assert np.all(preds == 0.0)
