"""
Stacking and meta-labeling models for time-series classification/regression.

Implements out-of-fold stacking with purged time-series splits to avoid leakage.
Optionally applies meta-labeling by training a classifier to gate base signals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from ai_trading.logging import logger
from ai_trading.data.splits import PurgedGroupTimeSeriesSplit


def _safe_import_sklearn():
    try:
        from sklearn.linear_model import Ridge, LogisticRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
    except Exception as exc:  # pragma: no cover
        raise ImportError("scikit-learn is required for stacking models") from exc
    return Ridge, LogisticRegression, RandomForestRegressor, StandardScaler, make_pipeline


@dataclass
class StackingMetaModel:
    """
    Out-of-fold stacking model with optional meta-label gating.

    - Base learners: ridge regression and random forest (light footprints).
    - Meta learner: ridge for regression; logistic for meta-label gating.
    - Meta labeling: if `meta_label_threshold` set, train a classifier to
      predict acceptance; final prediction = acceptance_prob * meta_base_pred.
    """

    cv_splits: int = 5
    embargo_pct: float = 0.01
    purge_pct: float = 0.02
    meta_label_threshold: float | None = None

    def __post_init__(self) -> None:
        (Ridge, LogisticRegression, RF, StandardScaler, make_pipeline) = _safe_import_sklearn()
        self._Ridge = Ridge
        self._LogisticRegression = LogisticRegression
        self._RF = RF
        self._StandardScaler = StandardScaler
        self._make_pipeline = make_pipeline
        self.base_models_: list[Any] = []
        self.meta_model_: Any | None = None
        self._fitted = False

    def _make_base_models(self) -> list[Any]:
        ridge = self._make_pipeline(self._StandardScaler(with_mean=True), self._Ridge(alpha=1.0))
        rf = self._RF(n_estimators=200, max_depth=5, random_state=42)
        return [ridge, rf]

    def fit(self, X, y) -> "StackingMetaModel":
        try:
            X = X.copy()
            y = y.copy()
            splitter = PurgedGroupTimeSeriesSplit(
                n_splits=self.cv_splits, embargo_pct=self.embargo_pct, purge_pct=self.purge_pct
            )
            bases = self._make_base_models()
            oof_meta = np.zeros((len(X), len(bases)), dtype=float)
            # OOF predictions for meta features
            for bi, base in enumerate(bases):
                for train_idx, test_idx in splitter.split(X, y):
                    base.fit(X.iloc[train_idx], y.iloc[train_idx])
                    preds = base.predict(X.iloc[test_idx])
                    oof_meta[test_idx, bi] = preds
            # Meta learner
            meta_X = oof_meta
            meta_y = y.values if hasattr(y, "values") else np.asarray(y)
            if self.meta_label_threshold is not None:
                # Binary acceptance target based on threshold on y
                target = (meta_y > float(self.meta_label_threshold)).astype(int)
                self.meta_model_ = self._LogisticRegression(max_iter=500)
                self.meta_model_.fit(meta_X, target)
            else:
                self.meta_model_ = self._make_pipeline(
                    self._StandardScaler(with_mean=True), self._Ridge(alpha=0.5)
                )
                self.meta_model_.fit(meta_X, meta_y)
            # Fit base models on all data for prediction time
            self.base_models_ = []
            for base in bases:
                m = base.__class__
                # Recreate fresh instance to avoid data leakage
                fitted = base
                try:
                    fitted = m() if m is not None else base
                except Exception:
                    fitted = base
                fitted.fit(X, y)
                self.base_models_.append(fitted)
            self._fitted = True
            return self
        except Exception as e:
            logger.error("STACKING_FIT_FAILED", extra={"cause": e.__class__.__name__, "detail": str(e)})
            raise

    def predict(self, X) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("StackingMetaModel not fitted")
        try:
            meta_X = np.column_stack([m.predict(X) for m in self.base_models_])
            if self.meta_label_threshold is not None:
                prob = self.meta_model_.predict_proba(meta_X)[:, 1]
                base_avg = meta_X.mean(axis=1)
                return prob * base_avg
            else:
                return self.meta_model_.predict(meta_X)
        except Exception as e:
            logger.error("STACKING_PREDICT_FAILED", extra={"cause": e.__class__.__name__, "detail": str(e)})
            raise

