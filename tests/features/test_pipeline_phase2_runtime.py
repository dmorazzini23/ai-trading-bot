from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from ai_trading.features import pipeline as pipe_mod


def _market_frame(rows: int = 80) -> Any:
    index = pd.date_range("2026-01-01", periods=rows, freq="D")
    close = np.linspace(100.0, 120.0, rows) + np.sin(np.arange(rows))
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(1_000, 2_000, rows),
        },
        index=index,
    )


def test_build_features_fit_records_training_only_parameters() -> None:
    frame = _market_frame()
    builder = pipe_mod.BuildFeatures(lookback_window=7, vol_span=12, regime_span=20)

    fitted = builder.fit(frame)

    assert fitted is builder
    assert builder.is_fitted_ is True
    assert builder.feature_params_["lookback_window"] == 7
    assert builder.feature_params_["vol_span"] == 12
    assert builder.feature_params_["regime_span"] == 20
    assert builder.feature_params_["columns"] == ["open", "high", "low", "close", "volume"]
    assert "regime_vol_low" in builder.feature_params_
    assert "regime_vol_high" in builder.feature_params_


def test_build_features_transform_adds_full_feature_family() -> None:
    frame = _market_frame()
    builder = pipe_mod.BuildFeatures(vol_span=10, regime_span=20).fit(frame)

    transformed = builder.transform(frame)

    expected_columns = {
        "ret_1d",
        "ret_10d",
        "log_ret_1d",
        "cum_ret_20d",
        "ret_momentum",
        "vol_5d",
        "vol_ewma_10",
        "vol_rank",
        "hl_vol",
        "hl_vol_ma",
        "vol_ma_10",
        "obv",
        "vwap_dev",
        "vol_regime",
        "trend_slope",
        "trend_regime",
        "microstructure_dev",
    }
    assert expected_columns <= set(transformed.columns)
    assert len(transformed) == len(frame)


def test_build_features_supports_price_column_without_volume_or_hilo() -> None:
    frame = pd.DataFrame({"price": np.linspace(10.0, 15.0, 40)})
    builder = pipe_mod.BuildFeatures(include_volume=True, regime_span=10).fit(frame)

    transformed = builder.transform(frame)

    assert "ret_1d" in transformed
    assert "microstructure_dev" not in transformed


def test_build_features_regime_without_volatility_computes_returns_for_fit() -> None:
    frame = _market_frame()
    builder = pipe_mod.BuildFeatures(include_volatility=False, include_regime=True, regime_span=10)

    builder.fit(frame)
    transformed = builder.transform(frame)

    assert "regime_vol_low" in builder.feature_params_
    assert "vol_regime" in transformed
    assert "vol_20d" not in transformed


def test_build_features_error_paths_raise_for_unfit_or_bad_fit() -> None:
    builder = pipe_mod.BuildFeatures()

    with pytest.raises(ValueError, match="fitted"):
        builder.transform(_market_frame())

    with pytest.raises(ValueError, match="DataFrame"):
        builder.fit([1, 2, 3])  # type: ignore[arg-type]


def test_create_feature_pipeline_uses_requested_scaler_and_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Pipeline:
        def __init__(self, steps: list[tuple[str, Any]]) -> None:
            self.steps = steps

    class StandardScaler:
        pass

    class RobustScaler:
        pass

    monkeypatch.setattr(pipe_mod, "sklearn_available", True)
    monkeypatch.setattr(
        pipe_mod,
        "load_sklearn_preprocessing",
        lambda: SimpleNamespace(StandardScaler=StandardScaler, RobustScaler=RobustScaler),
    )
    monkeypatch.setattr(
        pipe_mod,
        "load_sklearn_pipeline",
        lambda: SimpleNamespace(Pipeline=Pipeline),
    )

    standard = pipe_mod.create_feature_pipeline("standard", {"regime_span": 10})
    robust = pipe_mod.create_feature_pipeline("robust", {"regime_span": 10})
    none = pipe_mod.create_feature_pipeline("none", {"regime_span": 10})
    fallback = pipe_mod.create_feature_pipeline("surprise", {"regime_span": 10})

    assert [name for name, _step in standard.steps] == ["features", "scaler"]
    assert type(standard.steps[-1][1]).__name__ == "StandardScaler"
    assert type(robust.steps[-1][1]).__name__ == "RobustScaler"
    assert [name for name, _step in none.steps] == ["features"]
    assert type(fallback.steps[-1][1]).__name__ == "StandardScaler"


def test_validate_pipeline_no_leakage_flags_identical_stats_and_accepts_shifted_data() -> None:
    same_train = pd.DataFrame({"close": np.linspace(1.0, 2.0, 30)})
    same_test = same_train.copy()
    shifted_test = pd.DataFrame({"close": np.linspace(50.0, 80.0, 30)})

    assert pipe_mod.validate_pipeline_no_leakage(
        pipe_mod.create_feature_pipeline("none", {"regime_span": 5}),
        same_train,
        same_test,
    ) is False
    assert pipe_mod.validate_pipeline_no_leakage(
        pipe_mod.create_feature_pipeline("none", {"regime_span": 5}),
        same_train,
        shifted_test,
    ) is True


def test_validate_pipeline_no_leakage_returns_false_on_pipeline_failure() -> None:
    class BrokenPipeline:
        def fit(self, *_args: Any, **_kwargs: Any) -> None:
            raise ValueError("bad fit")

    assert pipe_mod.validate_pipeline_no_leakage(
        BrokenPipeline(),
        _market_frame(),
        _market_frame(),
    ) is False
