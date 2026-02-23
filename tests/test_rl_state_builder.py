from __future__ import annotations

import numpy as np
import pytest

from ai_trading.rl_trading.state_builder import MarketStateBuilder, StateBuilderConfig


def _ohlcv_matrix(rows: int = 120) -> np.ndarray:
    close = np.linspace(100.0, 105.0, rows, dtype=np.float32)
    open_px = close - 0.15
    high = close + 0.25
    low = close - 0.35
    volume = np.linspace(1_000.0, 3_000.0, rows, dtype=np.float32)
    return np.column_stack([open_px, high, low, close, volume]).astype(np.float32)


def test_state_builder_ohlcv_feature_path_has_expected_shape():
    builder = MarketStateBuilder(StateBuilderConfig(use_ohlcv_features=True, normalize=True))
    states = builder.fit_transform(_ohlcv_matrix())
    assert states.shape[1] == 6
    assert np.isfinite(states).all()
    assert builder.describe()["schema"] == "ohlcv"


def test_state_builder_raw_path_preserves_column_count():
    data = np.random.default_rng(7).normal(size=(80, 4)).astype(np.float32)
    builder = MarketStateBuilder(StateBuilderConfig(use_ohlcv_features=False, normalize=True))
    states = builder.fit_transform(data)
    assert states.shape == data.shape
    assert np.isfinite(states).all()


def test_state_builder_transform_requires_fit():
    builder = MarketStateBuilder()
    with pytest.raises(RuntimeError, match="not fitted"):
        builder.transform(_ohlcv_matrix(40))


def test_state_builder_transform_uses_fitted_stats():
    train = _ohlcv_matrix(100)
    eval_data = _ohlcv_matrix(60) * 1.02
    builder = MarketStateBuilder(StateBuilderConfig(use_ohlcv_features=True, normalize=True))
    train_states = builder.fit_transform(train)
    eval_states = builder.transform(eval_data)

    assert train_states.shape[1] == eval_states.shape[1]
    assert np.isfinite(eval_states).all()
    desc = builder.describe()
    assert desc["fitted"] is True
    assert desc["feature_count"] == train_states.shape[1]
