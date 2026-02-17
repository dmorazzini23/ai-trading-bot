import numpy as np
import pytest
pd = pytest.importorskip("pandas")
from ai_trading.rl_trading.features import FeatureConfig, compute_features


def test_compute_features_shape_and_finite():
    n = 100
    df = pd.DataFrame({
        "open":   np.linspace(100, 100.5, n),
        "high":   np.linspace(101, 101.5, n),
        "low":    np.linspace( 99,  99.5, n),
        "close":  np.linspace(100, 101.0, n),
        "volume": np.linspace(1e5, 1.1e5, n)
    })
    cfg = FeatureConfig(window=64)
    vec = compute_features(df, cfg)
    assert vec.shape == (64*6,)
    assert np.isfinite(vec).all()


def test_compute_features_accepts_window_kwarg():
    n = 80
    df = pd.DataFrame({
        "open": np.linspace(100, 100.5, n),
        "high": np.linspace(101, 101.5, n),
        "low": np.linspace(99, 99.5, n),
        "close": np.linspace(100, 101.0, n),
        "volume": np.linspace(1e5, 1.1e5, n),
    })
    vec = compute_features(df, window=32)
    assert vec.shape == (32 * 6,)
    assert np.isfinite(vec).all()


def test_compute_features_rejects_non_positive_window():
    n = 30
    df = pd.DataFrame({
        "open": np.linspace(100, 100.5, n),
        "high": np.linspace(101, 101.5, n),
        "low": np.linspace(99, 99.5, n),
        "close": np.linspace(100, 101.0, n),
        "volume": np.linspace(1e5, 1.1e5, n),
    })
    with pytest.raises(ValueError, match="Feature window must be positive"):
        compute_features(df, cfg=FeatureConfig(window=0))
