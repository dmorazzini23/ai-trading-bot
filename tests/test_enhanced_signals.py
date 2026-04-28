import numpy as np
import pytest
pd = pytest.importorskip("pandas")
try:
    import ai_trading.risk.engine as risk_engine  # AI-AGENT-REF: normalized import
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("risk_engine not available", allow_module_level=True)
from ai_trading import signals


def test_dynamic_position_size_scaling():
    s1 = risk_engine.dynamic_position_size(10000, volatility=0.02, drawdown=0.05)
    s2 = risk_engine.dynamic_position_size(10000, volatility=0.02, drawdown=0.15)
    assert s2 < s1 and s1 > 0


def test_signal_matrix_and_vote():
    close = np.linspace(100, 105, 30)
    df = pd.DataFrame({
        "close": close,
        "high": close * 1.01,
        "low": close * 0.99,
    })
    matrix = signals.compute_signal_matrix(df)
    assert not matrix.empty
    vote = signals.ensemble_vote_signals(matrix)
    assert len(vote) == len(matrix)


def test_signal_matrix_inverts_mean_reversion_zscore():
    close = np.array([100.0] * 25 + [120.0])
    df = pd.DataFrame(
        {
            "close": close,
            "high": close + 1.0,
            "low": close - 1.0,
        },
        index=pd.date_range("2026-04-27 14:30", periods=len(close), freq="min", tz="UTC"),
    )

    matrix = signals.compute_signal_matrix(df)

    assert matrix["mean_rev_z"].iloc[-1] < 0.0


def test_classify_regime_basic():
    df = pd.DataFrame({"close": np.linspace(100, 120, 40)})
    regime = signals.classify_regime(df)
    assert regime.iloc[-1] in {"trend", "mean_revert"}
