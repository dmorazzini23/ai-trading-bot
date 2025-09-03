import sys

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.core.bot_engine import SignalManager


def _minimal_df() -> pd.DataFrame:
    """Return a DataFrame with required features for ML and VSA heuristics."""
    data = {
        "rsi": [30.0] * 20,
        "macd": [0.1] * 20,
        "atr": [1.0] * 20,
        "vwap": [1.0] * 20,
        "sma_50": [1.0] * 20,
        "sma_200": [1.0] * 20,
        "close": [1.0] * 20,
        "open": [1.0] * 20,
        "volume": [1.0] * 20,
    }
    return pd.DataFrame(data)


def test_signal_ml_with_dummy_model(monkeypatch, caplog):
    """Calling signal_ml with a dummy model should emit no warnings."""
    monkeypatch.setenv("AI_TRADING_MODEL_MODULE", "dummy_model")

    dummy_model = sys.modules["dummy_model"].get_model()
    manager = SignalManager()
    df = _minimal_df()

    caplog.set_level("WARNING")
    result = manager.signal_ml(df, model=dummy_model)

    assert result is not None
    assert "ML predictions disabled" not in caplog.text


def test_signal_ml_warns_once(monkeypatch, caplog):
    """Missing models trigger a single warning and fall back to VSA."""
    monkeypatch.delenv("AI_TRADING_MODEL_MODULE", raising=False)
    monkeypatch.delenv("AI_TRADING_MODEL_PATH", raising=False)
    manager = SignalManager()
    df = _minimal_df()

    caplog.set_level("WARNING")
    manager.signal_ml(df)
    manager.signal_ml(df)

    warnings = [
        rec for rec in caplog.records if "ML predictions disabled" in rec.getMessage()
    ]
    assert len(warnings) == 1
