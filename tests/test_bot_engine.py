import types
from typing import Any, cast

import numpy as np
import pytest
pd = pytest.importorskip("pandas")

from ai_trading.alpaca_api import AlpacaAuthenticationError
from ai_trading.core import bot_engine

from ai_trading.core.bot_engine import BotEngine, prepare_indicators

np.random.seed(0)


def test_prepare_indicators_creates_required_columns():
    df = pd.DataFrame({
        'open': np.random.uniform(100, 200, 30),
        'high': np.random.uniform(100, 200, 30),
        'low': np.random.uniform(100, 200, 30),
        'close': np.random.uniform(100, 200, 30),
        'volume': np.random.randint(1_000_000, 5_000_000, 30)
    })

    result = prepare_indicators(df.copy())

    required = ['ichimoku_conv', 'ichimoku_base', 'stochrsi']
    for col in required:
        assert col in result.columns, f"Missing expected column: {col}"

    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_prepare_indicators_ignores_non_indicator_nans():
    base_close = np.concatenate([
        np.full(20, 150.0),
        np.linspace(150.5, 165.0, 100),
    ])
    noise = np.random.normal(0, 0.5, base_close.size)
    close = base_close + noise
    df = pd.DataFrame({
        'open': close + np.random.normal(0, 0.3, close.size),
        'high': close + np.abs(np.random.normal(0.5, 0.2, close.size)),
        'low': close - np.abs(np.random.normal(0.5, 0.2, close.size)),
        'close': close,
        'volume': np.random.randint(500_000, 5_000_000, close.size),
        'unused_feature': np.nan,
    })

    result = prepare_indicators(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert not result.empty

    required = {
        'rsi',
        'rsi_14',
        'ichimoku_conv',
        'ichimoku_base',
        'stochrsi',
    }
    assert required.issubset(result.columns)
    assert 'unused_feature' in result.columns
    assert not result['stochrsi'].isna().all()


def test_safe_trade_handles_auth_failure(monkeypatch):
    state = bot_engine.BotState()
    ctx = cast(Any, types.SimpleNamespace(api=types.SimpleNamespace(list_positions=lambda: [])))

    def raise_auth(*_a, **_k):
        raise AlpacaAuthenticationError("Unauthorized")

    monkeypatch.setattr(bot_engine, "trade_logic", raise_auth)

    result = bot_engine._safe_trade(
        ctx,
        state,
        "AAPL",
        1_000.0,
        object(),
        True,
    )

    assert result is False
    assert "AAPL" in state.auth_skipped_symbols


def test_prepare_indicators_insufficient_data():
    """prepare_indicators returns an empty frame when history is too short."""

    df = pd.DataFrame({
        'open': np.random.uniform(100, 200, 5),
        'high': np.random.uniform(100, 200, 5),
        'low': np.random.uniform(100, 200, 5),
        'close': np.random.uniform(100, 200, 5),
        'volume': np.random.randint(1_000_000, 5_000_000, 5),
    })

    result = prepare_indicators(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert result.empty


@pytest.mark.parametrize("attr", ["trading_client", "data_client"])
@pytest.mark.parametrize(
    "missing_key",
    ["ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_TRADING_BASE_URL"],
)
def test_bot_engine_missing_env(monkeypatch, caplog, attr, missing_key):
    """BotEngine properties should raise informative errors when env vars are missing."""

    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://example.com")
    monkeypatch.delenv(missing_key, raising=False)
    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(
        "ai_trading.core.bot_engine.get_data_client_cls",
        lambda: DummyClient,
    )
    monkeypatch.setattr(
        "ai_trading.core.bot_engine.get_trading_client_cls",
        lambda: DummyClient,
    )
    monkeypatch.setattr(
        "ai_trading.core.bot_engine._ensure_alpaca_env_or_raise",
        lambda: ("key", "secret", "https://example.com"),
    )
    monkeypatch.setattr(
        "ai_trading.core.bot_engine._initialize_alpaca_clients",
        lambda: True,
    )
    monkeypatch.setattr(
        "ai_trading.core.bot_engine.ensure_data_fetcher",
        lambda runtime=None: None,
    )
    monkeypatch.setattr(
        "ai_trading.core.bot_engine._load_required_model",
        lambda: None,
    )
    monkeypatch.setattr(
        "ai_trading.core.bot_engine.data_fetcher_module.build_fetcher",
        lambda config=None: None,
    )

    engine = BotEngine()
    caplog.set_level("ERROR")

    with pytest.raises(RuntimeError) as exc:
        getattr(engine, attr)

    masked = f"{missing_key[:8]}***"
    assert masked in str(exc.value)
    assert any(masked in rec.getMessage() for rec in caplog.records)


def test_prepare_indicators_all_nan_columns():
    """prepare_indicators should raise when input columns are entirely NaN."""

    df = pd.DataFrame({
        'open': [np.nan] * 30,
        'high': [np.nan] * 30,
        'low': [np.nan] * 30,
        'close': [np.nan] * 30,
        'volume': [np.nan] * 30,
    })

    from ai_trading.core import bot_engine

    ta_module = cast(Any, bot_engine.ta)
    original_rsi = ta_module.rsi
    ta_module.rsi = lambda close, length=14: pd.Series([np.nan] * len(close))
    try:
        result = prepare_indicators(df.copy())
    finally:
        ta_module.rsi = original_rsi

    assert isinstance(result, pd.DataFrame)
    assert result.empty
