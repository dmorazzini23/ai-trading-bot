import numpy as np
import pytest
pd = pytest.importorskip("pandas")

pytest.importorskip("alpaca_trade_api")

# AI-AGENT-REF: Replaced unsafe _raise_dynamic_exec_disabled() with direct import from shim module
import os

os.environ.setdefault("ALPACA_API_KEY", "x")
os.environ.setdefault("ALPACA_SECRET_KEY", "x")
os.environ.setdefault("ALPACA_BASE_URL", "https://example.com")
os.environ.setdefault("WEBHOOK_SECRET", "x")

import alpaca_trade_api.rest as _alpaca_rest

_MISSING = [
    "StockLatestQuoteRequest",
    "Quote",
    "OrderSide",
    "OrderStatus",
    "TimeInForce",
    "Order",
    "MarketOrderRequest",
]

for _name in _MISSING:
    if not hasattr(_alpaca_rest, _name):
        setattr(_alpaca_rest, _name, type(_name, (), {"__init__": lambda self, *a, **k: None}))  # pragma: no cover - stubs

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


def test_prepare_indicators_insufficient_data():
    """prepare_indicators should return an empty DataFrame when there is
    insufficient historical data for rolling calculations."""

    df = pd.DataFrame({
        'open': np.random.uniform(100, 200, 5),
        'high': np.random.uniform(100, 200, 5),
        'low': np.random.uniform(100, 200, 5),
        'close': np.random.uniform(100, 200, 5),
        'volume': np.random.randint(1_000_000, 5_000_000, 5),
    })

    result = prepare_indicators(df.copy())

    assert result.empty or result.shape[0] == 0


@pytest.mark.parametrize("attr", ["trading_client", "data_client"])
@pytest.mark.parametrize(
    "missing_key",
    ["ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_BASE_URL"],
)
def test_bot_engine_missing_env(monkeypatch, caplog, attr, missing_key):
    """BotEngine properties should raise informative errors when env vars are missing."""

    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://example.com")
    monkeypatch.delenv(missing_key, raising=False)
    engine = BotEngine()
    caplog.set_level("ERROR")

    with pytest.raises(RuntimeError) as exc:
        getattr(engine, attr)

    masked = f"{missing_key[:8]}***"
    assert masked in str(exc.value)
    assert any(masked in rec.getMessage() for rec in caplog.records)


def test_prepare_indicators_all_nan_columns():
    """prepare_indicators should drop all rows when input columns are entirely NaN."""

    df = pd.DataFrame({
        'open': [np.nan] * 30,
        'high': [np.nan] * 30,
        'low': [np.nan] * 30,
        'close': [np.nan] * 30,
        'volume': [np.nan] * 30,
    })

    from ai_trading.core import bot_engine

    original_rsi = bot_engine.ta.rsi
    bot_engine.ta.rsi = lambda close, length=14: pd.Series([np.nan] * len(close))
    try:
        result = prepare_indicators(df.copy())
    finally:
        bot_engine.ta.rsi = original_rsi

    assert result.empty or result.shape[0] == 0
