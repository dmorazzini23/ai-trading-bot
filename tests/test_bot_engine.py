import numpy as np
import pytest
pd = pytest.importorskip("pandas")

pytest.importorskip("alpaca")

# AI-AGENT-REF: Replaced unsafe _raise_dynamic_exec_disabled() with direct import from shim module
import os

os.environ.setdefault("ALPACA_API_KEY", "x")
os.environ.setdefault("ALPACA_SECRET_KEY", "x")
os.environ.setdefault("ALPACA_BASE_URL", "https://example.com")
os.environ.setdefault("WEBHOOK_SECRET", "x")

import alpaca.trading as _alpaca_trading

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
    if not hasattr(_alpaca_trading, _name):
        setattr(
            _alpaca_trading,
            _name,
            type(_name, (), {"__init__": lambda self, *a, **k: None}),
        )  # pragma: no cover - stubs

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
    ["ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_BASE_URL"],
)
def test_bot_engine_missing_env(monkeypatch, caplog, attr, missing_key):
    """BotEngine properties should raise informative errors when env vars are missing."""

    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://example.com")
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

    original_rsi = bot_engine.ta.rsi
    bot_engine.ta.rsi = lambda close, length=14: pd.Series([np.nan] * len(close))
    try:
        result = prepare_indicators(df.copy())
    finally:
        bot_engine.ta.rsi = original_rsi

    assert isinstance(result, pd.DataFrame)
    assert result.empty
