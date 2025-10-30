import os
from types import SimpleNamespace
from datetime import time
from unittest.mock import Mock, patch

import pandas as pd

# Ensure test environment variables
os.environ["PYTEST_RUNNING"] = "1"
os.environ.update({
    "ALPACA_API_KEY": "FAKE_TEST_API_KEY_NOT_REAL_123456789",
    "ALPACA_SECRET_KEY": "FAKE_TEST_SECRET_KEY_NOT_REAL_123456789",
    "ALPACA_BASE_URL": "https://paper-api.alpaca.markets",
    "WEBHOOK_SECRET": "fake-test-webhook-not-real",
    "FLASK_PORT": "9000",
    "TRADING_MODE": "balanced",
    "DOLLAR_RISK_LIMIT": "0.05",
    "TESTING": "1",
    "TRADE_LOG_FILE": "test_trades.csv",
    "SEED": "42",
    "RATE_LIMIT_BUDGET": "190",
    "DISABLE_DAILY_RETRAIN": "1",
    "DRY_RUN": "1",
    "SHADOW_MODE": "1",
})


class DummyLock:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _run(order):
    from ai_trading.core.bot_engine import _enter_short

    ctx = SimpleNamespace(
        api=SimpleNamespace(
            get_asset=lambda symbol: SimpleNamespace(
                shortable=True, shortable_shares=10
            )
        ),
        trade_logger=SimpleNamespace(log_entry=Mock()),
        market_open=time(9, 30),
        market_close=time(16, 0),
        stop_targets={},
        take_profit_targets={},
    )
    ctx.allow_short_selling = True
    state = SimpleNamespace(trade_cooldowns={}, last_trade_direction={})
    feat_df = pd.DataFrame({"atr": [1]})

    with patch("ai_trading.core.bot_engine.calculate_entry_size", return_value=1), \
        patch("ai_trading.core.bot_engine.get_latest_close", return_value=100), \
        patch("ai_trading.core.bot_engine._apply_sector_cap_qty", return_value=1), \
        patch("ai_trading.core.bot_engine.submit_order", return_value=order), \
        patch("ai_trading.core.bot_engine.is_high_vol_regime", return_value=False), \
        patch("ai_trading.core.bot_engine.scaled_atr_stop", return_value=(1, 2)), \
        patch("ai_trading.core.bot_engine.targets_lock", DummyLock()), \
        patch("ai_trading.core.bot_engine.trade_cooldowns_lock", DummyLock()), \
        patch("ai_trading.core.bot_engine._record_trade_in_frequency_tracker", lambda *a, **k: None), \
        patch("ai_trading.core.bot_engine.logger") as mock_logger:
        _enter_short(ctx, state, "AAPL", feat_df, 0.5, 0.9, "strat")
    return mock_logger


def test_enter_short_logs_string_order_id():
    logger = _run("order-123")
    logger.debug.assert_any_call(
        "TRADE_LOGIC_ORDER_PLACED | symbol=AAPL  order_id=order-123"
    )


def test_enter_short_logs_object_order_id():
    order_obj = SimpleNamespace(id="abc-789")
    logger = _run(order_obj)
    logger.debug.assert_any_call(
        "TRADE_LOGIC_ORDER_PLACED | symbol=AAPL  order_id=abc-789"
    )
