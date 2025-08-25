import os
from ai_trading.config.management import TradingConfig


def test_disable_daily_retrain_env_parsing():
    """DISABLE_DAILY_RETRAIN parses truthy and falsy values."""
    cases = [
        ("true", True),
        ("1", True),
        ("false", False),
        ("0", False),
        ("", False),
        ("invalid", False),
    ]
    for env_val, expected in cases:
        os.environ["DISABLE_DAILY_RETRAIN"] = env_val
        cfg = TradingConfig.from_env()
        assert cfg.disable_daily_retrain is expected
        os.environ.pop("DISABLE_DAILY_RETRAIN", None)


def test_disable_daily_retrain_unset():
    """Unset variable defaults to False."""
    os.environ.pop("DISABLE_DAILY_RETRAIN", None)
    cfg = TradingConfig.from_env()
    assert cfg.disable_daily_retrain is False

