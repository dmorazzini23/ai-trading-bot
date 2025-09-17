from ai_trading.core.bot_engine import asset_class_for
from ai_trading.strategies.base import Strategy


def test_asset_class_for_crypto():
    """Symbols starting with crypto prefixes are labelled crypto."""
    assert asset_class_for("ETHBTC") == "crypto"


def test_strategy_generate_base():
    """Default Strategy implementation returns no signals and does not raise."""
    strategy = Strategy()
    assert strategy.generate(None) == []
