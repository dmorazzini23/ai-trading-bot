import ai_trading.core.bot_engine as be
import ai_trading.portfolio.core as pc


def test_imports():
    assert be is not None and pc is not None
