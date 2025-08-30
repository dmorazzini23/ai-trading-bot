from ai_trading.core import bot_engine


def test_get_sector_known_symbols():
    assert bot_engine.get_sector("PLTR") == "Technology"
    assert bot_engine.get_sector("BABA") == "Technology"
