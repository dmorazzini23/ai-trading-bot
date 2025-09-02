import logging
from types import SimpleNamespace
from ai_trading.core import bot_engine


def test_too_many_positions_api_unavailable(caplog):
    ctx = SimpleNamespace(api=SimpleNamespace())
    with caplog.at_level(logging.WARNING):
        assert bot_engine.too_many_positions(ctx) is False
    assert "Positions API unavailable" in caplog.text

