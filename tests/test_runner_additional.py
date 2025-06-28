import pytest
import bot_engine  # replace old bot import

def test_runner_starts():
    assert bot_engine.pre_trade_health_check()
