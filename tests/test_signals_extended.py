from ai_trading import signals


def test_generate():
    assert signals.generate() == 0
