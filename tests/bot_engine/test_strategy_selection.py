from ai_trading.core import bot_engine


def test_get_strategies_from_env(monkeypatch):
    class Dummy:
        name = "dummy"

        def __init__(self):
            pass

    class Alt:
        name = "alt"

        def __init__(self):
            pass

    import ai_trading.strategies as strategies

    monkeypatch.setattr(strategies, "REGISTRY", {"dummy": Dummy, "alt": Alt}, raising=False)
    monkeypatch.setenv("STRATEGIES", "dummy,alt")
    strategies_list = bot_engine.get_strategies()
    assert {type(s) for s in strategies_list} == {Dummy, Alt}


def test_get_strategies_defaults_to_momentum(monkeypatch):
    class Momentum:
        name = "momentum"

        def __init__(self):
            pass

    import ai_trading.strategies as strategies

    monkeypatch.setattr(strategies, "REGISTRY", {"momentum": Momentum}, raising=False)
    monkeypatch.delenv("STRATEGIES", raising=False)
    strategies_list = bot_engine.get_strategies()
    assert len(strategies_list) == 1
    assert isinstance(strategies_list[0], Momentum)
