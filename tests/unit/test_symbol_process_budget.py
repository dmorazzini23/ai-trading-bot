import types

from ai_trading.core import bot_engine


def test_symbol_processing_budget(monkeypatch):
    """Symbol processing stops immediately when budget is exhausted."""
    monkeypatch.setenv("SYMBOL_PROCESS_BUDGET", "0")
    monkeypatch.setattr(bot_engine, "ensure_final_bar", lambda s, tf: True)
    bot_engine.state = types.SimpleNamespace(
        trade_cooldowns={}, last_trade_direction={}, position_cache={}
    )
    dummy_exec = types.SimpleNamespace(
        submit=lambda fn, *a, **k: types.SimpleNamespace(result=lambda: None)
    )
    bot_engine.executors = types.SimpleNamespace(
        _ensure_executors=lambda: None,
        prediction_executor=dummy_exec,
        executor=dummy_exec,
    )
    monkeypatch.setattr(bot_engine, "get_ctx", lambda: object())

    processed, _ = bot_engine._process_symbols(["AAPL", "MSFT"], 100.0, None, True)
    assert processed == []
