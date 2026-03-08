import importlib
import importlib.machinery
import logging
import sys
import types
from types import SimpleNamespace
from typing import Any, cast


def test_health_check_retries_on_api_error(monkeypatch, caplog):
    flask_stub = types.ModuleType("flask")
    class _Flask:
        def __init__(self, *args, **kwargs):
            pass
        def route(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator
    cast(Any, flask_stub).Flask = _Flask
    sys.modules.setdefault("flask", flask_stub)
    pandas_stub = types.ModuleType("pandas")
    pandas_stub.__spec__ = importlib.machinery.ModuleSpec("pandas", loader=None)
    cast(Any, pandas_stub).DataFrame = type("DataFrame", (), {})
    sys.modules.setdefault("pandas", pandas_stub)
    bot_engine = importlib.import_module("ai_trading.core.bot_engine")

    calls = {"n": 0}

    def failing_check(ctx, symbols):
        calls["n"] += 1
        if calls["n"] < 2:
            raise bot_engine.APIError("boom")

    setattr(bot_engine, "REGIME_SYMBOLS", ["AAPL"])
    monkeypatch.setattr(bot_engine, "data_source_health_check", failing_check)
    sleeps: list[int] = []
    monkeypatch.setattr(bot_engine.time, "sleep", lambda s: sleeps.append(s))

    ctx = SimpleNamespace()
    setattr(bot_engine, "_HEALTH_CHECK_FAILURES", 0)
    with caplog.at_level(logging.WARNING):
        bot_engine._initialize_bot_context_post_setup(ctx)

    assert calls["n"] == 2
    assert sleeps == [1]
    assert cast(int, getattr(bot_engine, "_HEALTH_CHECK_FAILURES")) == 0
