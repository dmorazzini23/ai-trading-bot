import threading
import time
import types

from ai_trading.core import bot_engine


def test_run_all_trades_overlap(monkeypatch, caplog):
    monkeypatch.delenv("AI_TRADING_MODEL_PATH", raising=False)
    monkeypatch.delenv("AI_TRADING_MODEL_MODULE", raising=False)
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setattr(bot_engine, "_MODEL_CACHE", None, raising=False)
    monkeypatch.setattr(bot_engine, "_global_ctx", None, raising=False)
    monkeypatch.setattr(bot_engine, "_ctx", None, raising=False)
    monkeypatch.setattr(bot_engine, "ctx", None, raising=False)

    state = bot_engine.BotState()
    runtime = bot_engine.get_ctx()
    caplog.set_level("INFO")

    monkeypatch.setattr(bot_engine, "is_market_open", lambda: True)
    monkeypatch.setattr(bot_engine, "check_pdt_rule", lambda ctx: False)
    monkeypatch.setattr(bot_engine, "_prepare_run", lambda ctx, st: (0.0, True, []))
    monkeypatch.setattr(bot_engine, "_process_symbols", lambda *a, **k: ([], {}))
    monkeypatch.setattr(bot_engine, "_send_heartbeat", lambda: None)

    api_obj = runtime.api
    if api_obj is None:
        api_obj = types.SimpleNamespace()
        runtime.api = api_obj
    monkeypatch.setattr(
        api_obj,
        "get_account",
        lambda: types.SimpleNamespace(cash=0, equity=0),
    )
    assert getattr(bot_engine._MODEL_CACHE, "is_placeholder_model", False)

    def slow_prepare(ctx, st):
        time.sleep(0.2)
        return (0.0, True, [])

    monkeypatch.setattr(bot_engine, "_prepare_run", slow_prepare)

    t = threading.Thread(target=bot_engine.run_all_trades_worker, args=(state, runtime))
    t.start()
    time.sleep(0.05)
    bot_engine.run_all_trades_worker(state, runtime)
    t.join()
    assert any("RUN_ALL_TRADES_SKIPPED_OVERLAP" in r.message for r in caplog.records)
