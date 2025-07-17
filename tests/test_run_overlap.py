import threading
import time
import types
import bot_engine


def test_run_all_trades_overlap(monkeypatch, caplog):
    state = bot_engine.BotState()
    caplog.set_level("INFO")

    monkeypatch.setattr(bot_engine, "is_market_open", lambda: True)
    monkeypatch.setattr(bot_engine, "check_pdt_rule", lambda ctx: False)
    monkeypatch.setattr(bot_engine, "_prepare_run", lambda ctx, st: (0.0, True, []))
    monkeypatch.setattr(bot_engine, "_process_symbols", lambda *a, **k: ([], {}))
    monkeypatch.setattr(bot_engine, "_send_heartbeat", lambda: None)
    monkeypatch.setattr(bot_engine.ctx.api, "get_account", lambda: types.SimpleNamespace(cash=0, equity=0))

    def slow_prepare(ctx, st):
        time.sleep(0.2)
        return (0.0, True, [])

    monkeypatch.setattr(bot_engine, "_prepare_run", slow_prepare)

    t = threading.Thread(target=bot_engine.run_all_trades_worker, args=(state, None))
    t.start()
    time.sleep(0.05)
    bot_engine.run_all_trades_worker(state, None)
    t.join()
    assert any("RUN_ALL_TRADES_SKIPPED_OVERLAP" in r.message for r in caplog.records)
