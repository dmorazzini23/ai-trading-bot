import threading
import types

from ai_trading.core import bot_engine


def test_run_all_trades_overlap(monkeypatch, caplog):
    from ai_trading.core import run_all_trades_worker as worker

    entered = threading.Event()
    release = threading.Event()
    errors = []
    calls = []
    state = bot_engine.BotState()
    runtime = types.SimpleNamespace(risk_engine=object())
    caplog.set_level("INFO")

    monkeypatch.setattr(bot_engine, "run_lock", threading.Lock())
    monkeypatch.setattr(
        worker, "prepare_run_all_trades_cycle",
        lambda *args, **kwargs: types.SimpleNamespace(
            ready=True, cfg_runtime=object(), loop_start=0.0,
            api=object(), previous_last_run_at=None,
        ),
    )
    monkeypatch.setattr(worker, "_finalize_run_all_trades_cycle", lambda **kwargs: None)

    def execute(**kwargs):
        calls.append(kwargs)
        entered.set()
        assert release.wait(timeout=10), "first cycle was not released"

    monkeypatch.setattr(worker, "execute_run_all_trades_cycle", execute)

    def run_first():
        try:
            worker.run_all_trades_worker_cycle(state, runtime)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_first)
    thread.start()
    try:
        assert entered.wait(timeout=5), "first cycle never acquired the lock"
        worker.run_all_trades_worker_cycle(state, runtime)
        assert len(calls) == 1
        assert any("RUN_ALL_TRADES_SKIPPED_OVERLAP" in r.message for r in caplog.records)
    finally:
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert not errors
    assert not bot_engine.run_lock.locked()


