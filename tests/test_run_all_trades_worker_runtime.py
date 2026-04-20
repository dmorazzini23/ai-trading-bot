from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from ai_trading.core import bot_engine
from ai_trading.core.run_all_trades_prelude import RunAllTradesPreludeResult
from ai_trading.core.run_all_trades_worker import run_all_trades_worker_cycle


def test_run_all_trades_worker_records_execute_stage_and_softens_finalizer_hooks(
    monkeypatch,
    caplog,
) -> None:
    class DummyLock:
        def __init__(self) -> None:
            self.released = False

        def acquire(self, blocking: bool = False) -> bool:
            return True

        def release(self) -> None:
            self.released = True

    state = bot_engine.BotState()
    state.running = True
    state._strategies_loaded = True
    runtime = SimpleNamespace(
        risk_engine=SimpleNamespace(wait_for_exposure_update=lambda _timeout: None)
    )
    dummy_lock = DummyLock()
    cycle_walls: list[tuple[float, dict[str, str]]] = []
    hook_order: list[str] = []
    guard_calls: list[float] = []
    monotonic_values = iter((15.0, 19.0, 21.0))

    monkeypatch.setattr(bot_engine, "_ensure_alpaca_classes", lambda: None)
    monkeypatch.setattr(bot_engine, "_ALPACA_IMPORT_ERROR", None)
    monkeypatch.setattr(bot_engine, "_init_metrics", lambda: None)
    monkeypatch.setattr(bot_engine, "run_lock", dummy_lock)
    monkeypatch.setattr(bot_engine, "monotonic_time", lambda: next(monotonic_values))
    monkeypatch.setattr(
        "ai_trading.core.run_all_trades_worker.prepare_run_all_trades_cycle",
        lambda **_kwargs: RunAllTradesPreludeResult(
            ready=True,
            cfg_runtime=SimpleNamespace(),
            effective_policy=SimpleNamespace(),
            now=datetime(2026, 4, 20, 14, 0, tzinfo=UTC),
            loop_start=10.0,
            api=object(),
            previous_last_run_at=None,
        ),
    )
    monkeypatch.setattr(
        "ai_trading.core.run_all_trades_worker.execute_run_all_trades_cycle",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        bot_engine,
        "record_cycle_wall",
        lambda elapsed, metadata: cycle_walls.append((elapsed, metadata)),
    )
    monkeypatch.setattr(
        bot_engine,
        "get_trading_config",
        lambda: SimpleNamespace(execution_stale_ratio_shadow=0.42),
    )
    monkeypatch.setattr(
        bot_engine,
        "guard_end_cycle",
        lambda stale_threshold_ratio=0.0: guard_calls.append(stale_threshold_ratio),
    )
    monkeypatch.setattr(bot_engine, "guard_shadow_active", lambda: False)
    monkeypatch.setattr(
        bot_engine,
        "EXEC_GUARD_STATE",
        SimpleNamespace(stale_symbols=1, universe_size=2),
        raising=False,
    )

    def _heartbeat(*_args, **_kwargs) -> None:
        hook_order.append("heartbeat")
        raise RuntimeError("heartbeat failed")

    monkeypatch.setattr(bot_engine, "_log_loop_heartbeat", _heartbeat)
    monkeypatch.setattr(
        bot_engine,
        "flush_log_throttle_summaries",
        lambda: hook_order.append("flush"),
    )
    monkeypatch.setattr(
        bot_engine,
        "_check_runtime_stops",
        lambda _runtime: hook_order.append("stops"),
    )
    monkeypatch.setattr(bot_engine, "MEMORY_OPTIMIZATION_AVAILABLE", False, raising=False)

    caplog.set_level("WARNING")
    run_all_trades_worker_cycle(state, runtime)

    assert cycle_walls == [(4.0, {"stage": "cycle_execute"})]
    assert guard_calls == [0.42]
    assert hook_order == ["heartbeat", "flush", "stops"]
    assert state.running is False
    assert state._strategies_loaded is False
    assert state.last_loop_duration == 11.0
    assert dummy_lock.released is True
    assert any(
        record.message == "RUN_ALL_TRADES_FINALIZER_HOOK_FAILED"
        and getattr(record, "hook", None) == "loop_heartbeat"
        for record in caplog.records
    )
