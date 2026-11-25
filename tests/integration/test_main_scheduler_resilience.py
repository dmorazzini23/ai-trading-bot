import logging
import sys
import types

from ai_trading import main
from ai_trading.monitoring import decision_loop_watchdog


class _DummyMetric:
    def labels(self, *args, **kwargs):  # noqa: D401 - test stub
        return self

    def observe(self, *args, **kwargs):  # noqa: D401 - test stub
        return None

    def inc(self, *args, **kwargs):  # noqa: D401 - test stub
        return None


def _patch_main_runtime(monkeypatch) -> None:
    monkeypatch.setenv("SCHEDULER_ITERATIONS", "2")
    monkeypatch.setenv("ALLOW_AFTER_HOURS", "1")

    settings = types.SimpleNamespace(
        alpaca_data_feed="iex",
        alpaca_adjustment="raw",
        iterations=None,
        scheduler_iterations=0,
        interval=1,
        interval_when_closed=1,
        api_port=9001,
        api_port_wait_seconds=0,
        health_tick_seconds=60,
        max_position_mode="STATIC",
        max_position_size=1.0,
        http_connect_timeout=1.0,
        http_read_timeout=1.0,
        http_pool_maxsize=1,
        http_total_retries=1,
        http_backoff_factor=0.1,
        alpaca_base_url="https://paper-api.example.com",
        capital_cap=0.1,
        dollar_risk_limit=0.1,
        paper=True,
        trading_mode="balanced",
    )

    monkeypatch.setattr(main, "_check_alpaca_sdk", lambda: None)
    monkeypatch.setattr(main, "_fail_fast_env", lambda: None)
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "_assert_singleton_api", lambda *_a, **_k: None)
    monkeypatch.setattr(main, "_init_http_session", lambda *_a, **_k: True)
    monkeypatch.setattr(main, "ensure_trade_log_path", lambda: None)
    monkeypatch.setattr(main, "preflight_import_health", lambda: None)
    monkeypatch.setattr(main, "ensure_dotenv_loaded", lambda: None)
    monkeypatch.setattr(main, "_is_market_open_base", lambda: True)
    monkeypatch.setattr(main, "optimize_memory", lambda: {})
    monkeypatch.setattr(main, "resolve_max_position_size", lambda *a, **k: (1.0, {}))
    monkeypatch.setattr(main.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(main, "_interruptible_sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(main, "get_histogram", lambda *a, **k: _DummyMetric())
    monkeypatch.setattr(main, "get_counter", lambda *a, **k: _DummyMetric())

    def _start_api_with_signal(api_ready, _api_error):
        api_ready.set()

    monkeypatch.setattr(main, "start_api_with_signal", _start_api_with_signal)

    class _InlineThread:
        def __init__(self, target, args=(), kwargs=None, daemon=None):
            self._target = target
            self._args = args
            self._kwargs = kwargs or {}
            self.daemon = daemon

        def start(self):
            self._target(*self._args, **self._kwargs)

        def is_alive(self):
            return False

    monkeypatch.setattr(main, "Thread", _InlineThread)


def test_scheduler_logs_and_continues_after_runtime_error(monkeypatch, caplog):
    _patch_main_runtime(monkeypatch)

    calls = {"count": 0}
    replay_triggers: list[dict[str, object]] = []

    def _run_cycle_stub():
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("boom")

    monkeypatch.setattr(main, "run_cycle", _run_cycle_stub)
    monkeypatch.setattr(
        main,
        "_maybe_build_bad_session_replay_dataset",
        lambda **kwargs: replay_triggers.append(dict(kwargs)),
    )
    progress: list[int | None] = []
    watchdog_started: list[bool] = []
    monkeypatch.setattr(
        decision_loop_watchdog,
        "note_decision_loop_progress",
        lambda *, cycle_index=None, now=None: progress.append(cycle_index),
    )
    monkeypatch.setattr(
        decision_loop_watchdog,
        "start_decision_loop_watchdog",
        lambda **_kwargs: watchdog_started.append(True),
    )

    with caplog.at_level(logging.ERROR, logger=main.logger.name):
        main.main(["--iterations", "2", "--interval", "1"])

    assert calls["count"] == 3  # warm-up + two scheduler iterations
    # Both the normal and the exception path count as a finished iteration;
    # the watchdog thread is not started under tests.
    assert progress == [1, 2]
    assert watchdog_started == []
    scheduler_errors = [
        record
        for record in caplog.records
        if "SCHEDULER_RUN_CYCLE_EXCEPTION" in record.getMessage()
    ]
    assert scheduler_errors
    assert all(
        getattr(record, "cycle_index", None) == getattr(record, "iteration", None)
        for record in scheduler_errors
    )

    scheduler_replays = [
        payload
        for payload in replay_triggers
        if str(payload.get("trigger")) == "scheduler_run_cycle_exception"
    ]
    assert scheduler_replays
    detail_payload = scheduler_replays[0].get("detail")
    detail = detail_payload if isinstance(detail_payload, dict) else {}
    assert detail.get("cycle_index") == detail.get("iteration")


def test_scheduler_trades_when_watchdog_module_cannot_load(monkeypatch, caplog):
    _patch_main_runtime(monkeypatch)
    calls = {"count": 0}

    def _run_cycle_stub():
        calls["count"] += 1

    monkeypatch.setattr(main, "run_cycle", _run_cycle_stub)
    # A None entry in sys.modules makes the watchdog import raise ImportError.
    monkeypatch.setitem(sys.modules, "ai_trading.monitoring.decision_loop_watchdog", None)

    with caplog.at_level(logging.ERROR, logger=main.logger.name):
        main.main(["--iterations", "2", "--interval", "1"])

    assert calls["count"] == 3  # warm-up + two scheduler iterations still run
    unavailable = [
        record
        for record in caplog.records
        if "DECISION_LOOP_WATCHDOG_UNAVAILABLE" in record.getMessage()
    ]
    assert len(unavailable) == 1
    assert unavailable[0].levelno == logging.ERROR
