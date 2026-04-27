from __future__ import annotations

import types

import ai_trading.core.startup_runtime as startup_runtime


class _Logger:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def info(self, message: str, *args, **kwargs) -> None:
        self.events.append(("info", message))

    def warning(self, message: str, *args, **kwargs) -> None:
        self.events.append(("warning", message))

    def error(self, message: str, *args, **kwargs) -> None:
        self.events.append(("error", message))

    def debug(self, message: str, *args, **kwargs) -> None:
        self.events.append(("debug", message))

    def exception(self, message: str, *args, **kwargs) -> None:
        self.events.append(("exception", message))


def test_normalized_sleep_seconds_rejects_invalid_and_nan() -> None:
    assert startup_runtime._normalized_sleep_seconds("2.5", default=1.0) == 2.5
    assert startup_runtime._normalized_sleep_seconds("-3", default=1.0) == 0.0
    assert startup_runtime._normalized_sleep_seconds("nan", default=1.0) == 1.0
    assert startup_runtime._normalized_sleep_seconds("bad", default=1.0) == 1.0


def test_ensure_rebalance_tracking_replaces_non_dicts() -> None:
    ctx = types.SimpleNamespace(rebalance_ids=[], rebalance_attempts={"AAPL": 1})

    startup_runtime._ensure_rebalance_tracking(ctx)

    assert ctx.rebalance_ids == {}
    assert ctx.rebalance_attempts == {"AAPL": 1}
    assert ctx.rebalance_buys == {}


def test_update_service_status_safe_logs_debug_on_failure() -> None:
    logger = _Logger()

    def _fail(**_kwargs) -> None:
        raise RuntimeError("state unavailable")

    be = types.SimpleNamespace(
        logger=logger,
        runtime_state=types.SimpleNamespace(update_service_status=_fail),
    )

    startup_runtime._update_service_status_safe(be, status="degraded", reason="stream_failed")

    assert logger.events == [("debug", "RUNTIME_SERVICE_STATUS_UPDATE_FAILED")]


def test_run_trade_updates_stream_marks_unexpected_return_degraded() -> None:
    logger = _Logger()
    updates: list[dict[str, object]] = []

    be = types.SimpleNamespace(
        ALPACA_API_KEY="key",
        ALPACA_SECRET_KEY="secret",
        trading_client=object(),
        state=object(),
        logger=logger,
        runtime_state=types.SimpleNamespace(update_service_status=lambda **kwargs: updates.append(kwargs)),
    )
    ctx = types.SimpleNamespace(stream_event=types.SimpleNamespace(is_set=lambda: True))

    async def _return(*_args, **_kwargs):
        return None

    startup_runtime._run_trade_updates_stream(be, ctx, _return)

    assert ("warning", "TRADE_UPDATES_STREAM_EXITED") in logger.events
    assert updates == [{"status": "degraded", "reason": "trade_updates_stream_exited"}]


def test_schedule_run_all_trades_starts_worker_thread(monkeypatch) -> None:
    started: list[dict[str, object]] = []

    class _Thread:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def start(self) -> None:
            started.append(self.kwargs)

    be = types.SimpleNamespace(
        is_market_open=lambda: True,
        _LAST_MARKET_CLOSED_LOG=99.0,
        ensure_alpaca_attached=lambda _runtime: None,
        _validate_trading_api=lambda _api: True,
        threading=types.SimpleNamespace(Thread=_Thread),
        run_all_trades_worker=lambda *_args: None,
        state="state",
    )
    monkeypatch.setattr(startup_runtime, "_bot_engine", lambda: be)
    runtime = types.SimpleNamespace(api="api")

    startup_runtime.schedule_run_all_trades_runtime(runtime)

    assert be._LAST_MARKET_CLOSED_LOG == 0.0
    assert started[0]["target"] is be.run_all_trades_worker
    assert started[0]["args"] == ("state", runtime)
    assert started[0]["daemon"] is True
