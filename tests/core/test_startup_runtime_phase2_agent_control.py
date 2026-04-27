from __future__ import annotations

import types

import pytest

import ai_trading.core.startup_runtime as startup_runtime


class _Logger:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def info(self, message: str, *args, **kwargs) -> None:
        self.events.append(("info", message))

    def warning(self, message: str, *args, **kwargs) -> None:
        self.events.append(("warning", message))

    def exception(self, message: str, *args, **kwargs) -> None:
        self.events.append(("exception", message))

    def debug(self, message: str, *args, **kwargs) -> None:
        self.events.append(("debug", message))


def test_schedule_run_all_trades_skips_when_market_closed(monkeypatch) -> None:
    logger = _Logger()
    calls: list[str] = []
    be = types.SimpleNamespace(
        is_market_open=lambda: False,
        _log_market_closed=lambda message: calls.append(message),
        ensure_alpaca_attached=lambda _runtime: calls.append("attached"),
        _validate_trading_api=lambda _api: True,
        threading=types.SimpleNamespace(Thread=lambda **_kwargs: calls.append("thread")),
        logger=logger,
    )
    monkeypatch.setattr(startup_runtime, "_bot_engine", lambda: be)

    startup_runtime.schedule_run_all_trades_runtime(types.SimpleNamespace(api=object()))

    assert calls == ["Market closed—skipping run_all_trades."]


def test_schedule_run_all_trades_stops_when_api_validation_fails(monkeypatch) -> None:
    calls: list[str] = []
    be = types.SimpleNamespace(
        is_market_open=lambda: True,
        _LAST_MARKET_CLOSED_LOG=123.0,
        ensure_alpaca_attached=lambda _runtime: calls.append("attached"),
        _validate_trading_api=lambda _api: False,
        threading=types.SimpleNamespace(Thread=lambda **_kwargs: calls.append("thread")),
        run_all_trades_worker=lambda *_args: calls.append("worker"),
        state=object(),
    )
    monkeypatch.setattr(startup_runtime, "_bot_engine", lambda: be)

    startup_runtime.schedule_run_all_trades_runtime(types.SimpleNamespace(api=None))

    assert calls == ["attached"]
    assert be._LAST_MARKET_CLOSED_LOG == 0.0


def test_configure_runtime_jobs_resets_configured_flag_on_failure(monkeypatch) -> None:
    logger = _Logger()

    class _FailingSchedule:
        def every(self, *_args):
            raise RuntimeError("scheduler unavailable")

    be = types.SimpleNamespace(schedule=_FailingSchedule(), logger=logger)
    monkeypatch.setattr(startup_runtime, "_bot_engine", lambda: be)
    ctx = types.SimpleNamespace()

    with pytest.raises(RuntimeError, match="scheduler unavailable"):
        startup_runtime.configure_main_runtime_jobs(ctx)

    assert ctx._runtime_jobs_configured is False


def test_run_trade_updates_stream_ignores_unreadable_event_state() -> None:
    warnings: list[tuple[str, object]] = []
    service_updates: list[dict[str, object]] = []

    class _BrokenEvent:
        def is_set(self) -> bool:
            raise RuntimeError("event unavailable")

    be = types.SimpleNamespace(
        ALPACA_API_KEY="key",
        ALPACA_SECRET_KEY="secret",
        trading_client=object(),
        state=object(),
        logger=types.SimpleNamespace(
            warning=lambda event, extra=None: warnings.append((event, extra)),
            debug=lambda *args, **kwargs: None,
        ),
        runtime_state=types.SimpleNamespace(
            update_service_status=lambda **kwargs: service_updates.append(kwargs)
        ),
    )
    ctx = types.SimpleNamespace(stream_event=_BrokenEvent())

    async def _return(*args, **kwargs):
        return None

    startup_runtime._run_trade_updates_stream(be, ctx, _return)

    assert warnings == []
    assert service_updates == []
