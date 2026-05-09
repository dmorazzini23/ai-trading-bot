import datetime
import types
from collections.abc import Callable

import pytest

from ai_trading.core.startup_runtime import (
    configure_main_runtime_jobs,
    _run_trade_updates_stream,
    initial_rebalance_runtime,
)


pd = pytest.importorskip("pandas")


class _DummyFetcher:
    def get_daily_df(self, ctx, symbol):
        return pd.DataFrame({"close": [100.0]})


class _DummyAPI:
    def __init__(self) -> None:
        self.positions: dict[str, int] = {}

    def get_account(self):
        return types.SimpleNamespace(cash=1000.0, equity=1000.0, buying_power=1000.0)

    def list_positions(self):
        return [types.SimpleNamespace(symbol=s, qty=q) for s, q in self.positions.items()]


def test_initial_rebalance_runtime_initializes_missing_tracking_attrs(monkeypatch):
    from ai_trading.core import bot_engine

    ctx = types.SimpleNamespace(api=_DummyAPI(), data_fetcher=_DummyFetcher())

    class FakeDateTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.datetime(2025, 7, 26, 0, 16, tzinfo=datetime.UTC)

    monkeypatch.setattr(bot_engine, "datetime", FakeDateTime)
    monkeypatch.setattr(
        "ai_trading.services.execution.submit_order",
        lambda ctx_, symbol, qty, side: ctx_.api.positions.setdefault(symbol, qty) or object(),
    )

    initial_rebalance_runtime(ctx, ["AAPL"])

    assert isinstance(ctx.rebalance_ids, dict)
    assert isinstance(ctx.rebalance_attempts, dict)
    assert isinstance(ctx.rebalance_buys, dict)
    assert ctx.api.positions["AAPL"] == 10


def test_run_trade_updates_stream_logs_failure() -> None:
    warnings: list[tuple[str, dict[str, object] | None]] = []
    service_updates: list[dict[str, object]] = []
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
    ctx = types.SimpleNamespace(stream_event=object())

    async def _raise(*args, **kwargs):
        raise RuntimeError("stream boom")

    _run_trade_updates_stream(be, ctx, _raise)

    assert warnings == [
        (
            "TRADE_UPDATES_STREAM_FAILED",
            {"error_type": "RuntimeError", "detail": "stream boom"},
        )
    ]
    assert service_updates == [
        {"status": "degraded", "reason": "trade_updates_stream_failed"}
    ]


def test_run_trade_updates_stream_marks_unexpected_exit() -> None:
    warnings: list[tuple[str, dict[str, object] | None]] = []
    service_updates: list[dict[str, object]] = []
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
    ctx = types.SimpleNamespace(
        stream_event=types.SimpleNamespace(is_set=lambda: True)
    )

    async def _return(*args, **kwargs):
        return None

    _run_trade_updates_stream(be, ctx, _return)

    assert warnings == [
        (
            "TRADE_UPDATES_STREAM_EXITED",
            {"reason": "unexpected_return"},
        )
    ]
    assert service_updates == [
        {"status": "degraded", "reason": "trade_updates_stream_exited"}
    ]


def test_initial_rebalance_runtime_skips_bad_symbol_and_continues(monkeypatch):
    from ai_trading.core import bot_engine

    class _MixedFetcher:
        def get_daily_df(self, ctx, symbol):
            if symbol == "BAD":
                raise RuntimeError("no daily data")
            return pd.DataFrame({"close": [100.0]})

    ctx = types.SimpleNamespace(
        api=_DummyAPI(),
        data_fetcher=_MixedFetcher(),
        rebalance_ids={},
        rebalance_attempts={},
        rebalance_buys={},
    )
    warnings: list[tuple[str, dict[str, object] | None]] = []

    class FakeDateTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.datetime(2025, 7, 26, 0, 16, tzinfo=datetime.UTC)

    monkeypatch.setattr(bot_engine, "datetime", FakeDateTime)
    monkeypatch.setattr(
        bot_engine.logger,
        "warning",
        lambda event, extra=None: warnings.append((event, extra)),
    )
    monkeypatch.setattr(
        "ai_trading.services.execution.submit_order",
        lambda ctx_, symbol, qty, side: ctx_.api.positions.setdefault(symbol, qty) or object(),
    )

    initial_rebalance_runtime(ctx, ["BAD", "AAPL"])

    assert ctx.api.positions["AAPL"] == 10
    assert warnings == [
        (
            "INITIAL_REBALANCE_PRICE_LOAD_FAILED",
            {
                "symbol": "BAD",
                "cause": "RuntimeError",
                "detail": "no daily data",
            },
        )
    ]


def test_initial_rebalance_runtime_buys_delta_for_existing_long(monkeypatch):
    from ai_trading.core import bot_engine

    ctx = types.SimpleNamespace(
        api=_DummyAPI(),
        data_fetcher=_DummyFetcher(),
        rebalance_ids={},
        rebalance_attempts={},
        rebalance_buys={},
    )
    ctx.api.positions["AAPL"] = 4
    submitted: list[tuple[str, int, str]] = []

    class FakeDateTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.datetime(2025, 7, 26, 0, 16, tzinfo=datetime.UTC)

    def _submit_order(ctx_, symbol, qty, side):
        submitted.append((symbol, qty, side))
        ctx_.api.positions[symbol] += qty
        return object()

    monkeypatch.setattr(bot_engine, "datetime", FakeDateTime)
    monkeypatch.setattr("ai_trading.services.execution.submit_order", _submit_order)

    initial_rebalance_runtime(ctx, ["AAPL"])

    assert submitted == [("AAPL", 6, "buy")]
    assert ctx.api.positions["AAPL"] == 10


def test_initial_rebalance_runtime_treats_short_side_as_signed(monkeypatch):
    from ai_trading.core import bot_engine

    class _ShortAPI(_DummyAPI):
        def __init__(self) -> None:
            super().__init__()
            self.positions["AAPL"] = -5

        def list_positions(self):
            return [
                types.SimpleNamespace(
                    symbol=s,
                    qty=abs(q),
                    side=("short" if q < 0 else "long"),
                )
                for s, q in self.positions.items()
            ]

    ctx = types.SimpleNamespace(
        api=_ShortAPI(),
        data_fetcher=_DummyFetcher(),
        rebalance_ids={},
        rebalance_attempts={},
        rebalance_buys={},
    )
    submitted: list[tuple[str, int, str]] = []

    class FakeDateTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.datetime(2025, 7, 26, 0, 16, tzinfo=datetime.UTC)

    def _submit_order(ctx_, symbol, qty, side):
        submitted.append((symbol, qty, side))
        ctx_.api.positions[symbol] += qty
        return object()

    monkeypatch.setattr(bot_engine, "datetime", FakeDateTime)
    monkeypatch.setattr("ai_trading.services.execution.submit_order", _submit_order)

    initial_rebalance_runtime(ctx, ["AAPL"])

    assert submitted == [("AAPL", 15, "buy")]
    assert ctx.api.positions["AAPL"] == 10
    assert bot_engine.state.position_cache["AAPL"] == 10
    assert "AAPL" not in bot_engine.state.short_positions


def test_configure_main_runtime_jobs_is_idempotent(monkeypatch):
    from ai_trading.core import bot_engine

    scheduled_jobs: list[Callable[[], object]] = []
    info_events: list[object] = []
    stream_thread_starts: list[str] = []
    run_calls: list[str] = []

    class _FakeJob:
        def __init__(self, jobs: list[Callable[[], object]]) -> None:
            self._jobs = jobs

        @property
        def minutes(self):
            return self

        @property
        def hours(self):
            return self

        @property
        def day(self):
            return self

        def at(self, _value: str):
            return self

        def do(self, fn: Callable[[], object]):
            self._jobs.append(fn)
            return fn

    class _FakeSchedule:
        def every(self, *args):
            return _FakeJob(scheduled_jobs)

    class _FakeThread:
        def __init__(self, *, target=None, args=(), daemon=None):
            self._target = target

        def start(self) -> None:
            stream_thread_starts.append("started")

    async def _start_trade_updates_stream(*args, **kwargs):
        return None

    monkeypatch.setattr(
        "ai_trading.core.startup_runtime.schedule_run_all_trades_runtime",
        lambda _ctx: run_calls.append("run"),
    )
    monkeypatch.setattr(bot_engine, "schedule", _FakeSchedule())
    monkeypatch.setattr(bot_engine, "CFG", types.SimpleNamespace(scheduler_sleep_seconds=0.0))
    monkeypatch.setattr(bot_engine, "validate_open_orders", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bot_engine, "_update_risk_engine_exposure", lambda: None)
    monkeypatch.setattr(bot_engine, "_emit_periodic_metrics", lambda: None)
    monkeypatch.setattr(bot_engine, "get_env", lambda _name, default=None, cast=None: default)
    monkeypatch.setattr(bot_engine, "update_signal_weights", lambda: None)
    monkeypatch.setattr(bot_engine, "update_bot_mode", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bot_engine, "adaptive_risk_scaling", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bot_engine, "get_rebalance_interval_min", lambda: 15)
    monkeypatch.setattr(bot_engine, "maybe_rebalance", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bot_engine, "check_disaster_halt", lambda: None)
    monkeypatch.setattr(bot_engine, "_alpaca_symbols", lambda: ([], _start_trade_updates_stream))
    monkeypatch.setattr(bot_engine, "threading", types.SimpleNamespace(Thread=_FakeThread))
    monkeypatch.setattr(
        bot_engine.logger,
        "info",
        lambda *args, **kwargs: info_events.append(args[0] if args else None),
    )

    ctx = types.SimpleNamespace()

    configure_main_runtime_jobs(ctx)
    first_job_count = len(scheduled_jobs)

    configure_main_runtime_jobs(ctx)

    assert first_job_count > 0
    assert len(scheduled_jobs) == first_job_count
    assert len(stream_thread_starts) == 1
    assert run_calls == ["run"]
    assert ctx._runtime_jobs_configured is True
    assert "RUNTIME_JOBS_ALREADY_CONFIGURED" in info_events


class _Logger:
    def info(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None


class _Lock:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _fake_bot_engine(*, stale_data: list[str], allow_stale: str | None = None):
    summary = {
        "failures": [],
        "insufficient_rows": [],
        "missing_columns": [],
        "invalid_values": [],
        "timezone_issues": [],
        "stale_data": stale_data,
    }

    def get_env(name, default=None):
        if name == "ALLOW_STALE_DATA_STARTUP" and allow_stale is not None:
            return allow_stale
        return default

    return types.SimpleNamespace(
        cancel_all_open_orders=lambda ctx: None,
        audit_positions=lambda ctx: None,
        load_tickers=lambda path: ["AAPL"],
        pre_trade_health_check=lambda ctx, symbols: summary,
        logger=_Logger(),
        get_env=get_env,
        TICKERS_FILE="tickers.csv",
        CFG=types.SimpleNamespace(min_health_rows=120),
        pytime=types.SimpleNamespace(time=lambda: 1.0),
        sentiment_lock=_Lock(),
        _SENTIMENT_CACHE={},
    )


def test_run_main_startup_runtime_stale_data_fails_closed_by_default(monkeypatch):
    from ai_trading.core import startup_runtime

    monkeypatch.setattr(
        startup_runtime,
        "_bot_engine",
        lambda: _fake_bot_engine(stale_data=["AAPL"]),
    )

    with pytest.raises(SystemExit) as excinfo:
        startup_runtime.run_main_startup_runtime(types.SimpleNamespace())

    assert excinfo.value.code == 1


def test_run_main_startup_runtime_stale_data_allows_explicit_override(monkeypatch):
    from ai_trading.core import startup_runtime

    be = _fake_bot_engine(stale_data=["AAPL"], allow_stale="true")
    monkeypatch.setattr(startup_runtime, "_bot_engine", lambda: be)

    ctx = types.SimpleNamespace(
        data_fetcher=types.SimpleNamespace(get_minute_df=lambda *args, **kwargs: None),
        api=types.SimpleNamespace(list_positions=lambda: []),
        _rebalance_done=True,
    )

    startup_runtime.run_main_startup_runtime(ctx)
