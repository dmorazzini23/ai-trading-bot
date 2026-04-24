from __future__ import annotations

import threading
from datetime import UTC, datetime, time
from types import SimpleNamespace

import pandas as pd
import pytest

from ai_trading.core import bot_engine, execution_flow


def _entry_ctx(*, buying_power: object = "10000", api: object | None = None) -> SimpleNamespace:
    if api is None:
        api = SimpleNamespace(get_account=lambda: SimpleNamespace(buying_power=buying_power))
    return SimpleNamespace(
        api=api,
        trade_logger=SimpleNamespace(log_entry=lambda *_args, **_kwargs: None),
        market_open=time(9, 30),
        market_close=time(16, 0),
        stop_targets={},
        take_profit_targets={},
    )


def test_latest_quote_request_retries_without_feed_when_sdk_class_rejects_feed(monkeypatch):
    calls: list[dict[str, object]] = []

    class _QuoteRequest:
        def __init__(self, **kwargs):
            calls.append(dict(kwargs))
            if "feed" in kwargs:
                raise TypeError("feed unsupported")

    monkeypatch.setattr(bot_engine, "StockLatestQuoteRequest", _QuoteRequest)
    monkeypatch.setattr(execution_flow, "get_execution_feed", lambda: "iex")

    request = execution_flow._latest_quote_request("AAPL")  # noqa: SLF001

    assert isinstance(request, _QuoteRequest)
    assert calls == [
        {"symbol_or_symbols": ["AAPL"], "feed": "iex"},
        {"symbol_or_symbols": ["AAPL"]},
    ]


def test_twap_submit_places_slices_and_sleeps(monkeypatch):
    orders: list[tuple[str, int, str]] = []
    sleeps: list[float] = []
    monkeypatch.setattr(
        bot_engine,
        "submit_order",
        lambda _ctx, symbol, qty, side: orders.append((symbol, qty, side)),
    )
    monkeypatch.setattr(execution_flow.pytime, "sleep", lambda seconds: sleeps.append(seconds))

    execution_flow.twap_submit(SimpleNamespace(), "MSFT", 9, "buy", window_secs=6, n_slices=3)

    assert orders == [("MSFT", 3, "buy"), ("MSFT", 3, "buy"), ("MSFT", 3, "buy")]
    assert sleeps == [2.0, 2.0, 2.0]


def test_pov_submit_aborts_after_missing_data_retries(monkeypatch):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    cfg = SimpleNamespace(
        sleep_interval=1.0,
        max_retries=1,
        backoff_factor=2.0,
        max_backoff_interval=5.0,
        pct=0.10,
    )

    def _raise_no_data(_symbol):
        raise bot_engine.DataFetchError("missing")

    monkeypatch.setattr(bot_engine, "fetch_minute_df_safe", _raise_no_data)

    assert execution_flow.pov_submit(SimpleNamespace(), "AAPL", 10, "buy", cfg) is False


def test_pov_submit_records_partial_fill_summary(monkeypatch):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    cfg = SimpleNamespace(
        sleep_interval=1.0,
        max_retries=1,
        backoff_factor=2.0,
        max_backoff_interval=5.0,
        pct=0.10,
    )
    df = pd.DataFrame({"volume": [20]})
    submitted: list[int] = []

    monkeypatch.setattr(bot_engine, "fetch_minute_df_safe", lambda _symbol: df)
    monkeypatch.setattr(bot_engine, "_ensure_alpaca_classes", lambda: None)
    monkeypatch.setattr(bot_engine, "_ALPACA_IMPORT_ERROR", None, raising=False)
    monkeypatch.setattr(
        bot_engine,
        "StockLatestQuoteRequest",
        lambda **_kwargs: SimpleNamespace(),
        raising=False,
    )

    def _submit_order(_ctx, _symbol, qty, _side):
        submitted.append(qty)
        return SimpleNamespace(id="order-1", status="partially_filled", filled_qty="1")

    monkeypatch.setattr(bot_engine, "submit_order", _submit_order)
    ctx = SimpleNamespace(
        data_client=SimpleNamespace(
            get_stock_latest_quote=lambda _request: SimpleNamespace(ask_price=100.01, bid_price=100.0)
        )
    )

    assert execution_flow.pov_submit(ctx, "AAPL", 2, "buy", cfg) is True

    assert submitted == [2]
    summary = ctx.partial_fill_tracker["AAPL"]
    assert summary["total_intended"] == 2
    assert summary["total_actual"] == 1
    assert summary["fill_gap"] == 1
    assert summary["terminated_on_gap"] is True


@pytest.mark.parametrize(
    ("now_et", "expected_allowed", "expected_reason"),
    [
        (datetime(2026, 4, 25, 15, 55, tzinfo=UTC), False, "weekend"),
        (datetime(2026, 4, 24, 15, 40, tzinfo=UTC), False, "before_window"),
        (datetime(2026, 4, 24, 16, 0, tzinfo=UTC), False, "after_close"),
        (datetime(2026, 4, 24, 15, 56, tzinfo=UTC), True, "session_close_window"),
    ],
)
def test_should_trigger_eod_flatten_resolves_session_window(
    monkeypatch,
    now_et,
    expected_allowed,
    expected_reason,
):
    monkeypatch.setenv("AI_TRADING_EOD_FLATTEN_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_EOD_FLATTEN_LEAD_SECONDS", "300")

    allowed, context = execution_flow._should_trigger_eod_flatten(now_et)  # noqa: SLF001

    assert allowed is expected_allowed
    assert context["reason"] == expected_reason
    assert context["enabled"] is True
    assert context["lead_seconds"] == 300


def test_liquidate_positions_if_needed_skips_missing_api(monkeypatch):
    monkeypatch.setattr(bot_engine, "check_halt_flag", lambda _runtime: False)
    monkeypatch.setattr(
        execution_flow,
        "_should_trigger_eod_flatten",
        lambda: (True, {"reason": "session_close_window"}),
    )

    execution_flow.liquidate_positions_if_needed(SimpleNamespace())


def test_liquidate_positions_if_needed_handles_position_listing_failure(monkeypatch):
    monkeypatch.setattr(bot_engine, "check_halt_flag", lambda _runtime: False)
    monkeypatch.setattr(
        execution_flow,
        "_should_trigger_eod_flatten",
        lambda: (True, {"reason": "session_close_window"}),
    )
    runtime = SimpleNamespace(
        api=SimpleNamespace(list_positions=lambda: (_ for _ in ()).throw(OSError("boom")))
    )

    execution_flow.liquidate_positions_if_needed(runtime)


def test_liquidate_positions_if_needed_returns_when_no_active_positions(monkeypatch):
    monkeypatch.setattr(bot_engine, "check_halt_flag", lambda _runtime: False)
    monkeypatch.setattr(
        execution_flow,
        "_should_trigger_eod_flatten",
        lambda: (True, {"reason": "session_close_window"}),
    )
    calls: list[str] = []
    monkeypatch.setattr(execution_flow, "exit_all_positions", lambda _runtime: calls.append("exit"))
    runtime = SimpleNamespace(api=SimpleNamespace(list_positions=lambda: [SimpleNamespace(qty="0")]))

    execution_flow.liquidate_positions_if_needed(runtime)

    assert calls == []


def test_liquidate_positions_if_needed_throttles_recent_attempt(monkeypatch):
    monkeypatch.setattr(bot_engine, "check_halt_flag", lambda _runtime: False)
    monkeypatch.setattr(
        execution_flow,
        "_should_trigger_eod_flatten",
        lambda: (True, {"reason": "session_close_window"}),
    )
    monkeypatch.setattr(execution_flow.pytime, "monotonic", lambda: 100.0)
    calls: list[str] = []
    monkeypatch.setattr(execution_flow, "exit_all_positions", lambda _runtime: calls.append("exit"))
    runtime = SimpleNamespace(
        _last_eod_flatten_attempt_mono=60.0,
        api=SimpleNamespace(list_positions=lambda: [SimpleNamespace(qty="5")]),
    )

    execution_flow.liquidate_positions_if_needed(runtime)

    assert calls == []


def test_execute_entry_returns_when_api_missing(monkeypatch):
    monkeypatch.setattr(bot_engine, "get_trade_logger", lambda: SimpleNamespace())
    ctx = _entry_ctx(api=None)
    ctx.api = None

    execution_flow.execute_entry(ctx, "AAPL", 1, "buy")

    assert ctx.trade_logger is not None


def test_execute_entry_returns_when_buying_power_exhausted(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(bot_engine, "get_trade_logger", lambda: SimpleNamespace())
    monkeypatch.setattr(bot_engine, "submit_order", lambda *_args, **_kwargs: calls.append("order"))

    execution_flow.execute_entry(_entry_ctx(buying_power="0"), "AAPL", 1, "buy")

    assert calls == []


def test_execute_entry_rejects_invalid_quantity(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(bot_engine, "get_trade_logger", lambda: SimpleNamespace())
    monkeypatch.setattr(bot_engine, "submit_order", lambda *_args, **_kwargs: calls.append("order"))

    execution_flow.execute_entry(_entry_ctx(), "AAPL", 0, "buy")

    assert calls == []


def test_execute_entry_returns_before_order_when_account_unavailable(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(bot_engine, "get_trade_logger", lambda: SimpleNamespace())
    monkeypatch.setattr(bot_engine, "submit_order", lambda *_args, **_kwargs: calls.append("order"))
    ctx = SimpleNamespace(api=SimpleNamespace(get_account=lambda: (_ for _ in ()).throw(OSError("down"))))

    execution_flow.execute_entry(ctx, "AAPL", 1, "buy")

    assert calls == []


def test_execute_entry_routes_large_order_to_pov(monkeypatch):
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(bot_engine, "get_trade_logger", lambda: SimpleNamespace())
    monkeypatch.setattr(bot_engine, "POV_SLICE_PCT", 0.1)
    monkeypatch.setattr(bot_engine, "SLICE_THRESHOLD", 10)
    monkeypatch.setattr(
        bot_engine,
        "pov_submit",
        lambda _ctx, symbol, qty, _side: calls.append((symbol, qty)),
    )
    monkeypatch.setattr(
        bot_engine,
        "fetch_minute_df_safe",
        lambda _symbol: (_ for _ in ()).throw(bot_engine.DataFetchError("missing")),
    )

    execution_flow.execute_entry(_entry_ctx(), "AAPL", 11, "buy")

    assert calls == [("AAPL", 11)]


def test_execute_entry_routes_large_order_to_vwap(monkeypatch):
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(bot_engine, "get_trade_logger", lambda: SimpleNamespace())
    monkeypatch.setattr(bot_engine, "POV_SLICE_PCT", 0)
    monkeypatch.setattr(bot_engine, "SLICE_THRESHOLD", 10)
    monkeypatch.setattr(
        bot_engine,
        "vwap_pegged_submit",
        lambda _ctx, symbol, qty, _side: calls.append((symbol, qty)),
    )
    monkeypatch.setattr(
        bot_engine,
        "fetch_minute_df_safe",
        lambda _symbol: pd.DataFrame(),
    )

    execution_flow.execute_entry(_entry_ctx(), "AAPL", 11, "buy")

    assert calls == [("AAPL", 11)]


def test_execute_entry_handles_indicator_preparation_failure(monkeypatch):
    orders: list[str] = []
    monkeypatch.setattr(bot_engine, "get_trade_logger", lambda: SimpleNamespace())
    monkeypatch.setattr(bot_engine, "POV_SLICE_PCT", 0)
    monkeypatch.setattr(bot_engine, "SLICE_THRESHOLD", 10)
    monkeypatch.setattr(bot_engine, "submit_order", lambda *_args, **_kwargs: orders.append("order"))
    monkeypatch.setattr(
        bot_engine,
        "fetch_minute_df_safe",
        lambda _symbol: pd.DataFrame({"close": [100.0], "atr": [1.0]}),
    )
    monkeypatch.setattr(
        bot_engine,
        "prepare_indicators",
        lambda _raw: (_ for _ in ()).throw(ValueError("bad indicators")),
    )

    execution_flow.execute_entry(_entry_ctx(), "AAPL", 1, "buy")

    assert orders == ["order"]


def test_execute_exit_clears_targets_when_trade_log_fails(monkeypatch):
    monkeypatch.setattr(bot_engine, "fetch_minute_df_safe", lambda _symbol: SimpleNamespace())
    monkeypatch.setattr(bot_engine, "get_latest_close", lambda _raw: 101.5)
    exit_calls: list[tuple[str, int, float, str]] = []
    monkeypatch.setattr(
        bot_engine,
        "send_exit_order",
        lambda _ctx, symbol, qty, price, reason: exit_calls.append((symbol, qty, price, reason)),
    )
    monkeypatch.setattr(bot_engine, "targets_lock", threading.Lock(), raising=False)
    ctx = SimpleNamespace(
        trade_logger=SimpleNamespace(log_exit=lambda *_args: (_ for _ in ()).throw(OSError("log"))),
        take_profit_targets={"AAPL": 110.0},
        stop_targets={"AAPL": 95.0},
    )

    execution_flow.execute_exit(ctx, SimpleNamespace(), "AAPL", 3)

    assert exit_calls == [("AAPL", 3, 101.5, "manual_exit")]
    assert ctx.take_profit_targets == {}
    assert ctx.stop_targets == {}


def test_execute_exit_rejects_non_positive_quantity(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(bot_engine, "send_exit_order", lambda *_args, **_kwargs: calls.append("exit"))

    execution_flow.execute_exit(SimpleNamespace(), SimpleNamespace(), "AAPL", 0)

    assert calls == []
