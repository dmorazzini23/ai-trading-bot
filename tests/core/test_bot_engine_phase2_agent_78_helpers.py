from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from ai_trading.core import bot_engine


class _BadStr:
    def __str__(self) -> str:
        raise ValueError("bad string")


def _patch_get_env(monkeypatch: pytest.MonkeyPatch, values: dict[str, Any]) -> None:
    def _fake_get_env(key: str, default: Any = None, *, cast: Any = None) -> Any:
        value = values.get(key, default)
        if isinstance(value, BaseException):
            raise value
        if cast is not None and value is not None:
            return cast(value)
        return value

    monkeypatch.setattr(bot_engine, "get_env", _fake_get_env)


def test_order_normalizers_cover_enum_bad_values_and_nested_fill_timestamp() -> None:
    naive_fill = datetime(2026, 4, 27, 12, 30)
    nested = SimpleNamespace(order={"filled_at": naive_fill, "status": "FILLED"})

    assert bot_engine._normalize_order_side_value(bot_engine.CoreOrderSide.BUY) == "buy"
    assert bot_engine._normalize_order_side_value(bot_engine.CoreOrderSide.SELL_SHORT) == "sell_short"
    assert bot_engine._normalize_order_side_value(bot_engine.CoreOrderSide.SELL) == "sell"
    assert bot_engine._normalize_order_side_value(" short ") == "sell_short"
    assert bot_engine._normalize_order_side_value("hold") is None
    assert bot_engine._normalize_broker_order_status(None) == ""
    assert bot_engine._normalize_broker_order_status(_BadStr()) == ""
    assert bot_engine._normalize_order_status_token(None) == ""
    assert bot_engine._normalize_order_status_token(_BadStr()) == ""
    assert bot_engine._normalize_order_status_token(" Alpaca.Order-Status ") == "order_status"
    assert bot_engine._extract_order_value(None, "id") is None
    assert bot_engine._extract_order_value({"id": "order-1"}, "id") == "order-1"

    fill_ts = bot_engine._extract_order_fill_timestamp(nested)

    assert fill_ts == naive_fill.replace(tzinfo=UTC)


def test_sparse_and_quote_source_helpers_cover_error_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NoLen:
        def __len__(self) -> int:
            raise TypeError("length unavailable")

    assert bot_engine._expected_minute_bars_window(_BadStr(), datetime.now(UTC)) == 0
    assert bot_engine._frame_is_sparse(None, 3) is True
    assert bot_engine._frame_is_sparse([], 0) is False
    assert bot_engine._frame_is_sparse(_NoLen(), 2) is True
    assert bot_engine._normalize_quote_source_token(_BadStr()) is None

    monkeypatch.setattr(
        bot_engine,
        "get_price_source",
        lambda _symbol: pytest.fail("metadata source should win"),
    )
    order = SimpleNamespace(metadata={"fallback_source": " Yahoo "})

    assert (
        bot_engine._resolve_quote_proxy_source(
            order,
            symbol="AAPL",
            default_source="last_trade",
        )
        == "yahoo"
    )


def test_count_trading_minutes_skips_calendar_failures_and_bad_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = datetime(2026, 4, 27, 13, 30, tzinfo=UTC)
    end = datetime(2026, 4, 27, 14, 0, tzinfo=UTC)

    monkeypatch.setattr(
        bot_engine,
        "market_calendar",
        SimpleNamespace(is_trading_day=lambda _day: False),
    )
    assert bot_engine._count_trading_minutes(start, end) == 0

    def _raise_day(_day: object) -> bool:
        raise RuntimeError("calendar down")

    monkeypatch.setattr(
        bot_engine,
        "market_calendar",
        SimpleNamespace(is_trading_day=_raise_day),
    )
    assert bot_engine._count_trading_minutes(start, end) == 0

    monkeypatch.setattr(
        bot_engine,
        "market_calendar",
        SimpleNamespace(
            is_trading_day=lambda _day: True,
            session_info=lambda _day: (_ for _ in ()).throw(RuntimeError("no session")),
        ),
    )
    assert bot_engine._count_trading_minutes(start, end) == 0

    monkeypatch.setattr(
        bot_engine,
        "market_calendar",
        SimpleNamespace(
            is_trading_day=lambda _day: True,
            session_info=lambda _day: SimpleNamespace(start_utc=end, end_utc=start),
        ),
    )
    assert bot_engine._count_trading_minutes(start, end) == 0


def test_confirmed_pending_orders_requires_and_uses_broker_confirmation(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    order = SimpleNamespace(id="ord-1", status="new")
    api_without_confirmation = SimpleNamespace()

    assert bot_engine.get_confirmed_pending_orders(None, [order]) == []
    with caplog.at_level(logging.DEBUG):
        assert (
            bot_engine.get_confirmed_pending_orders(
                api_without_confirmation,
                [order],
                require_confirmation=True,
            )
            == []
        )

    monkeypatch.setattr(bot_engine, "list_open_orders", lambda _api: [order])
    api = SimpleNamespace(get_order=lambda _order_id: SimpleNamespace(id="ord-1", status=""))

    pending = bot_engine.get_confirmed_pending_orders(api, None)

    assert [item.id for item in pending] == ["ord-1"]
    assert any(
        record.getMessage() == "PENDING_ORDER_CONFIRMATION_UNAVAILABLE"
        for record in caplog.records
    )


def test_confirmed_pending_orders_drops_terminal_refresh(
    caplog: pytest.LogCaptureFixture,
) -> None:
    order = SimpleNamespace(id="ord-2", status="pending_new")
    api = SimpleNamespace(get_order=lambda _order_id: SimpleNamespace(id="ord-2", status="filled"))

    with caplog.at_level(logging.DEBUG):
        pending = bot_engine.get_confirmed_pending_orders(api, [order])

    assert pending == []
    assert any(record.getMessage() == "PENDING_ORDER_STATUS_CHANGED" for record in caplog.records)


def test_degraded_state_and_timeframe_helpers_cover_fatal_inference() -> None:
    assert bot_engine._degrade_state((True, "provider_disabled")) == (
        True,
        "provider_disabled",
        True,
    )
    assert bot_engine._degrade_state((True, "recovering", False)) == (
        True,
        "recovering",
        False,
    )
    assert bot_engine._degrade_state(True) == (True, None, False)
    assert bot_engine._reason_implies_fatal(None) is False
    assert bot_engine._reason_implies_fatal(" ") is False
    assert bot_engine._reason_implies_fatal("safe_mode:halt") is True
    assert bot_engine._reason_implies_fatal(_BadStr()) is False
    assert bot_engine._fallback_active_for_timeframes(None, include_daily=True) is False
    assert bot_engine._fallback_active_for_timeframes({}, include_daily=True) is False
    assert (
        bot_engine._fallback_active_for_timeframes(
            {"timeframes": {"1day": True}},
            include_daily=False,
        )
        is False
    )
    assert (
        bot_engine._fallback_active_for_timeframes(
            {"timeframes": {"1day": True}},
            include_daily=True,
        )
        is True
    )
    assert (
        bot_engine._fallback_active_for_timeframes(
            {"timeframes": {"minute": True}},
            include_daily=False,
        )
        is True
    )


def test_quote_age_helpers_cover_seconds_invalid_and_env_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_get_env(monkeypatch, {"QUOTE_MAX_AGE_MS": ValueError("bad")})

    assert bot_engine._quote_age_limit_ms() == 2000.0
    assert bot_engine._quote_age_ms_from_state(None) is None
    assert bot_engine._quote_age_ms_from_state({"age_sec": "1.25"}) == 1250.0
    assert bot_engine._quote_age_ms_from_state({"quote_age_ms": "bad"}) is None
    assert bot_engine._quote_age_ms_from_state({"quote_age_ms": -1}) is None


def test_resolve_data_provider_degraded_uses_snapshot_safe_mode_and_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bot_engine.runtime_state,
        "observe_data_provider_state",
        lambda: {
            "status": "degraded",
            "timeframes": {"1Min": True},
            "using_backup": False,
            "reason": None,
        },
    )
    monkeypatch.setattr(bot_engine, "safe_mode_reason", lambda: None)
    monkeypatch.setattr(bot_engine.provider_monitor, "is_disabled", lambda _provider: False)

    assert bot_engine._resolve_data_provider_degraded() == (
        True,
        "degraded",
        False,
    )

    monkeypatch.setattr(bot_engine, "safe_mode_reason", lambda: "data_quality:minute_gap")
    assert bot_engine._resolve_data_provider_degraded() == (
        True,
        "data_quality:minute_gap",
        False,
    )

    monkeypatch.setattr(bot_engine, "safe_mode_reason", lambda: None)
    monkeypatch.setattr(bot_engine.provider_monitor, "is_disabled", lambda _provider: True)
    monkeypatch.setattr(bot_engine.runtime_state, "observe_data_provider_state", lambda: {})
    assert bot_engine._resolve_data_provider_degraded() == (
        True,
        "provider_disabled",
        True,
    )


def test_primary_feed_derisk_state_blocks_after_prolonged_degradation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SimpleNamespace(state={bot_engine._PRIMARY_FEED_DERISK_SINCE_TS_KEY: 100.0})
    _patch_get_env(
        monkeypatch,
        {
            "AI_TRADING_PRIMARY_FEED_DERISK_ENABLED": True,
            "AI_TRADING_PRIMARY_FEED_DERISK_MODE": "block",
            "AI_TRADING_PRIMARY_FEED_DERISK_AFTER_SEC": 10.0,
            "AI_TRADING_PRIMARY_FEED_DERISK_BLOCK_AFTER_SEC": 30.0,
            "AI_TRADING_PRIMARY_FEED_DERISK_SCALE_MULT": 0.25,
            "AI_TRADING_PRIMARY_FEED_DERISK_INCLUDE_DAILY": True,
            "AI_TRADING_PRIMARY_FEED_DERISK_SUPPRESS_ON_HEALTHY_BACKUP": False,
            "AI_TRADING_PRIMARY_FEED_DERISK_EXIT_ONLY_ON_DEGRADED": False,
            "QUOTE_MAX_AGE_MS": 1000.0,
        },
    )
    monkeypatch.setattr(
        bot_engine.runtime_state,
        "observe_data_provider_state",
        lambda: {
            "timeframes": {"1Min": True},
            "status": "degraded",
            "data_status": "error",
            "reason": "gap",
            "active": "yahoo",
        },
    )
    monkeypatch.setattr(
        bot_engine.runtime_state,
        "observe_quote_status",
        lambda: {"age_sec": 5, "synthetic": True},
    )
    monkeypatch.setattr(bot_engine.time, "time", lambda: 200.0)

    state = bot_engine._resolve_primary_feed_derisk_state(runtime)

    assert state["triggered"] is True
    assert state["block"] is True
    assert state["prolonged_block"] is True
    assert state["scale"] == 1.0
    assert state["quote_age_ms"] == 5000.0
    assert state["reason"] == "gap,synthetic_quote,stale_quote,prolonged_degraded_feed"


def test_pre_trade_gate_blocks_sip_and_unsafe_fallback_quotes(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _patch_get_env(monkeypatch, {"PYTEST_RUNNING": "", "ALLOW_EXECUTION_ON_FALLBACK_QUOTES": False})
    monkeypatch.setattr(bot_engine, "_sip_lockout_active", lambda: True)

    assert bot_engine._pre_trade_gate() is True

    monkeypatch.setattr(bot_engine, "_sip_lockout_active", lambda: False)
    monkeypatch.setattr(
        bot_engine.runtime_state,
        "observe_data_provider_state",
        lambda: {"status": "healthy", "timeframes": {"1Min": True}},
    )
    monkeypatch.setattr(
        bot_engine.runtime_state,
        "observe_quote_status",
        lambda: {"synthetic": True, "quote_age_ms": 5000.0, "source": "fallback"},
    )
    monkeypatch.setattr(bot_engine.provider_monitor, "is_safe_mode_active", lambda: False)
    monkeypatch.setattr(bot_engine.provider_monitor, "is_disabled", lambda _provider: False)
    monkeypatch.setattr(bot_engine, "safe_mode_reason", lambda: None)
    monkeypatch.setattr(bot_engine, "_failsoft_mode_active", lambda _provider_state=None: False)
    monkeypatch.setattr(bot_engine, "_safe_mode_blocks_trading", lambda: True)
    monkeypatch.setattr(
        bot_engine,
        "get_trading_config",
        lambda: SimpleNamespace(execution_mode="sim", safe_mode_allow_paper=False),
    )

    with caplog.at_level(logging.WARNING):
        assert bot_engine._pre_trade_gate() is True

    assert any(record.getMessage() == "EXECUTION_BLOCKED_UNSAFE_QUOTES" for record in caplog.records)


def test_pending_order_thresholds_and_warmup_state_machine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = SimpleNamespace(orders_pending_new_warn_s="bad", orders_pending_new_error_s="2")

    assert bot_engine._pending_order_log_thresholds(cfg) == (
        bot_engine._PENDING_ORDER_WARN_SEC_DEFAULT,
        bot_engine._PENDING_ORDER_WARN_SEC_DEFAULT,
    )
    assert bot_engine._pending_order_log_level(1, warn_after_s=10, error_after_s=0) == logging.ERROR
    assert bot_engine._pending_order_log_level(11, warn_after_s=10, error_after_s=20) == logging.WARNING
    assert bot_engine._pending_order_log_level(5, warn_after_s=0, error_after_s=20) == logging.WARNING
    assert bot_engine._pending_order_log_level(1, warn_after_s=10, error_after_s=20) == logging.INFO
    assert (
        bot_engine._pending_order_scope_log_level(logging.ERROR, block_scope="symbol")
        == logging.WARNING
    )

    _patch_get_env(monkeypatch, {"AI_TRADING_PENDING_CLEANUP_WARMUP_CYCLES": 2})
    runtime = SimpleNamespace(state={})

    assert (
        bot_engine._arm_pending_cleanup_warmup(
            runtime,
            source="startup",
            open_count=2,
            pending_count=1,
        )
        is True
    )
    assert bot_engine._consume_pending_cleanup_warmup(runtime, open_count=1) is True
    assert bot_engine._consume_pending_cleanup_warmup(runtime, open_count=1) is False


def test_startup_cancel_mode_identifier_subset_and_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bot_engine, "_startup_cancel_stale_seconds", lambda: 60.0)

    _patch_get_env(monkeypatch, {"AI_TRADING_STARTUP_CANCEL_MODE": "all"})
    assert bot_engine._startup_cancel_mode() == "all"
    _patch_get_env(monkeypatch, {"AI_TRADING_STARTUP_CANCEL_MODE": "stale_pending"})
    assert bot_engine._startup_cancel_mode() == "stale_only"
    _patch_get_env(
        monkeypatch,
        {
            "AI_TRADING_STARTUP_CANCEL_MODE": "",
            "AI_TRADING_STARTUP_CANCEL_UNEXPECTED_ORDERS": True,
        },
    )
    assert bot_engine._startup_cancel_mode() == "all"

    old_ts = datetime.now(UTC) - timedelta(seconds=120)
    stale_order = SimpleNamespace(id="stale-1", status="new", submitted_at=old_ts)
    active_order = SimpleNamespace(id="filled-1", status="filled", submitted_at=old_ts)

    should_cancel, details = bot_engine._startup_cancel_decision(
        [stale_order, active_order],
        mode="stale_only",
    )

    assert bot_engine._extract_order_identifier({"client_order_id": "cid-1"}) == "cid-1"
    assert bot_engine._extract_order_identifier(SimpleNamespace(id="oid-1")) == "oid-1"
    assert should_cancel is True
    assert details["reason"] == "stale_pending"
    assert details["stale_ids"] == ["stale-1"]


def test_cancel_open_orders_subset_covers_empty_missing_api_and_mixed_results() -> None:
    assert bot_engine._cancel_open_orders_subset(
        SimpleNamespace(api=None),
        orders=[],
        reason_code="test",
    ).total_open == 0

    missing_api = bot_engine._cancel_open_orders_subset(
        SimpleNamespace(api=None),
        orders=[SimpleNamespace(id="ord-1")],
        reason_code="test",
    )
    assert missing_api.failed == 1
    assert missing_api.errors == [{"error": "missing_api"}]

    missing_cancel = bot_engine._cancel_open_orders_subset(
        SimpleNamespace(api=SimpleNamespace()),
        orders=[SimpleNamespace(id="ord-1")],
        reason_code="test",
    )
    assert missing_cancel.errors == [{"error": "missing_cancel_capability"}]

    cancelled: list[str] = []

    def _cancel_order(order_id: str) -> None:
        if order_id == "bad":
            raise RuntimeError("reject")
        cancelled.append(order_id)

    mixed = bot_engine._cancel_open_orders_subset(
        SimpleNamespace(api=SimpleNamespace(cancel_order=_cancel_order)),
        orders=[
            SimpleNamespace(id="good"),
            SimpleNamespace(id=""),
            SimpleNamespace(id="bad"),
        ],
        reason_code="test",
    )

    assert cancelled == ["good"]
    assert mixed.cancelled == 1
    assert mixed.failed == 2
    assert mixed.errors == [
        {"error": "missing_order_identifier"},
        {"order_id": "bad", "error": "reject"},
    ]


def test_pending_env_helpers_clamp_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get_env(
        monkeypatch,
        {
            "AI_TRADING_PENDING_NEW_FORCE_CANCEL_SEC": 1.0,
            "AI_TRADING_PENDING_ORDERS_BLOCK_SCOPE": "per_symbol",
            "AI_TRADING_PENDING_ORDERS_WARN_AFTER_SEC": float("inf"),
            "AI_TRADING_PENDING_ORDERS_WARN_EVERY_SEC": ValueError("bad"),
            "AI_TRADING_PENDING_STALE_SWEEP_ENABLED": False,
            "AI_TRADING_PENDING_STALE_SWEEP_SEC": float("nan"),
            "AI_TRADING_PENDING_STALE_SWEEP_INCLUDE_PARTIALLY_FILLED": False,
            "AI_TRADING_PENDING_STALE_SWEEP_PARTIALLY_FILLED_SEC": 30.0,
            "AI_TRADING_PENDING_STALE_SWEEP_MAX_CANCELS": 500,
            "AI_TRADING_PENDING_STALE_SWEEP_COOLDOWN_SEC": 5000.0,
        },
    )
    monkeypatch.setattr(bot_engine, "_startup_cancel_stale_seconds", lambda: 120.0)

    assert bot_engine._pending_order_force_cleanup_seconds() == 5.0
    assert bot_engine._pending_orders_block_scope() == "symbol"
    assert bot_engine._pending_orders_warn_after_seconds() == bot_engine._PENDING_BACKLOG_WARN_AFTER_SEC_DEFAULT
    assert bot_engine._pending_orders_warn_every_seconds() == bot_engine._PENDING_BACKLOG_WARN_EVERY_SEC_DEFAULT
    assert bot_engine._pending_stale_sweep_enabled() is False
    assert bot_engine._pending_stale_sweep_age_seconds() == 120.0
    assert bot_engine._pending_stale_sweep_include_partially_filled() is False
    assert bot_engine._pending_stale_sweep_partial_fill_age_seconds(100.0) == 100.0
    assert bot_engine._pending_stale_sweep_max_cancels() == 100
    assert bot_engine._pending_stale_sweep_cooldown_seconds() == 3600.0
