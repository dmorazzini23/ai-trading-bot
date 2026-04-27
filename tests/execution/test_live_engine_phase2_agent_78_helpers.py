from __future__ import annotations

from collections import deque
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from ai_trading.core.enums import OrderSide, OrderStatus, OrderType
from ai_trading.execution import engine as core_engine
from ai_trading.execution import live_trading
from ai_trading.math.money import Money
from ai_trading.monitoring.order_health_monitor import OrderInfo


class _FakeIntentStore:
    def __init__(self, intents: list[SimpleNamespace] | None = None) -> None:
        self.intents = {intent.intent_id: intent for intent in intents or []}
        self.created: list[dict[str, Any]] = []
        self.claimed: list[tuple[str, int]] = []
        self.submit_errors: list[tuple[str, str]] = []
        self.submitted: list[tuple[str, str]] = []
        self.fills: dict[str, list[SimpleNamespace]] = {}
        self.closed: list[tuple[str, str, str | None]] = []

    def create_intent(self, **kwargs: Any) -> tuple[SimpleNamespace, bool]:
        self.created.append(dict(kwargs))
        intent = SimpleNamespace(
            intent_id=kwargs["intent_id"],
            idempotency_key=kwargs["idempotency_key"],
            symbol=kwargs["symbol"],
            side=kwargs["side"],
            quantity=kwargs["quantity"],
            status="PENDING_SUBMIT",
            broker_order_id=None,
            updated_at=datetime.now(UTC).isoformat(),
            metadata_json="{}",
        )
        self.intents[intent.intent_id] = intent
        return intent, True

    def claim_for_submit(self, intent_id: str, *, stale_after_seconds: int) -> None:
        self.claimed.append((intent_id, stale_after_seconds))
        self.intents[intent_id].status = "SUBMITTING"

    def get_intent(self, intent_id: str) -> SimpleNamespace | None:
        return self.intents.get(intent_id)

    def get_open_intents(self) -> list[SimpleNamespace]:
        return list(self.intents.values())

    def record_submit_error(self, intent_id: str, error: str) -> None:
        self.submit_errors.append((intent_id, error))
        self.intents[intent_id].status = "PENDING_SUBMIT"

    def mark_submitted(self, intent_id: str, broker_order_id: str) -> None:
        self.submitted.append((intent_id, broker_order_id))
        self.intents[intent_id].broker_order_id = broker_order_id
        self.intents[intent_id].status = "SUBMITTED"

    def record_fill(
        self,
        intent_id: str,
        *,
        fill_qty: float,
        fill_price: float | None,
    ) -> None:
        self.fills.setdefault(intent_id, []).append(
            SimpleNamespace(fill_qty=fill_qty, fill_price=fill_price),
        )
        self.intents[intent_id].status = "PARTIALLY_FILLED"

    def list_fills(self, intent_id: str) -> list[SimpleNamespace]:
        return list(self.fills.get(intent_id, ()))

    def close_intent(
        self,
        intent_id: str,
        *,
        final_status: str,
        last_error: str | None = None,
    ) -> None:
        self.closed.append((intent_id, final_status, last_error))
        self.intents[intent_id].status = final_status


def _intent(
    intent_id: str,
    status: str,
    *,
    broker_order_id: str | None = None,
    quantity: float = 1.0,
    updated_at: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        intent_id=intent_id,
        idempotency_key=f"key-{intent_id}",
        symbol="AAPL",
        side="buy",
        quantity=quantity,
        status=status,
        broker_order_id=broker_order_id,
        updated_at=updated_at or datetime.now(UTC).isoformat(),
        metadata_json="{}",
    )


def test_live_fill_source_and_submit_no_result_fingerprints() -> None:
    assert live_trading._normalize_fill_source("") == "live"
    assert live_trading._normalize_fill_source("broker_reconcile_poll") == "live"
    assert live_trading._normalize_fill_source("backfill") == "reconcile_backfill"
    assert live_trading._is_live_fill_source("manual_probe") is True
    assert live_trading._is_live_fill_source("reconcile_backfill") is False

    assert live_trading._is_submit_no_result_reason("submit_no_result_timeout")
    assert (
        live_trading._submit_no_result_error_fingerprint(
            reason="submit_no_result",
            detail="missing_client_order_id in response",
            status_code=504,
        )
        == "504:missing_client_order_id"
    )
    assert (
        live_trading._submit_no_result_error_fingerprint(
            reason="submit_no_result",
            detail="connection reset by peer",
            error_code="EPIPE",
        )
        == "EPIPE:connection"
    )
    assert (
        live_trading._submit_no_result_error_fingerprint(
            reason="ordinary_reject",
            detail="timeout",
            status_code=504,
        )
        is None
    )


def test_live_backup_quote_and_broker_kwarg_helpers() -> None:
    accepted, details = live_trading._maybe_accept_backup_quote(
        {
            "fallback_quote_age": "0.4",
            "fallback_quote_limit": "1.0",
            "gap_limit": "0.10",
            "fallback_quote_timestamp": "2026-04-27T14:30:00+00:00",
        },
        provider_hint="polygon",
        gap_ratio_value=0.02,
        min_quote_fresh_ms=900.0,
        quote_age_ms=None,
        quote_timestamp_present=False,
    )
    assert accepted is True
    assert details["provider"] == "polygon"
    assert details["quote_source"] == "backup"
    assert details["age_limit_ms"] == 900.0

    rejected, rejected_details = live_trading._maybe_accept_backup_quote(
        {"fallback_quote_age": "0.1", "gap_limit": "0.01"},
        provider_hint="alpaca_iex",
        gap_ratio_value=0.001,
        min_quote_fresh_ms=2000.0,
        quote_age_ms=None,
        quote_timestamp_present=True,
    )
    assert (rejected, rejected_details) == (False, {})
    rejected, _ = live_trading._maybe_accept_backup_quote(
        {"fallback_quote_age": "0.1", "gap_limit": "0.01"},
        provider_hint="backup",
        gap_ratio_value=0.02,
        min_quote_fresh_ms=2000.0,
        quote_age_ms=None,
        quote_timestamp_present=True,
    )
    assert rejected is False

    assert live_trading._broker_kwargs_for_route("market", {"limit_price": 10}) == {}
    assert live_trading._broker_kwargs_for_route(
        "stop_limit",
        {
            "asset_class": "us_equity",
            "extended_hours": False,
            "limit_price": 10.01,
            "stop_limit_price": 10.02,
            "diagnostic": "drop-me",
        },
    ) == {
        "asset_class": "us_equity",
        "extended_hours": False,
        "limit_price": 10.01,
        "stop_limit_price": 10.02,
    }

    holder = SimpleNamespace(_pending_order_kwargs={"limit_price": 9.9, "tag": "old"})
    merged = live_trading._merge_pending_order_kwargs(holder, {"tag": "new"})
    assert merged == {"limit_price": 9.9, "tag": "new"}
    assert not hasattr(holder, "_pending_order_kwargs")


def test_live_status_payload_error_and_retry_helpers() -> None:
    assert live_trading.apply_order_status(None, "OrderStatus.PENDING_NEW") == (
        "pending_new",
        True,
    )
    assert live_trading.apply_order_status("accepted", "partially_filled") == (
        "partially_filled",
        True,
    )
    assert live_trading.apply_order_status("filled", "accepted") == ("filled", False)
    assert live_trading.apply_order_status("accepted", "acknowledged") == (
        "acknowledged",
        True,
    )

    order_obj, status, filled_qty, requested_qty, order_id, client_id = (
        live_trading._normalize_order_payload(
            {
                "order_id": "ord-1",
                "client_order_id": "cid-1",
                "status": "filled",
                "filled_quantity": "2.5",
                "requested_quantity": "",
                "symbol": "msft",
                "side": "buy",
            },
            qty_fallback=7,
        )
    )
    assert order_obj.id == "ord-1"
    assert (status, filled_qty, requested_qty, order_id, client_id) == (
        "filled",
        2.5,
        7.0,
        "ord-1",
        "cid-1",
    )

    err = SimpleNamespace(
        __class__=RuntimeError,
        message="client_order_id already exists",
        code=42210000,
        status_code=422,
    )
    metadata = live_trading._extract_api_error_metadata(err)
    assert metadata["detail"] == "client_order_id already exists"
    assert metadata["code"] == 42210000
    assert live_trading._is_duplicate_client_order_id_error(err)
    missing = SimpleNamespace(message="order not found", status_code=404)
    assert live_trading._is_missing_order_lookup_error(missing)
    assert live_trading._parse_retry_after_seconds("-3") == 0.0
    assert live_trading._parse_retry_after_seconds("not a date") is None
    assert live_trading._classify_rejection_reason("market is closed now") == (
        "market_closed"
    )
    assert live_trading._should_retry_limit_as_market(
        {"status_code": 422, "detail": "limit price outside price band"},
        using_fallback_price=True,
    )
    assert not live_trading._should_retry_limit_as_market(
        {"status_code": 422, "detail": "limit price outside price band"},
        using_fallback_price=False,
    )


def test_live_short_sale_precheck_and_capacity_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(live_trading, "_allow_shorts_configured", lambda: True)
    monkeypatch.setattr(live_trading, "_LONG_ONLY_ACCOUNT_MODE", False)
    monkeypatch.setattr(live_trading, "_LONG_ONLY_ACCOUNT_REASON", None)
    account = SimpleNamespace(shorting_enabled=True, margin_enabled=True)

    allowed, extras, reason = live_trading._short_sale_precheck(
        None,
        SimpleNamespace(
            get_asset=lambda _symbol: SimpleNamespace(
                shortable=False,
                easy_to_borrow=True,
                marginable=True,
            ),
        ),
        symbol="XYZ",
        side="sell",
        quantity=3,
        closing_position=False,
        account_snapshot=account,
    )
    assert allowed is False
    assert reason == "shortability"
    assert extras["reason"] == "asset_not_shortable"

    assert live_trading._short_sale_precheck(
        None,
        None,
        symbol="XYZ",
        side="sell",
        quantity=3,
        closing_position=True,
        account_snapshot=None,
    ) == (True, None, None)

    monkeypatch.setattr(live_trading, "_config_int", lambda _name, default: default)
    monkeypatch.setattr(live_trading, "_config_decimal", lambda _name, default: default)
    monkeypatch.setattr(live_trading, "_config_float", lambda _name, default: default)
    monkeypatch.setattr(
        live_trading,
        "_config_int_alias",
        lambda _names, default=None: default,
    )
    broker = SimpleNamespace(
        list_orders=lambda status="open": [
            {"symbol": "AAPL", "side": "buy", "qty": "2", "limit_price": "20"}
        ],
    )
    downsized = live_trading.preflight_capacity(
        "AAPL",
        "buy",
        "20",
        10,
        broker,
        account=SimpleNamespace(buying_power="100"),
    )
    assert downsized == live_trading.CapacityCheck(True, 3, None)
    sell_close = live_trading.preflight_capacity(
        "AAPL",
        "sell",
        "20",
        10,
        broker,
        account=None,
    )
    assert sell_close == live_trading.CapacityCheck(True, 10, None)
    assert live_trading._capacity_precheck_side("sell", closing_position=False) == (
        "sell_short"
    )
    assert live_trading._is_capacity_exhaustion_reason(
        "insufficient_day_trading_buying_power",
    )


def test_live_cooldown_and_snapshot_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 1000.0
    monkeypatch.setattr(live_trading, "monotonic_time", lambda: now)
    monkeypatch.setattr(
        live_trading,
        "_resolve_bool_env",
        lambda name: False
        if name.endswith("REGIME_ADAPTIVE_ENABLED")
        else True,
    )
    monkeypatch.setattr(
        live_trading,
        "_config_float",
        lambda name, default: 60.0
        if "SYMBOL_REENTRY_COOLDOWN_SEC" in name
        else default,
    )

    engine = object.__new__(live_trading.ExecutionEngine)
    engine._symbol_reentry_cooldown_until = {}
    engine._precheck_failure_pressure_context = {}
    engine._arm_symbol_reentry_cooldown_from_fill(symbol="aapl", side="long")
    allowed, context = engine._symbol_reentry_cooldown_allows_opening(
        symbol="AAPL",
        side="buy",
    )
    assert allowed is False
    assert context["reason"] == "symbol_reentry_cooldown"
    assert context["remaining_seconds"] == 60.0

    engine._symbol_loss_cooldown_until = {"MSFT": now + 20.0}
    engine._symbol_loss_cooldown_reason = {"MSFT": "realized_loss_streak"}
    allowed, context = engine._symbol_loss_cooldown_allows_opening(symbol="msft")
    assert allowed is False
    assert context["cooldown_reason"] == "realized_loss_streak"

    engine._symbol_submit_no_result_backoff_until = {}
    engine._symbol_submit_no_result_backoff_reason = {}
    engine._symbol_submit_no_result_last_cluster_count = {}
    engine._symbol_submit_no_result_last_fingerprint = {"TSLA": "504:timeout"}
    engine._submit_no_result_fingerprint_backoff_until = {"504:timeout": now + 30.0}
    engine._submit_no_result_fingerprint_last_cluster_count = {"504:timeout": 3}
    allowed, context = engine._symbol_submit_no_result_backoff_allows_opening(
        symbol="tsla",
    )
    assert allowed is False
    assert context["reason"] == "submit_no_result_fingerprint_backoff"
    assert context["remaining_seconds"] == 30.0

    engine._broker_sync = None
    engine._open_order_qty_index = {}
    engine._position_tracker = {}
    engine._runtime_snapshot_persistence_enabled = False
    snapshot = engine._update_broker_snapshot(
        [
            {"symbol": "aapl", "side": "buy", "qty": "2"},
            SimpleNamespace(symbol="AAPL", side="sell_short", remaining_qty="1.5"),
            {"symbol": "MSFT", "side": "ignored", "qty": "9"},
        ],
        [
            {"symbol": "aapl", "qty": "5", "side": "long"},
            SimpleNamespace(symbol="msft", qty="4", side="short"),
        ],
    )
    assert snapshot.open_buy_by_symbol == {"AAPL": 2.0}
    assert snapshot.open_sell_by_symbol == {"AAPL": 1.5}
    assert engine.open_order_totals("aapl") == (2.0, 1.5)
    assert engine._position_tracker == {"AAPL": 5, "MSFT": -4}


def test_core_execution_result_order_and_validation_helpers() -> None:
    assert core_engine._ensure_positive_qty("3") == 3.0
    with pytest.raises(ValueError, match="invalid_qty"):
        core_engine._ensure_positive_qty(0)
    assert core_engine._ensure_valid_price("10.5") == 10.5
    with pytest.raises(ValueError, match="invalid_price"):
        core_engine._ensure_valid_price(float("nan"))
    assert core_engine._normalize_order_side("buy-to-cover") is OrderSide.BUY
    assert core_engine._normalize_order_side("sell short") is OrderSide.SELL_SHORT
    assert core_engine._normalize_order_side("mystery") is None
    assert core_engine._deterministic_fill_jitter_ratio("AAPL", "buy") == (
        core_engine._deterministic_fill_jitter_ratio("AAPL", "buy")
    )

    order = core_engine.Order(
        "AAPL",
        OrderSide.BUY,
        4,
        price=Money("10"),
        id="ord-core",
        expected_price="10.25",
    )
    assert order.remaining_quantity == 4
    order.add_fill(2, Money("10"))
    assert order.status is OrderStatus.PARTIALLY_FILLED
    order.add_fill(2, Money("10.50"))
    assert order.status is OrderStatus.FILLED
    assert order.fill_percentage == 100
    assert order.cancel("too late") is False

    result = core_engine.ExecutionResult(order, "filled", "2", "4", 0.8)
    assert str(result) == "ord-core"
    assert result.side == "buy"
    assert result.symbol == "AAPL"
    assert result.has_fill is True
    assert result.fill_ratio == 0.5
    assert result.filled_weight == 0.4
    unknown_side = SimpleNamespace(id="x", side="exit", symbol=" spy ")
    assert core_engine.ExecutionResult(unknown_side, "bad", "nan", 0, None).side == (
        "sell"
    )


def test_core_algorithm_slices_and_broker_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = object.__new__(core_engine.ExecutionEngine)
    engine.logger = core_engine.logger
    engine._runtime_snapshot_persistence_enabled = False
    engine._runtime_snapshot_store_init_failed = False
    engine._runtime_snapshot_store = None
    engine._runtime_snapshot_source = "test"
    engine._open_order_qty_index = {}
    engine._broker_sync = None
    engine._position_tracker = {}
    monkeypatch.setattr(core_engine, "monotonic_time", lambda: 123.0)

    assert core_engine.ExecutionEngine._allocate_weighted_quantities(
        7,
        [0.5, 1.5, 0, "bad"],
    ) == [2, 5]
    slices, meta = engine._build_algorithmic_slices(
        algorithm=core_engine.ExecutionAlgorithm.VWAP,
        total_quantity=9,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        kwargs={"volume_profile": [1, 2], "limit_price": "12.34"},
    )
    assert slices == [
        {"qty": 3, "order_type": "limit", "limit_price": 12.34},
        {"qty": 6, "order_type": "limit", "limit_price": 12.34},
    ]
    assert meta["benchmark_style"] == "vwap"
    assert meta["volume_profile_name"] == "custom"

    snapshot = engine._update_broker_snapshot(
        [
            {"symbol": "aapl", "side": "long", "remaining_qty": "3"},
            SimpleNamespace(symbol="AAPL", side="short", qty="-2"),
            {"symbol": "", "side": "buy", "qty": "99"},
        ],
        [
            {"symbol": "aapl", "qty": "8", "side": "long"},
            SimpleNamespace(symbol="tsla", quantity="2.5", side="short"),
        ],
    )
    assert snapshot.timestamp == 123.0
    assert snapshot.open_buy_by_symbol == {"AAPL": 3.0}
    assert snapshot.open_sell_by_symbol == {"AAPL": 2.0}
    assert engine.open_order_totals("AAPL") == (3.0, 2.0)
    assert engine._position_tracker == {"AAPL": 8.0, "TSLA": -2.5}


def test_core_order_manager_external_lifecycle_and_reconcile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        core_engine,
        "get_env",
        lambda _name, default=None, cast=None: cast(default) if cast else default,
    )
    manager = core_engine.OrderManager()
    store = _FakeIntentStore()
    manager.configure_intent_store(store)  # type: ignore[arg-type]

    intent_id = manager.begin_external_order_lifecycle(
        intent_id="intent-1",
        idempotency_key="key-1",
        symbol="aapl",
        side="BUY",
        quantity=3,
        decision_ts="2026-04-27T14:00:00+00:00",
        metadata={"route": "limit"},
        stale_after_seconds=12,
    )
    assert intent_id == "intent-1"
    assert store.created[0]["symbol"] == "AAPL"
    assert store.created[0]["side"] == "buy"
    assert store.claimed == [("intent-1", 12)]

    assert manager.record_external_submit_error(
        order_id="intent-1",
        error="temporary outage",
    ) == "intent-1"
    assert store.submit_errors[-1] == ("intent-1", "temporary outage")

    assert manager.sync_external_order_state(
        intent_id="intent-1",
        order_id="broker-1",
        client_order_id="cid-1",
        status="filled",
        filled_qty="3",
        fill_price="10.5",
    ) == "intent-1"
    assert store.submitted[-1] == ("intent-1", "broker-1")
    assert store.fills["intent-1"][0].fill_qty == 3.0
    assert store.fills["intent-1"][0].fill_price == 10.5
    assert store.closed[-1] == ("intent-1", "FILLED", None)
    assert "broker-1" not in manager._intent_by_order_id
    assert "intent-1" not in manager._intent_reported_fill_qty

    old = (datetime.now(UTC) - timedelta(seconds=300)).isoformat()
    reconcile_store = _FakeIntentStore(
        [
            _intent("pending-old", "PENDING_SUBMIT", updated_at=old),
            _intent("submitting-old", "SUBMITTING", updated_at=old),
            _intent("open-match", "SUBMITTED"),
            _intent("recover-fill", "PARTIALLY_FILLED", broker_order_id="bf-1", quantity=2),
        ],
    )
    manager.configure_intent_store(reconcile_store)  # type: ignore[arg-type]
    summary = manager.reconcile_open_intents(
        broker_orders=[{"id": "bo-1", "client_order_id": "open-match"}],
        get_order_by_id_fn=lambda _order_id: SimpleNamespace(
            id="bf-1",
            client_order_id="recover-fill",
            status="filled",
            filled_qty="2",
            filled_avg_price="11.25",
        ),
    )
    assert summary["intents_checked"] == 4
    assert summary["matched_open_orders"] == 1
    assert summary["marked_submitted"] == 1
    assert summary["marked_failed"] == 2
    assert ("open-match", "bo-1") in reconcile_store.submitted
    assert ("recover-fill", "FILLED", None) in reconcile_store.closed
    assert reconcile_store.fills["recover-fill"][0].fill_qty == 2.0


def test_core_stale_order_cleanup_calls_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    core_engine._active_orders.clear()
    core_engine._active_orders.update(
        {
            "fresh": OrderInfo("fresh", "AAPL", "buy", 1, 95.0, "new"),
            "stale": OrderInfo("stale", "MSFT", "sell", 1, 10.0, "new"),
        },
    )
    canceled: list[str] = []
    engine = object.__new__(core_engine.ExecutionEngine)
    engine.broker_interface = SimpleNamespace(
        get_order=lambda order_id: SimpleNamespace(status="new", id=order_id),
        cancel_order=lambda order_id: canceled.append(order_id),
    )
    removed = engine.cleanup_stale_orders(now=100.0, max_age_seconds=60)
    assert removed == 1
    assert canceled == ["stale"]
    assert set(core_engine._active_orders) == {"fresh"}
    core_engine._active_orders.clear()
