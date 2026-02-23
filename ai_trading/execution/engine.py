"""
Core execution engine for institutional order management.

Provides order lifecycle management, execution algorithms,
and real-time execution monitoring with institutional controls.
"""

from __future__ import annotations
from ai_trading.logging import get_logger
import builtins
import hashlib
import math
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, is_dataclass, replace
from datetime import UTC, datetime
from enum import Enum
from typing import Any, TYPE_CHECKING
from types import SimpleNamespace

try:  # pragma: no cover - Alpaca SDK optional in tests
    from alpaca.common.exceptions import APIError
except Exception:  # ImportError

    class APIError(Exception):
        """Fallback when Alpaca SDK is unavailable."""


from ai_trading.logging.emit_once import emit_once
from ai_trading.metrics import CollectorRegistry, get_counter, get_registry, register_reset_hook
from ai_trading.config.management import get_env
from ai_trading.oms.intent_store import IntentStore
from ai_trading.utils.time import monotonic_time, safe_utcnow
from ai_trading.meta_learning.persistence import record_trade_fill

logger = get_logger(__name__)
ORDER_STALE_AFTER_S = 8 * 60


@dataclass(slots=True)
class BrokerSyncResult:
    """Snapshot of broker open orders and positions after synchronization."""

    open_orders: tuple[Any, ...]
    positions: tuple[Any, ...]
    open_buy_by_symbol: dict[str, float]
    open_sell_by_symbol: dict[str, float]
    timestamp: float

# Lightweight Prometheus counters (no-op if client unavailable)
ORDERS_SUBMITTED: Any | None = None
_orders_submitted_total: Any | None = None


def _ensure_orders_submitted_metric(registry: CollectorRegistry):
    collectors = getattr(registry, "_names_to_collectors", None)
    if collectors is None:
        collectors = {}
        setattr(registry, "_names_to_collectors", collectors)
    metric = get_counter(
        "orders_submitted_total",
        "Orders submitted total",
        registry=registry,
    )
    collectors.setdefault("orders_submitted_total", metric)
    globals()["ORDERS_SUBMITTED"] = metric
    globals()["_orders_submitted_total"] = metric
    return metric


_metrics_registry = get_registry()
ORDERS_SUBMITTED = _ensure_orders_submitted_metric(_metrics_registry)
register_reset_hook(_ensure_orders_submitted_metric)
_orders_rejected_total = get_counter("orders_rejected_total", "Orders rejected")
_orders_duplicate_total = get_counter("orders_duplicate_total", "Duplicate orders prevented")


def _safe_counter_inc(counter: Any | None, metric_name: str, *, extra: Mapping[str, Any] | None = None) -> None:
    """Increment a metric without raising runtime exceptions."""

    if counter is None:
        return
    payload: dict[str, Any] = {"metric": metric_name}
    if extra:
        payload.update(extra)
    try:
        counter.inc()
    except Exception as exc:
        logger.debug("METRIC_INCREMENT_FAILED", extra=payload, exc_info=exc)

from ai_trading.monitoring.order_health_monitor import (
    OrderInfo,
    _active_orders as _mon_active,
    _order_tracking_lock as _mon_lock,
)

_active_orders = _mon_active
_order_tracking_lock = _mon_lock

_PRIMARY_FALLBACK_SOURCE = "unknown"


KNOWN_EXECUTE_ORDER_KWARGS: frozenset[str] = frozenset(
    {
        "allow_partial",
        "asset_class",
        "client_order_id",
        "closing_position",
        "execution_algorithm",
        "expected_price",
        "expected_price_source",
        "expected_price_timestamp",
        "expected_price_ts",
        "extended_hours",
        "id",
        "adv",
        "avg_daily_volume",
        "limit_price",
        "max_participation_rate",
        "max_slippage_bps",
        "max_total_slippage_bps",
        "metadata",
        "min_quantity",
        "notional",
        "notes",
        "order_class",
        "parent_order_id",
        "post_only",
        "quote",
        "price",
        "price_hint",
        "price_improvement",
        "price_source",
        "participation_mode",
        "rolling_volume",
        "using_fallback_price",
        "annotations",
        "reduce_only",
        "signal",
        "signal_weight",
        "slice_id",
        "source_system",
        "stop_loss",
        "stop_price",
        "strategy_id",
        "tag",
        "tags",
        "take_profit",
        "target_price",
        "tif",
        "time_in_force",
        "trail_percent",
        "trail_price",
        "urgency_level",
        "user_data",
        "volume",
        "volume_1d",
    }
)


def _cleanup_stale_orders(now: float | None = None, max_age_s: int | None = None) -> int:
    """Remove orders older than ``max_age_s`` and return count."""
    max_age = max_age_s if max_age_s is not None else ORDER_STALE_AFTER_S
    now_s = now if now is not None else time.time()
    removed = 0
    with _order_tracking_lock:
        for oid, info in list(_active_orders.items()):
            if now_s - info.submitted_time >= max_age:
                _active_orders.pop(oid, None)
                removed += 1
    return removed


from ai_trading.market.symbol_specs import TICK_BY_SYMBOL, get_lot_size, get_tick_size
from ai_trading.math.money import Money, round_to_lot, round_to_tick
from ..core.constants import EXECUTION_PARAMETERS
from ..core.enums import OrderSide, OrderStatus, OrderType
from .idempotency import OrderIdempotencyCache

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ai_trading.risk.engine import TradeSignal


def _ensure_positive_qty(qty: float) -> float:
    if qty is None:
        raise ValueError("qty_none")
    q = float(qty)
    if not math.isfinite(q) or q <= 0.0:
        raise ValueError(f"invalid_qty:{qty}")
    return q


def _ensure_valid_price(price: float | None) -> float | None:
    if price is None:
        return None
    p = float(price)
    if not math.isfinite(p) or p <= 0.0:
        raise ValueError(f"invalid_price:{price}")
    return p


def _normalize_order_side(side: OrderSide | str | None) -> OrderSide | None:
    """Best-effort normalization of ``side`` to :class:`OrderSide`."""

    if isinstance(side, OrderSide):
        return side
    if side is None:
        return None
    try:
        return OrderSide(str(side).lower())
    except Exception:
        logger.debug("ORDER_SIDE_NORMALIZE_FAILED", extra={"side": side}, exc_info=True)
        return None


def _as_bool(value: Any) -> bool:
    """Parse booleans consistently for env-driven flags."""

    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_bool(key: str, default: bool = False) -> bool:
    raw_default = "1" if default else "0"
    raw = get_env(key, raw_default)
    return _as_bool(raw)


def _deterministic_fill_jitter_ratio(*parts: Any) -> float:
    """Return stable fill jitter ratio in +/- 50 bps range for simulation paths."""

    # Preserve deterministic test controls that monkeypatch ``engine.hash`` without
    # using builtin ``hash()`` in production paths.
    patched_hash = globals().get("hash")
    if callable(patched_hash) and patched_hash is not builtins.hash:
        try:
            bucket = int(patched_hash("|".join(str(part) for part in parts))) % 100
        except Exception:
            bucket = 0
        return (bucket - 50) / 10000.0

    token = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(token).digest()
    bucket = int.from_bytes(digest[:2], "big") % 100
    return (bucket - 50) / 10000.0


class ExecutionAlgorithm(Enum):
    """Execution algorithm types."""

    MARKET = "market"
    LIMIT = "limit"
    VWAP = "vwap"
    TWAP = "twap"
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"
    ICEBERG = "iceberg"


class Order:
    """
    Order representation for institutional execution.

    Comprehensive order model with execution tracking,
    partial fills, and institutional metadata.
    """

    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: Money = None,
        **kwargs,
    ):
        """Initialize order with institutional parameters."""
        self.id = kwargs.get("id", str(uuid.uuid4()))
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.order_type = order_type
        tick = TICK_BY_SYMBOL.get(symbol)
        self.price = Money(price, tick) if price is not None else None
        exp = kwargs.get("expected_price")
        self.expected_price = Money(exp, tick) if exp is not None else None
        self.status = OrderStatus.PENDING
        self.filled_quantity = 0
        self.average_fill_price = Money(0)
        self.fills = []
        self.created_at = safe_utcnow()
        self._created_monotonic = kwargs.get("created_monotonic", monotonic_time())
        self.updated_at = self.created_at
        self.executed_at = None
        self.client_order_id = kwargs.get("client_order_id", f"ord_{int(time.time())}")
        self.strategy_id = kwargs.get("strategy_id")
        self.execution_algorithm = kwargs.get("execution_algorithm", ExecutionAlgorithm.MARKET)
        self.time_in_force = kwargs.get("time_in_force", "DAY")
        self.min_quantity = kwargs.get("min_quantity", 0)
        self.stop_price = kwargs.get("stop_price")
        self.target_price = kwargs.get("target_price")
        self.price_source = kwargs.get("price_source", "unknown")
        self.expected_price_source = kwargs.get("expected_price_source", self.price_source)
        self.max_participation_rate = kwargs.get("max_participation_rate", 0.1)
        self.max_slippage_bps = kwargs.get("max_slippage_bps", EXECUTION_PARAMETERS["MAX_SLIPPAGE_BPS"])
        self.urgency_level = kwargs.get("urgency_level", "normal")
        self.notes = kwargs.get("notes", "")
        self.source_system = kwargs.get("source_system", "ai_trading")
        self.parent_order_id = kwargs.get("parent_order_id")
        self.slippage_bps = 0.0
        logger.debug(f"Order created: {self.id} {self.side} {self.quantity} {self.symbol}")

    @property
    def remaining_quantity(self) -> int:
        """Get remaining quantity to fill."""
        return self.quantity - self.filled_quantity

    @property
    def fill_percentage(self) -> float:
        """Get fill percentage."""
        return self.filled_quantity / self.quantity * 100 if self.quantity > 0 else 0

    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.filled_quantity >= self.quantity

    @property
    def is_partially_filled(self) -> bool:
        """Check if order is partially filled."""
        return 0 < self.filled_quantity < self.quantity

    @property
    def notional_value(self) -> Money:
        """Calculate notional value of order with precise money math."""
        price = self.price or self.average_fill_price or Money(0)
        return Money(abs(self.quantity)) * price

    def add_fill(self, quantity: int, price: Money, timestamp: datetime = None):
        """Add a fill to the order with precise money math."""
        if timestamp is None:
            timestamp = datetime.now(UTC)
        tick = TICK_BY_SYMBOL.get(self.symbol)
        if not isinstance(price, Money):
            price = Money(price, tick)
        fill = {"quantity": quantity, "price": price, "timestamp": timestamp, "fill_id": str(uuid.uuid4())}
        self.fills.append(fill)
        self.filled_quantity += quantity
        total_value = sum((Money(f["quantity"]) * f["price"] for f in self.fills))
        self.average_fill_price = total_value / Money(self.filled_quantity) if self.filled_quantity > 0 else Money(0)
        if self.is_filled:
            self.status = OrderStatus.FILLED
            self.executed_at = timestamp
        elif self.is_partially_filled:
            self.status = OrderStatus.PARTIALLY_FILLED
        self.updated_at = timestamp
        logger.debug(f"Fill added to order {self.id}: {quantity}@{price} ({self.fill_percentage:.1f}% filled)")

    def cancel(self, reason: str = "User cancelled"):
        """Cancel the order."""
        if self.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED]:
            logger.warning(f"Cannot cancel order {self.id} in status {self.status}")
            return False
        self.status = OrderStatus.CANCELED
        self.updated_at = datetime.now(UTC)
        self.notes += f" | Cancelled: {reason}"
        logger.info(f"Order {self.id} cancelled: {reason}")
        return True

    def to_dict(self) -> dict:
        """Convert order to dictionary representation."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value if isinstance(self.side, OrderSide) else self.side,
            "quantity": self.quantity,
            "order_type": self.order_type.value if isinstance(self.order_type, OrderType) else self.order_type,
            "price": self.price,
            "expected_price": self.expected_price,
            "status": self.status.value if isinstance(self.status, OrderStatus) else self.status,
            "filled_quantity": self.filled_quantity,
            "average_fill_price": self.average_fill_price,
            "slippage_bps": self.slippage_bps,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "client_order_id": self.client_order_id,
            "strategy_id": self.strategy_id,
            "fills": self.fills,
            "notional_value": self.notional_value,
            "fill_percentage": self.fill_percentage,
        }


@dataclass
class _SignalMeta:
    """Track signal context needed for post-fill exposure updates."""

    signal: Any | None
    requested_qty: int
    signal_weight: float | None
    reported_fill_qty: int = 0


class ExecutionResult(str):
    """Rich execution response preserving backwards-compatible ``str`` semantics."""

    __slots__ = (
        "order",
        "status",
        "filled_quantity",
        "requested_quantity",
        "signal_weight",
        "reconciled",
        "ack_timed_out",
    )

    def __new__(
        cls,
        order: Order | None,
        status: OrderStatus | str | None,
        filled_quantity: float,
        requested_quantity: float,
        signal_weight: float | None,
    ) -> "ExecutionResult":
        order_id = getattr(order, "id", "") or ""
        obj = str.__new__(cls, order_id)
        obj.order = order
        obj.status = cls._normalize_status(status)
        obj.filled_quantity = cls._coerce_quantity(filled_quantity)
        obj.requested_quantity = cls._coerce_quantity(requested_quantity)
        obj.signal_weight = signal_weight
        obj.reconciled = True
        obj.ack_timed_out = False
        return obj

    @staticmethod
    def _coerce_quantity(value: Any) -> float:
        if value is None:
            return 0.0
        try:
            qty = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(qty):
            return 0.0
        return qty

    @property
    def side(self) -> str | None:
        """Return normalized textual side when available."""

        order = getattr(self, "order", None)
        if order is None:
            return None
        raw_side = getattr(order, "side", None)
        if raw_side is None:
            return None
        candidate = getattr(raw_side, "value", raw_side)
        try:
            normalized = str(candidate).strip().lower()
        except Exception:  # pragma: no cover - defensive
            logger.debug("ORDER_EVENT_SIDE_NORMALIZE_FAILED", exc_info=True)
            return None
        if not normalized:
            return None
        if normalized in {"buy", "sell"}:
            return normalized
        if normalized in {"short", "sell_short", "sell-short", "sell short", "exit"}:
            return "sell"
        if normalized in {"cover", "long"}:
            return "buy"
        return None

    @property
    def symbol(self) -> str | None:
        """Return order symbol when available."""

        order = getattr(self, "order", None)
        if order is None:
            return None
        sym = getattr(order, "symbol", None)
        if sym is None:
            return None
        try:
            text = str(sym).strip()
        except Exception:  # pragma: no cover - defensive
            logger.debug("ORDER_EVENT_SYMBOL_NORMALIZE_FAILED", exc_info=True)
            return None
        return text or None

    @staticmethod
    def _normalize_status(status: OrderStatus | str | None) -> OrderStatus | None:
        if isinstance(status, OrderStatus):
            return status
        if status is None:
            return None
        try:
            return OrderStatus(str(status))
        except Exception:
            logger.debug("ORDER_STATUS_NORMALIZE_FAILED", extra={"status": status}, exc_info=True)
            return None

    @property
    def has_fill(self) -> bool:
        """Return ``True`` when any quantity filled."""

        return self.filled_quantity > 0

    @property
    def fill_ratio(self) -> float:
        """Return ratio of filled quantity to requested quantity (0–1)."""

        if self.requested_quantity <= 0:
            return 0.0
        try:
            ratio = self.filled_quantity / self.requested_quantity
        except ZeroDivisionError:
            return 0.0
        return max(0.0, min(1.0, float(ratio)))

    @property
    def filled_weight(self) -> float | None:
        """Return proportional signal weight filled, when available."""

        if self.signal_weight is None:
            return None
        try:
            return float(self.signal_weight) * self.fill_ratio
        except (TypeError, ValueError):
            return None


class OrderManager:
    """
    Order lifecycle management for institutional execution.

    Manages order routing, execution tracking, and provides
    real-time order monitoring with institutional controls.
    """

    def __init__(self):
        """Initialize order manager."""
        self.orders: dict[str, Order] = {}
        self.active_orders: dict[str, Order] = {}
        self.execution_callbacks: list[Callable] = []
        self.max_concurrent_orders = EXECUTION_PARAMETERS.get("MAX_CONCURRENT_ORDERS", 100)
        self.order_timeout = EXECUTION_PARAMETERS.get("ORDER_TIMEOUT_SECONDS", 300)
        self.retry_attempts = EXECUTION_PARAMETERS.get("RETRY_ATTEMPTS", 3)
        self._monitor_thread = None
        self._monitor_running = False
        self._idempotency_cache: OrderIdempotencyCache | None = None
        self._intent_store: IntentStore | None = None
        self._intent_by_order_id: dict[str, str] = {}
        self._intent_reported_fill_qty: dict[str, float] = {}
        self._test_mode = bool(get_env("PYTEST_RUNNING", default=""))
        self._init_intent_store()
        emit_once(logger, "ORDER_MANAGER_INIT", "info", "OrderManager initialized")

    def _init_intent_store(self) -> None:
        """Initialize durable intent store when enabled."""

        enabled = _env_bool("AI_TRADING_OMS_INTENT_STORE_ENABLED", True)
        allow_in_tests = _env_bool("AI_TRADING_OMS_INTENT_STORE_IN_TESTS", False)
        execution_mode = str(get_env("EXECUTION_MODE", "paper") or "").strip().lower()
        database_url = str(get_env("DATABASE_URL", "") or "").strip()
        allow_sqlite_live = _env_bool("AI_TRADING_OMS_INTENT_STORE_ALLOW_SQLITE_LIVE", False)
        if not enabled:
            return
        if self._test_mode and not allow_in_tests:
            return
        if execution_mode == "live" and not database_url and not allow_sqlite_live:
            message = (
                "DATABASE_URL is required for live durable intent store. "
                "Set DATABASE_URL=postgresql+psycopg://... or explicitly opt into "
                "AI_TRADING_OMS_INTENT_STORE_ALLOW_SQLITE_LIVE=1."
            )
            logger.error(
                "OMS_INTENT_STORE_DATABASE_URL_REQUIRED",
                extra={
                    "execution_mode": execution_mode,
                    "allow_sqlite_live": allow_sqlite_live,
                },
            )
            raise RuntimeError(message)
        path = get_env("AI_TRADING_OMS_INTENT_STORE_PATH", "runtime/oms_intents.db")
        try:
            self._intent_store = IntentStore(
                path=str(path),
                url=(database_url or None),
            )
            logger.info(
                "OMS_INTENT_STORE_ENABLED",
                extra={
                    "path": str(self._intent_store.path),
                    "database_url_configured": bool(database_url),
                },
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "OMS_INTENT_STORE_INIT_FAILED",
                extra={
                    "error": str(exc),
                    "path": str(path),
                    "database_url_configured": bool(database_url),
                    "execution_mode": execution_mode,
                },
            )
            if execution_mode == "live":
                raise RuntimeError("OMS_INTENT_STORE_INIT_FAILED") from exc
            self._intent_store = None

    def configure_intent_store(self, intent_store: IntentStore | None) -> None:
        """Override intent store instance (used by tests/integration harnesses)."""

        self._intent_store = intent_store
        self._idempotency_cache = None

    @staticmethod
    def _extract_payload_value(payload: Any, *keys: str) -> Any:
        """Return first non-empty key/attribute value from mapping or object."""

        for key in keys:
            if isinstance(payload, Mapping):
                value = payload.get(key)
            else:
                value = getattr(payload, key, None)
            if value not in (None, ""):
                return value
        return None

    @staticmethod
    def _parse_iso_utc(raw: Any) -> datetime | None:
        """Best-effort ISO8601 parser returning UTC-aware datetimes."""

        if raw in (None, ""):
            return None
        text = str(raw).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)

    def reconcile_open_intents(
        self,
        *,
        broker_orders: Iterable[Any] | None = None,
        list_orders_fn: Callable[..., Iterable[Any]] | None = None,
    ) -> dict[str, int]:
        """Reconcile non-terminal intents against broker open-order state."""

        summary: dict[str, int] = {
            "intents_checked": 0,
            "matched_open_orders": 0,
            "marked_submitted": 0,
            "marked_failed": 0,
            "deferred_submitting": 0,
            "pending_submit": 0,
            "errors": 0,
        }
        if self._intent_store is None:
            return summary

        try:
            open_intents = self._intent_store.get_open_intents()
        except Exception:
            logger.debug("OMS_INTENT_RECONCILE_LOAD_FAILED", exc_info=True)
            summary["errors"] += 1
            return summary
        if not open_intents:
            return summary

        orders_payload: list[Any]
        if broker_orders is not None:
            orders_payload = list(broker_orders)
        elif callable(list_orders_fn):
            try:
                fetched = list_orders_fn(status="open")  # type: ignore[misc]
            except TypeError:
                fetched = list_orders_fn()  # type: ignore[misc]
            except Exception:
                logger.debug("OMS_INTENT_RECONCILE_BROKER_FETCH_FAILED", exc_info=True)
                summary["errors"] += 1
                fetched = ()
            orders_payload = list(fetched or ())
        else:
            orders_payload = []

        open_by_order_id: dict[str, Any] = {}
        open_by_client_order_id: dict[str, Any] = {}
        for order_payload in orders_payload:
            raw_order_id = self._extract_payload_value(
                order_payload,
                "id",
                "order_id",
            )
            if raw_order_id not in (None, ""):
                open_by_order_id[str(raw_order_id)] = order_payload
            raw_client_order_id = self._extract_payload_value(
                order_payload,
                "client_order_id",
            )
            if raw_client_order_id not in (None, ""):
                open_by_client_order_id[str(raw_client_order_id)] = order_payload

        submitting_stale_seconds = max(
            1,
            int(get_env("AI_TRADING_OMS_RECONCILE_SUBMIT_STALE_SEC", "90", cast=int)),
        )
        now_utc = safe_utcnow()
        for intent in open_intents:
            summary["intents_checked"] += 1
            intent_status = str(intent.status or "").strip().upper()

            if intent_status == "PENDING_SUBMIT":
                summary["pending_submit"] += 1
                continue

            matched_order: Any | None = None
            if intent.broker_order_id:
                matched_order = open_by_order_id.get(str(intent.broker_order_id))
            if matched_order is None:
                # Local fallback client_order_id defaults to intent_id.
                matched_order = open_by_client_order_id.get(str(intent.intent_id))
            if matched_order is not None:
                summary["matched_open_orders"] += 1
                if not intent.broker_order_id:
                    raw_order_id = self._extract_payload_value(
                        matched_order,
                        "id",
                        "order_id",
                    )
                    if raw_order_id not in (None, ""):
                        try:
                            self._intent_store.mark_submitted(
                                intent.intent_id,
                                str(raw_order_id),
                            )
                        except Exception:
                            logger.debug(
                                "OMS_INTENT_RECONCILE_MARK_SUBMITTED_FAILED",
                                exc_info=True,
                            )
                            summary["errors"] += 1
                        else:
                            summary["marked_submitted"] += 1
                continue

            # Fresh SUBMITTING intents can legitimately be in-flight and absent
            # from broker snapshots for a short interval after restart.
            if intent_status == "SUBMITTING":
                updated_at = self._parse_iso_utc(intent.updated_at)
                if updated_at is not None:
                    age_seconds = max(
                        0.0,
                        (now_utc - updated_at).total_seconds(),
                    )
                    if age_seconds < submitting_stale_seconds:
                        summary["deferred_submitting"] += 1
                        continue

            if intent_status not in {"SUBMITTED", "SUBMITTING", "PARTIALLY_FILLED"}:
                continue

            try:
                self._intent_store.close_intent(
                    intent.intent_id,
                    final_status="FAILED",
                    last_error="reconcile_missing_broker_order",
                )
            except Exception:
                logger.debug("OMS_INTENT_RECONCILE_MARK_FAILED_FAILED", exc_info=True)
                summary["errors"] += 1
            else:
                summary["marked_failed"] += 1

        if summary["intents_checked"] > 0:
            logger.info("OMS_INTENT_RECONCILE", extra=summary)
        return summary

    def _ensure_idempotency_cache(self) -> OrderIdempotencyCache:
        """Ensure idempotency cache is instantiated."""
        if self._idempotency_cache is None:
            try:
                self._idempotency_cache = OrderIdempotencyCache(
                    intent_store=self._intent_store
                )
            except (KeyError, ValueError, TypeError, RuntimeError) as e:
                logger.error("IDEMPOTENCY_CACHE_FAILED", extra={"cause": e.__class__.__name__, "detail": str(e)})
                raise
        return self._idempotency_cache

    def submit_order(self, order: Order) -> object | None:
        """
        Submit order for execution.

        Parameters
        ----------
        order:
            The :class:`Order` to submit.

        Returns
        -------
        object | None
            ``SimpleNamespace`` mirroring a broker response with fields like
            ``id`` and ``filled_qty`` when accepted, ``None`` if rejected.
            Returning an object keeps behaviour consistent with external
        broker APIs and provides useful metadata even in dry‑run tests.
        """
        try:
            if not self._test_mode and getattr(order, "order_type", None) == OrderType.MARKET:
                try:
                    import importlib

                    be = importlib.import_module("ai_trading.core.bot_engine")
                    quote_info: SimpleNamespace | None = None
                    if hasattr(be, "resolve_trade_quote"):
                        quote_info = be.resolve_trade_quote(order.symbol)
                    elif hasattr(be, "get_latest_price"):
                        price = be.get_latest_price(order.symbol)
                        source = (
                            be.get_price_source(order.symbol)
                            if hasattr(be, "get_price_source")
                            else _PRIMARY_FALLBACK_SOURCE
                        )
                        quote_info = SimpleNamespace(price=price, source=source)
                    if quote_info is not None and quote_info.price is not None:
                        tick = TICK_BY_SYMBOL.get(order.symbol)
                        order.expected_price = Money(quote_info.price, tick)
                        order.expected_price_source = getattr(quote_info, "source", order.expected_price_source)
                        if not getattr(order, "price_source", None):
                            order.price_source = getattr(quote_info, "source", order.price_source)
                        logger.debug(
                            "EXPECTED_PRICE_REFRESHED",
                            extra={
                                "order_id": order.id,
                                "symbol": order.symbol,
                                "expected_price": float(order.expected_price),
                                "price_source": getattr(quote_info, "source", "unknown"),
                            },
                        )
                except Exception as e:  # pragma: no cover - diagnostics only
                    logger.debug(
                        "EXPECTED_PRICE_REFRESH_FAILED",
                        extra={
                            "order_id": order.id,
                            "symbol": order.symbol,
                            "cause": e.__class__.__name__,
                        },
                    )
            if not self._validate_order(order):
                _safe_counter_inc(
                    _orders_rejected_total,
                    "orders_rejected_total",
                    extra={"symbol": order.symbol, "reason": "validation_failed"},
                )
                return None
            cache = self._ensure_idempotency_cache()
            key = cache.generate_key(order.symbol, order.side, order.quantity, datetime.now(UTC))
            if len(self.active_orders) >= self.max_concurrent_orders:
                logger.error(f"Cannot submit order: max concurrent orders reached ({self.max_concurrent_orders})")
                order.status = OrderStatus.REJECTED
                order.notes += " | Rejected: Max concurrent orders reached"
                _safe_counter_inc(
                    _orders_rejected_total,
                    "orders_rejected_total",
                    extra={"symbol": order.symbol, "reason": "max_concurrent_orders"},
                )
                return None
            is_duplicate, existing_order_id = cache.check_and_mark_submitted(key, order.id)
            if is_duplicate:
                logger.warning(
                    "ORDER_DUPLICATE_SKIPPED",
                    extra={
                        "symbol": order.symbol,
                        "side": getattr(order.side, "value", order.side),
                        "quantity": order.quantity,
                        "existing_order_id": existing_order_id,
                    },
                )
                order.status = OrderStatus.REJECTED
                order.notes += " | Rejected: Duplicate order detected"
                _safe_counter_inc(
                    _orders_duplicate_total,
                    "orders_duplicate_total",
                    extra={"symbol": order.symbol, "reason": "duplicate_order"},
                )
                _safe_counter_inc(
                    _orders_rejected_total,
                    "orders_rejected_total",
                    extra={"symbol": order.symbol, "reason": "duplicate_order"},
                )
                return None
            if self._intent_store is not None:
                try:
                    intent_row = self._intent_store.get_intent_by_key(key.hash())
                    if intent_row is not None:
                        self._intent_by_order_id[order.id] = intent_row.intent_id
                    else:
                        self._intent_by_order_id[order.id] = order.id
                    self._intent_reported_fill_qty.setdefault(order.id, 0.0)
                except Exception:
                    logger.debug("OMS_INTENT_LOOKUP_FAILED", exc_info=True)
                    self._intent_by_order_id[order.id] = order.id
            self.orders[order.id] = order
            self.active_orders[order.id] = order
            if not self._monitor_running:
                # Avoid starting background monitor threads automatically during
                # unit tests. Tests that need the thread can call
                # ``start_monitoring()`` explicitly.
                pytest_running = _env_bool("PYTEST_RUNNING", False)
                if not pytest_running:
                    self.start_monitoring()
            logger.info(f"Order submitted: {order.id} {order.side} {order.quantity} {order.symbol}")
            if getattr(order, "expected_price", None) is not None:
                logger.debug(
                    "ORDER_EXPECTED_PRICE",
                    extra={
                        "order_id": order.id,
                        "symbol": order.symbol,
                        "expected_price": float(order.expected_price),
                    },
                )
            self._notify_callbacks(order, "submitted")

            # Mimic a broker-style response object even when running without a
            # real broker (dry‑run).  ``filled_qty`` mirrors Alpaca's string
            # field for compatibility with existing call sites.
            _safe_counter_inc(
                _orders_submitted_total,
                "orders_submitted_total",
                extra={"symbol": order.symbol},
            )
            return SimpleNamespace(
                id=order.id,
                status="pending_new",
                symbol=order.symbol,
                side=getattr(order.side, "value", order.side),
                qty=order.quantity,
                filled_qty="0",
                filled_avg_price=None,
            )
        except (APIError, TimeoutError, ConnectionError) as e:
            logger.error(
                "ORDER_API_FAILED",
                extra={
                    "cause": e.__class__.__name__,
                    "detail": str(e),
                    "op": "submit",
                    "symbol": order.symbol,
                    "qty": order.quantity,
                    "side": getattr(order.side, "value", order.side),
                    "type": getattr(order.order_type, "value", order.order_type),
                },
            )
            order.status = OrderStatus.REJECTED
            order.notes += f" | Error: {e}"
            _safe_counter_inc(
                _orders_rejected_total,
                "orders_rejected_total",
                extra={"symbol": order.symbol, "reason": "api_error"},
            )
            if self._intent_store is not None:
                try:
                    self._intent_store.record_submit_error(order.id, str(e))
                except Exception:
                    logger.debug("OMS_INTENT_SUBMIT_ERROR_RECORD_FAILED", exc_info=True)
            return None

    def cancel_order(self, order_id: str, reason: str = "User request") -> bool:
        """Cancel an active order."""
        if not order_id:
            logger.warning("CANCEL_SKIPPED", extra={"reason": "empty_order_id"})
            return False
        try:
            order = self.active_orders.get(order_id)
            if not order:
                logger.warning(f"Cannot cancel order {order_id}: not found in active orders")
                return False
            success = order.cancel(reason)
            if success:
                self.active_orders.pop(order_id, None)
                self._notify_callbacks(order, "cancelled")
            return success
        except (APIError, TimeoutError, ConnectionError) as e:
            logger.error(
                "ORDER_API_FAILED",
                extra={"cause": e.__class__.__name__, "detail": str(e), "op": "cancel", "order_id": order_id},
            )
            return False

    def get_order_status(self, order_id: str) -> dict | None:
        """Get current order status."""
        order = self.orders.get(order_id)
        if order:
            return order.to_dict()
        return None

    def get_active_orders(self) -> list[dict]:
        """Get all active orders."""
        return [order.to_dict() for order in self.active_orders.values()]

    def get_order_history(self, symbol: str = None, limit: int = 100) -> list[dict]:
        """Get order history with optional filtering."""
        orders = list(self.orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        orders.sort(key=lambda x: x.created_at, reverse=True)
        return [order.to_dict() for order in orders[:limit]]

    def add_execution_callback(self, callback: Callable):
        """Add callback for execution events."""
        self.execution_callbacks.append(callback)

    def start_monitoring(self):
        """Start order monitoring thread."""
        if self._monitor_running:
            return
        self._monitor_running = True
        self._monitor_thread = threading.Thread(target=self._monitor_orders, daemon=True)
        self._monitor_thread.start()
        logger.info("Order monitoring started")

    def stop_monitoring(self):
        """Stop order monitoring thread."""
        self._monitor_running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Order monitoring stopped")

    def _validate_order(self, order: Order) -> bool:
        """Validate order before submission."""
        try:
            if not order.symbol or order.quantity <= 0:
                logger.error(f"Invalid order parameters: symbol={order.symbol}, quantity={order.quantity}")
                return False
            tick = get_tick_size(order.symbol)
            lot = get_lot_size(order.symbol)
            original_quantity = order.quantity
            order.quantity = round_to_lot(order.quantity, lot)
            if original_quantity != order.quantity:
                logger.debug(
                    f"Quantity adjusted for {order.symbol}: {original_quantity} -> {order.quantity} (lot={lot})"
                )
            if order.order_type == OrderType.LIMIT:
                if not order.price or order.price <= 0:
                    logger.error(f"Limit order requires valid price: {order.price}")
                    return False
                if not isinstance(order.price, Money):
                    order.price = Money(order.price)
                original_price = order.price
                order.price = round_to_tick(order.price, tick)
                if float(original_price) != float(order.price):
                    logger.debug(f"Price adjusted for {order.symbol}: {original_price} -> {order.price} (tick={tick})")
            if order.side not in [OrderSide.BUY, OrderSide.SELL, OrderSide.SELL_SHORT]:
                logger.error(f"Invalid order side: {order.side}")
                return False
            return True
        except (KeyError, ValueError, TypeError, RuntimeError) as e:
            logger.error("ORDER_VALIDATION_FAILED", extra={"cause": e.__class__.__name__, "detail": str(e)})
            return False

    def _monitor_orders(self):
        """Monitor active orders for timeouts and updates."""
        while self._monitor_running:
            try:
                self._monitor_orders_tick()
                time.sleep(1)
            except (APIError, TimeoutError, ConnectionError) as e:
                logger.error("ORDER_MONITOR_FAILED", extra={"cause": e.__class__.__name__, "detail": str(e)})
                time.sleep(5)

    def _monitor_orders_tick(self) -> None:
        """Evaluate active orders once using monotonic time for expiry checks."""

        current_time = safe_utcnow()
        current_monotonic = monotonic_time()
        expired_orders: list[str] = []
        for order_id, order in list(self.active_orders.items()):
            created_monotonic = getattr(order, "_created_monotonic", None)
            age_seconds: float
            if isinstance(created_monotonic, (int, float)):
                age_seconds = current_monotonic - float(created_monotonic)
            else:
                try:
                    age_seconds = (current_time - order.created_at).total_seconds()
                except Exception:
                    age_seconds = float("inf")
            if age_seconds > self.order_timeout:
                expired_orders.append(order_id)
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED]:
                self.active_orders.pop(order_id, None)
                self._notify_callbacks(order, "completed")

        for order_id in expired_orders:
            order = self.active_orders.get(order_id)
            if not order:
                continue
            order.status = OrderStatus.EXPIRED
            order.updated_at = current_time
            self.active_orders.pop(order_id, None)
            logger.warning(f"Order {order_id} expired after {self.order_timeout} seconds")
            self._notify_callbacks(order, "expired")

        from .reconcile import reconcile_positions_and_orders

        reconcile_positions_and_orders()

    def _sync_intent_with_order_event(self, order: Order, event_type: str) -> None:
        """Persist fill and lifecycle updates for durable OMS intents."""

        if self._intent_store is None:
            return
        order_id = getattr(order, "id", None)
        if not order_id:
            return
        intent_id = self._intent_by_order_id.get(order_id)
        if intent_id is None:
            intent_id = order_id
            self._intent_by_order_id[order_id] = intent_id

        try:
            filled_qty = float(getattr(order, "filled_quantity", 0.0) or 0.0)
        except (TypeError, ValueError):
            filled_qty = 0.0
        reported_qty = self._intent_reported_fill_qty.get(order_id, 0.0)
        fill_delta = max(0.0, filled_qty - reported_qty)
        if fill_delta > 0.0:
            fill_price_raw = getattr(order, "average_fill_price", None)
            try:
                fill_price = (
                    float(fill_price_raw) if fill_price_raw is not None else None
                )
            except (TypeError, ValueError):
                fill_price = None
            self._intent_store.record_fill(
                intent_id,
                fill_qty=fill_delta,
                fill_price=fill_price,
            )
            self._intent_reported_fill_qty[order_id] = reported_qty + fill_delta

        normalized_status = ExecutionResult._normalize_status(
            getattr(order, "status", None)
        )
        status_token = (
            str(getattr(normalized_status, "value", normalized_status)).strip().upper()
            if normalized_status is not None
            else ""
        )
        if not status_token:
            status_token = str(event_type).strip().upper()
        terminal_events = {"COMPLETED", "CANCELLED", "CANCELED", "EXPIRED"}
        is_terminal = (
            bool(getattr(normalized_status, "is_terminal", False))
            or status_token in {"FILLED", "REJECTED", "CANCELED", "CANCELLED", "EXPIRED"}
            or str(event_type).strip().upper() in terminal_events
        )
        if not is_terminal:
            return

        final_status = status_token or "CLOSED"
        self._intent_store.close_intent(intent_id, final_status=final_status)
        self._intent_by_order_id.pop(order_id, None)
        self._intent_reported_fill_qty.pop(order_id, None)

    def _notify_callbacks(self, order: Order, event_type: str):
        """Notify registered callbacks of order events."""
        try:
            try:
                self._sync_intent_with_order_event(order, event_type)
            except Exception:
                logger.debug("OMS_INTENT_SYNC_FAILED", exc_info=True)
            for callback in self.execution_callbacks:
                try:
                    callback(order, event_type)
                except (KeyError, ValueError, TypeError, RuntimeError) as e:
                    logger.error(
                        "CALLBACK_FAILED", extra={"cause": e.__class__.__name__, "detail": str(e), "order_id": order.id}
                    )
        except (KeyError, ValueError, TypeError, RuntimeError) as e:
            logger.error(
                "CALLBACK_NOTIFICATION_FAILED",
                extra={"cause": e.__class__.__name__, "detail": str(e), "order_id": order.id},
            )


class ExecutionEngine:
    """
    Main execution engine for institutional order processing.

    Coordinates order management, execution algorithms,
    and provides unified execution interface.
    """

    _minute_stats: dict[str, float] = {}
    _latest_quote: dict[str, float] = {}

    def __init__(self, ctx=None, market_data_feed=None, broker_interface=None):
        """Initialize execution engine."""
        self.ctx = ctx
        self.order_manager = OrderManager()
        self.market_data_feed = market_data_feed
        self.broker_interface = broker_interface
        self.logger = logger
        self._open_orders: dict[str, OrderInfo] = {}
        self._available_qty: float = 0
        self._order_signal_meta: dict[str, _SignalMeta] = {}
        # In-memory fallback for position tracking when broker data is unavailable
        self._position_ledger: dict[str, int] = {}
        self.execution_stats = {
            "total_orders": 0,
            "filled_orders": 0,
            "cancelled_orders": 0,
            "rejected_orders": 0,
            "total_volume": 0.0,
            "average_fill_time": 0.0,
        }
        self._last_partial_fill_summary: dict[str, Any] | None = None
        self._broker_sync: BrokerSyncResult | None = None
        self._open_order_qty_index: dict[str, tuple[float, float]] = {}
        self._policy_selector_enabled = _env_bool(
            "AI_TRADING_EXEC_POLICY_SELECTOR_ENABLED",
            False,
        )
        self._cost_model_enabled = _env_bool(
            "AI_TRADING_EXEC_COST_MODEL_ENABLED",
            False,
        )
        self._cost_model_apply_limits = _env_bool(
            "AI_TRADING_EXEC_COST_MODEL_APPLY_LIMITS",
            False,
        )
        self._cost_model_path = str(
            get_env(
                "AI_TRADING_EXEC_COST_MODEL_PATH",
                "runtime/execution_cost_model.json",
            )
        )
        self._execution_cost_model: Any | None = None
        self._load_execution_cost_model()
        emit_once(logger, "EXECUTION_ENGINE_INIT", "info", "ExecutionEngine initialized")
        try:
            self.order_manager.add_execution_callback(self._handle_execution_event)
        except Exception:  # pragma: no cover - defensive, callbacks optional in some tests
            logger.debug("EXECUTION_CALLBACK_REGISTRATION_FAILED", exc_info=True)

    def _load_execution_cost_model(self) -> None:
        """Load persisted execution cost model when enabled."""

        if not self._cost_model_enabled:
            return
        try:
            from ai_trading.execution.cost_model import CostModel

            self._execution_cost_model = CostModel.load(self._cost_model_path)
            self.logger.info(
                "EXEC_COST_MODEL_LOADED",
                extra={"path": self._cost_model_path},
            )
        except Exception:
            self.logger.debug("EXEC_COST_MODEL_LOAD_FAILED", exc_info=True)
            self._execution_cost_model = None

    def _estimate_cost_floor_bps(
        self,
        *,
        symbol: str,
        spread_bps: float | None,
        volatility_pct: float | None,
        participation_rate: float | None,
    ) -> float | None:
        """Return bounded estimated cost in bps from model + realized slippage EWMA."""

        model = self._execution_cost_model
        if model is None:
            return None
        tca_hint: float | None = None
        try:
            from ai_trading.execution.slippage_log import get_ewma_cost_bps

            tca_hint = get_ewma_cost_bps(symbol, default=2.0)
        except Exception:
            tca_hint = None
        try:
            estimate = model.estimate_cost_bps(
                spread_bps=spread_bps,
                volatility_pct=volatility_pct,
                participation_rate=participation_rate,
                tca_cost_bps=tca_hint,
            )
        except Exception:
            self.logger.debug("EXEC_COST_ESTIMATE_FAILED", exc_info=True)
            return None
        if not math.isfinite(estimate) or estimate <= 0:
            return None
        return float(estimate)

    def _select_execution_policy(
        self,
        *,
        symbol: str,
        side: OrderSide,
        quantity: int,
        kwargs: dict[str, Any],
        limit_price: float | None,
    ) -> dict[str, Any]:
        """Select policy and derive optional routing hints for execute_order."""

        if not self._policy_selector_enabled:
            return {"selected": False}
        try:
            from ai_trading.execution.policy_selector import (
                ExecutionPolicy,
                select_execution_policy,
            )
        except Exception:
            self.logger.debug("EXEC_POLICY_SELECTOR_IMPORT_FAILED", exc_info=True)
            return {"selected": False}

        spread_bps = kwargs.get("spread_bps")
        if spread_bps is None:
            bid = kwargs.get("bid")
            ask = kwargs.get("ask")
            try:
                bid_f = float(bid) if bid is not None else None
                ask_f = float(ask) if ask is not None else None
            except (TypeError, ValueError):
                bid_f = None
                ask_f = None
            if bid_f and ask_f and bid_f > 0 and ask_f >= bid_f:
                spread_bps = ((ask_f - bid_f) / bid_f) * 10000.0

        reference_price = limit_price
        if reference_price is None:
            raw_price = kwargs.get("price")
            try:
                reference_price = float(raw_price) if raw_price is not None else None
            except (TypeError, ValueError):
                reference_price = None
        if reference_price is None:
            expected = kwargs.get("expected_price")
            try:
                reference_price = float(expected) if expected is not None else None
            except (TypeError, ValueError):
                reference_price = None
        if reference_price is None or reference_price <= 0:
            reference_price = 1.0

        avg_daily_volume = kwargs.get("avg_daily_volume")
        if avg_daily_volume is None:
            avg_daily_volume = kwargs.get("volume_1d")
        try:
            adv_shares = float(avg_daily_volume) if avg_daily_volume is not None else 0.0
        except (TypeError, ValueError):
            adv_shares = 0.0
        adv_notional = max(0.0, adv_shares * reference_price)

        volatility_pct = kwargs.get("volatility_pct")
        if volatility_pct is None:
            volatility_pct = kwargs.get("volatility")

        urgency = kwargs.get("urgency_level")
        if urgency is None:
            urgency = kwargs.get("urgency")

        provenance = str(
            get_env(
                "DATA_PROVENANCE",
                get_env("ALPACA_DATA_FEED", "iex"),
            )
        )
        order_notional = abs(float(quantity) * float(reference_price))
        decision = select_execution_policy(
            spread_bps=spread_bps,
            volatility_pct=volatility_pct,
            order_notional=order_notional,
            avg_daily_volume_notional=adv_notional,
            urgency=urgency,
            data_provenance=provenance,
            allow_twap=_env_bool("AI_TRADING_TWAP_ENABLED", True),
        )

        routing: dict[str, Any] = {
            "selected": True,
            "policy": decision.policy.value,
            "reasons": decision.reasons,
            "participation_rate": decision.participation_rate,
            "urgency_score": decision.urgency_score,
            "spread_bps": spread_bps,
            "volatility_pct": volatility_pct,
            "data_provenance": provenance,
            "symbol": symbol,
            "side": getattr(side, "value", side),
            "quantity": quantity,
        }
        if decision.policy == ExecutionPolicy.TWAP:
            routing["use_twap"] = True
            routing["execution_algorithm"] = ExecutionAlgorithm.TWAP
        elif decision.policy == ExecutionPolicy.POV:
            routing["execution_algorithm"] = ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL
        elif decision.policy == ExecutionPolicy.PASSIVE_LIMIT:
            routing["execution_algorithm"] = ExecutionAlgorithm.LIMIT
        else:
            routing["execution_algorithm"] = ExecutionAlgorithm.MARKET
        return routing

    def _select_api(self):
        """Return the active broker API interface."""
        ctx_api = getattr(getattr(self, "ctx", None), "api", None)
        return ctx_api or self.broker_interface

    @property
    def available_qty(self) -> float:
        """Get the currently available quantity for trading."""
        return self._available_qty

    @available_qty.setter
    def available_qty(self, qty: float) -> None:
        """Set the available quantity for trading."""
        self._available_qty = qty

    def set_available_qty(self, qty: float) -> None:
        """Helper to set :attr:`available_qty`."""
        self.available_qty = qty

    def get_available_qty(self) -> float:
        """Helper to get :attr:`available_qty`."""
        return self.available_qty

    @property
    def position_ledger(self) -> dict[str, int]:
        """Return a copy of the tracked positions."""
        return self._position_ledger.copy()

    @property
    def last_partial_fill_summary(self) -> dict[str, Any] | None:
        """Return the most recent partial fill summary recorded by the engine."""
        return self._last_partial_fill_summary

    def _update_position(self, symbol: str, side: OrderSide, quantity: int) -> None:
        """Update in-memory position ledger for a fill event."""
        delta = quantity if side == OrderSide.BUY else -quantity
        new_qty = self._position_ledger.get(symbol, 0) + delta
        if new_qty:
            self._position_ledger[symbol] = new_qty
        else:
            self._position_ledger.pop(symbol, None)

    # Lifecycle hooks to align with bot_engine expectations -----------------
    def start_cycle(self) -> None:
        """Hook called at the start of a trading cycle (no-op)."""
        try:
            # Best-effort cleanup of very old tracked orders
            self.cleanup_stale_orders()
        except Exception as exc:
            # Never allow lifecycle hooks to raise
            logger.debug("START_CYCLE_CLEANUP_FAILED", exc_info=exc)

    def end_cycle(self) -> None:  # optional hook
        return None

    def _track_order(self, order: Order) -> None:
        """Track an order in the shared monitoring structure."""
        _cleanup_stale_orders()
        info = OrderInfo(
            order_id=order.id,
            symbol=order.symbol,
            side=getattr(order.side, "value", order.side),
            qty=getattr(order, "quantity", getattr(order, "qty", 0)),
            submitted_time=time.time(),
            last_status=getattr(order.status, "value", getattr(order, "status", "new")),
        )
        with _order_tracking_lock:
            _active_orders[order.id] = info

    def track_order(self, order: Order) -> None:
        """Public wrapper for :meth:`_track_order`."""
        self._track_order(order)

    def _update_order_status(self, order_id: str, status: str) -> None:
        """Update tracked order status and remove if terminal."""
        terminal = {"filled", "canceled", "cancelled", "rejected"}
        with _order_tracking_lock:
            info = _active_orders.get(order_id)
            if info:
                info.last_status = status
                if status.lower() in terminal:
                    _active_orders.pop(order_id, None)

    def get_pending_orders(self) -> list[OrderInfo]:
        """Return list of currently tracked orders."""
        with _order_tracking_lock:
            return list(_active_orders.values())

    def _update_broker_snapshot(
        self,
        open_orders: Iterable[Any] | None,
        positions: Iterable[Any] | None,
    ) -> BrokerSyncResult:
        """Store normalized broker state for downstream consumers."""

        open_orders_tuple = tuple(open_orders or ())
        positions_tuple = tuple(positions or ())
        buy_index: dict[str, float] = {}
        sell_index: dict[str, float] = {}

        def _normalize_symbol(value: Any) -> str | None:
            if value in (None, ""):
                return None
            try:
                text = str(value).strip()
            except Exception:  # pragma: no cover - defensive
                self.logger.debug("BROKER_SNAPSHOT_SYMBOL_NORMALIZE_FAILED", exc_info=True)
                return None
            if not text:
                return None
            return text.upper()

        def _extract_side(value: Any) -> str | None:
            if value in (None, ""):
                return None
            try:
                text = str(value).strip().lower()
            except Exception:  # pragma: no cover - defensive
                self.logger.debug("BROKER_SNAPSHOT_SIDE_NORMALIZE_FAILED", exc_info=True)
                return None
            if text in {"buy", "long", "cover"}:
                return "buy"
            if text in {"sell", "sell_short", "sellshort", "short"}:
                return "sell"
            return None

        def _extract_qty(value: Any) -> float:
            candidates = []
            if isinstance(value, Mapping):
                candidates.extend(
                    value.get(key)
                    for key in ("qty", "quantity", "remaining_qty", "unfilled_qty", "filled_qty")
                )
            else:
                for key in ("qty", "quantity", "remaining_qty", "unfilled_qty", "filled_qty"):
                    if hasattr(value, key):
                        candidates.append(getattr(value, key))
            for candidate in candidates:
                if candidate in (None, ""):
                    continue
                try:
                    qty = float(candidate)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(qty):
                    continue
                return abs(qty)
            return 0.0

        for order in open_orders_tuple:
            if isinstance(order, Mapping):
                symbol = _normalize_symbol(order.get("symbol"))
                side = _extract_side(order.get("side"))
            else:
                symbol = _normalize_symbol(getattr(order, "symbol", None))
                side = _extract_side(getattr(order, "side", None))
            if symbol is None or side is None:
                continue
            qty_val = _extract_qty(order)
            if qty_val <= 0:
                continue
            if side == "buy":
                buy_index[symbol] = buy_index.get(symbol, 0) + qty_val
            else:
                sell_index[symbol] = sell_index.get(symbol, 0) + qty_val

        qty_index: dict[str, tuple[float, float]] = {}
        for sym in set(buy_index) | set(sell_index):
            qty_index[sym] = (buy_index.get(sym, 0.0), sell_index.get(sym, 0.0))

        snapshot = BrokerSyncResult(
            open_orders=open_orders_tuple,
            positions=positions_tuple,
            open_buy_by_symbol=buy_index,
            open_sell_by_symbol=sell_index,
            timestamp=monotonic_time(),
        )
        self._broker_sync = snapshot
        self._open_order_qty_index = qty_index
        return snapshot

    def synchronize_broker_state(self) -> BrokerSyncResult:
        """Return the latest broker snapshot (no-op for base engine)."""

        if self._broker_sync is None:
            self._broker_sync = BrokerSyncResult((), (), {}, {}, monotonic_time())
        return self._broker_sync

    def open_order_totals(self, symbol: str) -> tuple[float, float]:
        """Return aggregate (buy_qty, sell_qty) for *symbol* from last sync."""

        if not symbol:
            return (0.0, 0.0)
        key = symbol.upper()
        return self._open_order_qty_index.get(key, (0.0, 0.0))

    def _cancel_stale_order(self, order_id: str) -> bool:
        """Attempt to cancel a stale order via broker interface."""
        if self.broker_interface is None:
            return False
        try:
            ord_obj = self.broker_interface.get_order(order_id)
            if getattr(ord_obj, "status", "").lower() == "new":
                self.broker_interface.cancel_order(order_id)
            return True
        except Exception as exc:  # pragma: no cover - broker interface may vary
            logger.debug("Failed to cancel stale order %s: %s", order_id, exc)
            return False

    def _assess_liquidity(self, symbol: str, quantity: int) -> tuple[int, bool]:
        """Assess liquidity and optionally adjust quantity."""
        bid, ask = (0.0, 0.0)
        try:
            bid, ask = self._latest_quote()
        except (RuntimeError, ValueError):
            return (quantity, False)
        spread_pct = (ask - bid) / bid if bid else 0.0
        if spread_pct >= 0.01:
            return (int(quantity * 0.75), False)
        return (quantity, False)

    def cleanup_stale_orders(self, now: float | None = None, max_age_seconds: int | None = None) -> int:
        """Remove stale orders and attempt cancelation via broker."""
        now_s = now if now is not None else time.time()
        max_age = max_age_seconds if max_age_seconds is not None else ORDER_STALE_AFTER_S
        with _order_tracking_lock:
            stale_ids = [oid for oid, info in _active_orders.items() if now_s - info.submitted_time >= max_age]
        for oid in stale_ids:
            self._cancel_stale_order(oid)
        return _cleanup_stale_orders(now_s, max_age)

    def check_stops(self) -> None:
        """
        Safety hook invoked after each cycle. It should never raise.
        For now: best-effort inspection of open positions; no-op if unsupported.
        """
        try:
            broker = getattr(self, "broker", None) or getattr(self, "broker_interface", None)
            if broker is not None and hasattr(broker, "list_positions"):
                positions = broker.list_positions() or []
                logger.debug("check_stops: inspected %d positions", len(positions))
            else:
                positions = [SimpleNamespace(symbol=s, qty=q) for s, q in self._position_ledger.items()]
                logger.debug("check_stops: inspected %d positions (ledger)", len(positions))
        except (ValueError, TypeError) as e:
            logger.info("check_stops: suppressed exception: %s", e)

    def check_trailing_stops(self) -> None:  # optional hook
        return None

    def _validate_short_selling(self, _api, symbol: str, qty: float, price: float | None = None) -> bool:
        """Run short selling validation and return True on success."""

        from ai_trading.risk.short_selling import validate_short_selling

        return validate_short_selling(symbol, qty, price)

    def _reconcile_partial_fills(
        self, *args, requested_qty=None, remaining_qty=None, symbol=None, side=None, **_kwargs
    ) -> None:
        """Detect partial fills and emit guardrail alerts.

        Test contract:
        - Log 'PARTIAL_FILL_DETECTED' at WARNING when a partial fill occurs.
        - For fill rate around 30%, emit a MODERATE_FILL_RATE_ALERT at WARNING; 50% should not trigger error alerts.
        """
        try:
            # Accept alias used in tests: submitted_qty -> requested_qty
            if requested_qty is None:
                requested_qty = _kwargs.get("submitted_qty")
            if requested_qty is None or remaining_qty is None:
                return
            rq = float(requested_qty)
            rem = float(remaining_qty)
            if rq <= 0:
                return
            # Reset summary before processing a new fill snapshot
            self._last_partial_fill_summary = None
            # Prefer last_order.filled_qty when available
            order = _kwargs.get("last_order")
            if order is not None and hasattr(order, "filled_qty") and order.filled_qty is not None:
                try:
                    filled = float(order.filled_qty)
                except (ValueError, TypeError):
                    filled = max(0.0, rq - rem)
            else:
                filled = max(0.0, rq - rem)
            calc_filled = max(0.0, rq - rem)
            # Detect quantity mismatch when broker's last_order disagrees with calculation
            ord_filled: float | None
            if order is not None:
                try:
                    ord_filled = float(order.filled_qty) if getattr(order, "filled_qty", None) is not None else None
                except (ValueError, TypeError):
                    ord_filled = None
            else:
                ord_filled = None
            mismatch = ord_filled is not None and abs(float(ord_filled) - calc_filled) > 0.5
            if mismatch:
                try:
                    self.logger.warning(
                        f"QUANTITY_MISMATCH_DETECTED: calculated_filled_qty={int(calc_filled)} reported_filled_qty={int(ord_filled or 0)}",
                        extra={
                            "symbol": symbol,
                            "side": side,
                            "reported_filled_qty": int(ord_filled or 0),
                            "calculated_filled_qty": int(calc_filled),
                            "requested_qty": int(rq),
                        },
                    )
                except Exception as exc:
                    self.logger.debug("QUANTITY_MISMATCH_LOG_FAILED", exc_info=exc)

            # Choose the more reliable filled metric: prefer broker value only when it matches calculation
            filled_for_eval = calc_filled if mismatch or ord_filled is None else ord_filled

            if filled_for_eval < rq:
                fill_rate = (filled_for_eval / rq) if rq else 0.0
                summary_payload = {
                    "symbol": symbol,
                    "side": side,
                    "filled_qty": int(filled_for_eval),
                    "requested_qty": int(rq),
                    "fill_rate_pct": round(fill_rate * 100.0, 2),
                }
                if ord_filled is not None:
                    summary_payload["reported_filled_qty"] = float(ord_filled)
                if mismatch:
                    summary_payload["mismatch_detected"] = True
                self._last_partial_fill_summary = summary_payload
                # Structured log for tests to parse via `extra`, keep message token stable
                self.logger.warning(
                    "PARTIAL_FILL_DETECTED",
                    extra={
                        **summary_payload,
                    },
                )
                # Human-readable detail to satisfy substring checks in some tests
                try:
                    self.logger.info(f"PARTIAL_FILL_DETAILS requested={int(rq)} filled={int(filled_for_eval)}")
                except Exception as exc:
                    self.logger.debug("PARTIAL_FILL_DETAIL_LOG_FAILED", exc_info=exc)
                # Thresholds per tests:
                # - LOW at <= 20%
                # - MODERATE at (20%, 35%]
                if fill_rate <= 0.20:
                    self.logger.error("LOW_FILL_RATE_ALERT")
                elif fill_rate <= 0.35:
                    self.logger.warning(f"MODERATE_FILL_RATE_ALERT: {fill_rate:.2%}")
            else:
                # Full fill
                # Only claim full-fill when sources agree or no mismatch detected
                if not mismatch:
                    self._last_partial_fill_summary = None
                    self.logger.info(
                        "FULL_FILL_SUCCESS",
                        extra={
                            "symbol": symbol,
                            "side": side,
                            "filled_qty": int(filled_for_eval),
                            "requested_qty": int(rq),
                        },
                    )
        except (ValueError, TypeError) as exc:
            self.logger.debug("PARTIAL_FILL_EVAL_FAILED", exc_info=exc)

    def execute_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        *,
        asset_class: str | None = None,
        **kwargs: Any,
    ):
        """Execute a trading order.

        Args:
            symbol: Trading symbol.
            side: Order side, ``OrderSide.BUY``, ``OrderSide.SELL`` or ``OrderSide.SELL_SHORT``.
            quantity: Quantity to trade.
            order_type: Type of order.
            **kwargs: Additional order parameters.

        Returns:
            Order ID if successful, ``None`` if rejected.
        """
        try:
            kwargs = dict(kwargs)
            signal = kwargs.pop("signal", None)
            explicit_signal_weight = kwargs.pop("signal_weight", None)
            raw_keys = set(kwargs)
            ignored_keys = set()
            if raw_keys:
                ignored_keys = {k for k in raw_keys if k not in KNOWN_EXECUTE_ORDER_KWARGS}
                for key in list(ignored_keys):
                    kwargs.pop(key, None)
            api = self._select_api()
            if api is None:
                logger.debug("NO_API_SELECTED")
            if isinstance(side, str):
                side = OrderSide(side)

            available_qty_raw = self.get_available_qty()
            if callable(available_qty_raw):  # accommodate mocked attributes
                available_qty_raw = available_qty_raw()
            try:
                available_qty = float(available_qty_raw)
            except (TypeError, ValueError):
                available_qty = 0.0
            if side == OrderSide.SELL:
                if available_qty <= 0:
                    self.logger.info("SKIP_NO_POSITION | no shares to sell, skipping")
                    self.execution_stats["rejected_orders"] += 1
                    return None
            elif side == OrderSide.SELL_SHORT:
                self.logger.info("SHORT_SELL_INITIATED | symbol=%s qty=%d", symbol, quantity)
                if not self._validate_short_selling(api, symbol, quantity):
                    self.execution_stats["rejected_orders"] += 1
                    return None
            quantity = int(_ensure_positive_qty(quantity))
            raw_price = kwargs.get("price")
            price_alias = _ensure_valid_price(raw_price)
            if price_alias is None:
                kwargs.pop("price", None)
            else:
                kwargs["price"] = price_alias
            limit_price = _ensure_valid_price(kwargs.get("limit_price"))
            if limit_price is None and price_alias is not None:
                limit_price = price_alias
            stop_price = _ensure_valid_price(kwargs.get("stop_price"))
            kwargs["limit_price"] = limit_price
            kwargs["stop_price"] = stop_price
            tif_alias = kwargs.pop("tif", None)
            if tif_alias is not None and not kwargs.get("time_in_force"):
                kwargs["time_in_force"] = tif_alias

            policy_routing = self._select_execution_policy(
                symbol=symbol,
                side=side,
                quantity=quantity,
                kwargs=kwargs,
                limit_price=limit_price,
            )
            if policy_routing.get("selected"):
                self.logger.info("EXEC_POLICY_SELECTED", extra=policy_routing)
                if kwargs.get("execution_algorithm") is None:
                    kwargs["execution_algorithm"] = policy_routing.get("execution_algorithm")
                if "use_twap" in policy_routing and "use_twap" not in kwargs:
                    kwargs["use_twap"] = bool(policy_routing["use_twap"])

                cost_floor_bps = self._estimate_cost_floor_bps(
                    symbol=symbol,
                    spread_bps=policy_routing.get("spread_bps"),
                    volatility_pct=policy_routing.get("volatility_pct"),
                    participation_rate=policy_routing.get("participation_rate"),
                )
                if cost_floor_bps is not None:
                    self.logger.info(
                        "EXEC_COST_FLOOR_ESTIMATE",
                        extra={"symbol": symbol, "cost_floor_bps": round(cost_floor_bps, 4)},
                    )
                    if self._cost_model_apply_limits:
                        current_cap = kwargs.get("max_slippage_bps")
                        try:
                            cap_value = float(current_cap) if current_cap is not None else float(
                                EXECUTION_PARAMETERS.get("MAX_SLIPPAGE_BPS", 50)
                            )
                        except Exception:
                            cap_value = float(EXECUTION_PARAMETERS.get("MAX_SLIPPAGE_BPS", 50))
                        derived_cap = max(cap_value, cost_floor_bps * 2.0)
                        kwargs["max_slippage_bps"] = derived_cap
                        self.logger.info(
                            "EXEC_COST_MODEL_APPLIED",
                            extra={"symbol": symbol, "max_slippage_bps": round(derived_cap, 4)},
                        )
            tif_value = kwargs.get("time_in_force") or "day"
            payload: dict[str, Any] = {
                "symbol": symbol,
                "side": getattr(side, "value", side),
                "qty": quantity,
                "type": getattr(order_type, "value", order_type),
                "time_in_force": tif_value,
                "limit_price": limit_price,
                "stop_price": stop_price,
            }
            extended_hours = kwargs.get("extended_hours")
            if extended_hours is not None:
                payload["extended_hours"] = extended_hours
            kwargs.pop("asset_class", None)
            supported_asset_class = False
            if asset_class:
                supported_asset_class = self._supports_asset_class()
                if supported_asset_class:
                    payload["asset_class"] = asset_class
                else:
                    ignored_keys = set(ignored_keys)
                    ignored_keys.add("asset_class")
            payload_extra = {
                key: payload.get(key)
                for key in (
                    "symbol",
                    "side",
                    "qty",
                    "type",
                    "time_in_force",
                    "limit_price",
                    "stop_price",
                    "extended_hours",
                    "asset_class",
                )
            }
            payload_extra["ignored_keys"] = tuple(sorted(ignored_keys)) if ignored_keys else ()
            logger.debug("ORDER_SUBMIT_PAYLOAD", extra=payload_extra)
            if ignored_keys:
                for key in sorted(ignored_keys):
                    logger.debug("EXEC_IGNORED_KWARG", extra={"kw": key})
                logger.debug(
                    "EXECUTE_ORDER_IGNORED_KWARGS",
                    extra={"ignored_keys": tuple(sorted(ignored_keys))},
                )
            testing_mode = _env_bool("TESTING", False)
            if order_type == OrderType.MARKET:
                if not testing_mode:
                    try:
                        import importlib

                        be = importlib.import_module("ai_trading.core.bot_engine")
                        quote_info: SimpleNamespace | None = None
                        if hasattr(be, "resolve_trade_quote"):
                            quote_info = be.resolve_trade_quote(symbol)
                        elif hasattr(be, "get_latest_price"):
                            price = be.get_latest_price(symbol)
                            source = (
                                be.get_price_source(symbol)
                                if hasattr(be, "get_price_source")
                                else _PRIMARY_FALLBACK_SOURCE
                            )
                            quote_info = SimpleNamespace(price=price, source=source)
                        if quote_info is not None and quote_info.price is not None:
                            kwargs["expected_price"] = quote_info.price
                            kwargs.setdefault("price_source", getattr(quote_info, "source", "unknown"))
                            kwargs["expected_price_source"] = getattr(quote_info, "source", "unknown")
                            logger.debug(
                                "EXPECTED_PRICE_FETCHED",
                                extra={
                                    "symbol": symbol,
                                    "expected_price": float(quote_info.price),
                                    "price_source": getattr(quote_info, "source", "unknown"),
                                },
                            )
                    except Exception as e:  # pragma: no cover - diagnostics only
                        logger.debug(
                            "EXPECTED_PRICE_FETCH_FAILED",
                            extra={"symbol": symbol, "cause": e.__class__.__name__},
                        )
            if (
                order_type == OrderType.MARKET
                and testing_mode
            ):
                max_slippage_cfg = EXECUTION_PARAMETERS.get("MAX_SLIPPAGE_BPS", 0)
                try:
                    max_slippage_bps = float(max_slippage_cfg)
                except (TypeError, ValueError):
                    max_slippage_bps = 0.0
                if max_slippage_bps > 0 and price_alias is not None:
                    limit_price = price_alias
                    kwargs["limit_price"] = limit_price
                    kwargs["price"] = limit_price
                    order_type = OrderType.LIMIT
                    payload["type"] = getattr(order_type, "value", order_type)
                    payload["limit_price"] = limit_price
                    payload_extra["type"] = payload["type"]
                    payload_extra["limit_price"] = limit_price
                    conversion_extra = {
                        "symbol": symbol,
                        "side": getattr(side, "value", str(side)),
                        "limit_price": limit_price,
                        "max_slippage_bps": round(max_slippage_bps, 6),
                        "requested_qty": quantity,
                    }
                    self.logger.warning("SLIPPAGE_LIMIT_CONVERSION", extra=conversion_extra)
                    original_qty = quantity
                    scale = max(0.0, 1.0 - (max_slippage_bps / 10000.0))
                    adjusted_qty = int(max(1, math.floor(original_qty * scale)))
                    if adjusted_qty < original_qty:
                        quantity = adjusted_qty
                        payload["qty"] = quantity
                        self.logger.warning(
                            "SLIPPAGE_QTY_REDUCED",
                            extra={
                                "symbol": symbol,
                                "side": getattr(side, "value", str(side)),
                                "original_qty": original_qty,
                                "adjusted_qty": quantity,
                                "slippage_scale": round(scale, 6),
                            },
                        )
                    payload_extra["qty"] = quantity
            order = Order(symbol, side, quantity, order_type, **kwargs)

            # Optional TWAP routing for large orders (config/kwargs gated)
            use_twap = bool(kwargs.get("use_twap", False))
            if not use_twap and getattr(order, "execution_algorithm", None) == ExecutionAlgorithm.TWAP:
                use_twap = True
            auto_twap_enabled = _env_bool(
                "AI_TRADING_EXEC_AUTO_TWAP_ENABLED",
                False,
            )
            if not use_twap and auto_twap_enabled:
                try:
                    from ai_trading.core.constants import EXECUTION_PARAMETERS as _EXECUTION_PARAMETERS

                    twap_min_qty = int(_EXECUTION_PARAMETERS.get("TWAP_MIN_QTY", 5000))
                    use_twap = quantity >= twap_min_qty
                except Exception:
                    use_twap = False
            if use_twap and quantity > 0:
                try:
                    from ai_trading.execution.algorithms import TWAPExecutor

                    duration_min = int(kwargs.get("twap_duration_min", 15))
                    twap = TWAPExecutor(self.order_manager)
                    child_ids = twap.execute_twap_order(
                        symbol,
                        side,
                        quantity,
                        duration_minutes=duration_min,
                        parent_order_id=getattr(order, "id", None),
                        strategy_id=order.strategy_id,
                    )
                    self.logger.info(
                        "TWAP_SUBMITTED",
                        extra={"symbol": symbol, "qty": quantity, "slices": len(child_ids)},
                    )
                    # Track parent for monitoring/telemetry
                    self._track_order(order)
                    return ExecutionResult(order, order.status, 0, quantity, self._coerce_signal_weight(explicit_signal_weight, signal))
                except Exception:
                    self.logger.debug("TWAP_FALLBACK_TO_DIRECT", exc_info=True)
            if self.order_manager.submit_order(order):
                self.execution_stats["total_orders"] += 1
                meta_weight = self._coerce_signal_weight(explicit_signal_weight, signal)
                if signal is not None or meta_weight is not None:
                    self._order_signal_meta[order.id] = _SignalMeta(signal, quantity, meta_weight)
                else:
                    self._order_signal_meta.pop(order.id, None)
                if order_type == OrderType.MARKET:
                    self._simulate_market_execution(order)
                weight_for_result = meta_weight
                meta = self._order_signal_meta.get(order.id)
                if meta is not None:
                    weight_for_result = meta.signal_weight
                result = ExecutionResult(
                    order,
                    getattr(order, "status", None),
                    getattr(order, "filled_quantity", 0),
                    quantity,
                    weight_for_result,
                )
                return result
            self.execution_stats["rejected_orders"] += 1
            return None
        except (ValueError, TypeError, KeyError) as e:
            logger.error(
                "EXECUTE_FAILED",
                extra={"cause": e.__class__.__name__, "detail": str(e)},
            )
            self.execution_stats["rejected_orders"] += 1
            raise

    def execute_sliced(self, slices: Iterable[Any] | None, **kwargs: Any) -> list[Any]:
        """Execute an order broken into ``slices`` of varying sizes.

        ``slices`` may be an iterable of ints/floats representing quantities or
        ratios, or mappings containing ``quantity``/``qty``/``ratio`` keys along
        with additional kwargs to apply to each slice.
        """

        if isinstance(slices, (str, bytes)):
            raise TypeError("slices must be an iterable of slice definitions")

        kwargs = dict(kwargs)
        symbol = kwargs.pop("symbol", None)
        if symbol is None:
            raise TypeError("execute_sliced requires 'symbol'")
        side_value = kwargs.pop("side", None)
        side = _normalize_order_side(side_value)
        if side is None:
            raise ValueError("execute_sliced requires a valid 'side'")
        raw_quantity = (
            kwargs.pop("quantity", None)
            or kwargs.pop("total_quantity", None)
            or kwargs.pop("qty", None)
        )
        if raw_quantity is None:
            raise TypeError("execute_sliced requires 'quantity'")
        try:
            total_quantity = int(_ensure_positive_qty(raw_quantity))
        except ValueError:
            try:
                if float(raw_quantity) == 0:
                    return []
            except (TypeError, ValueError):
                raise
            return []

        order_type_value = kwargs.pop("order_type", OrderType.MARKET)
        if isinstance(order_type_value, OrderType):
            order_type = order_type_value
        else:
            try:
                order_type = OrderType(str(order_type_value).lower())
            except Exception:
                order_type = OrderType.MARKET
        asset_class = kwargs.pop("asset_class", None)

        if slices is None:
            slice_defs: list[Any] = []
        else:
            try:
                slice_defs = list(slices)
            except TypeError:
                slice_defs = [slices]

        def _as_float(value: Any) -> float | None:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(number):
                return None
            return number

        def _quantity_from_value(value: Any, *, treat_fraction_as_ratio: bool = True) -> int | None:
            if value is None:
                return None
            if isinstance(value, bool):
                return 1 if value else 0
            number = _as_float(value)
            if number is None:
                return None
            if treat_fraction_as_ratio and 0 < number < 1 and total_quantity > 1:
                number *= total_quantity
            quantity_int = int(round(number))
            if quantity_int < 0:
                return 0
            return quantity_int

        quantity_keys = ("quantity", "qty", "shares", "size")
        ratio_keys = ("ratio", "weight", "fraction")
        percent_keys = ("percent", "percentage", "pct")

        slice_specs: list[dict[str, Any]] = []
        for entry in slice_defs:
            per_slice_kwargs: dict[str, Any] = {}
            quantity_hint: Any | None = None
            ratio_hint: Any | None = None
            percent_hint: Any | None = None

            if isinstance(entry, Mapping):
                per_slice_kwargs = {k: v for k, v in entry.items()}
                for key in quantity_keys:
                    if key in per_slice_kwargs:
                        quantity_hint = per_slice_kwargs.pop(key)
                        break
                if quantity_hint is None:
                    for key in ratio_keys:
                        if key in per_slice_kwargs:
                            ratio_hint = per_slice_kwargs.pop(key)
                            break
                if quantity_hint is None and ratio_hint is None:
                    for key in percent_keys:
                        if key in per_slice_kwargs:
                            percent_hint = per_slice_kwargs.pop(key)
                            break
            elif isinstance(entry, tuple) and entry:
                quantity_hint = entry[0]
                if len(entry) > 1 and isinstance(entry[1], Mapping):
                    per_slice_kwargs = {k: v for k, v in entry[1].items()}
            else:
                quantity_hint = entry

            quantity = _quantity_from_value(quantity_hint)
            if quantity is None and ratio_hint is not None:
                ratio_value = _as_float(ratio_hint)
                if ratio_value is not None:
                    quantity = _quantity_from_value(
                        ratio_value * total_quantity,
                        treat_fraction_as_ratio=False,
                    )
            if quantity is None and percent_hint is not None:
                percent_value = _as_float(percent_hint)
                if percent_value is not None:
                    quantity = _quantity_from_value(
                        (percent_value / 100.0) * total_quantity,
                        treat_fraction_as_ratio=False,
                    )
            if quantity is None:
                quantity = 0

            slice_specs.append({"quantity": int(quantity), "kwargs": per_slice_kwargs})

        if not slice_specs:
            slice_specs.append({"quantity": total_quantity, "kwargs": {}})

        for spec in slice_specs:
            qty = int(spec.get("quantity", 0) or 0)
            if qty < 0:
                qty = 0
            spec["quantity"] = qty

        if total_quantity <= 0:
            return []

        current_sum = sum(spec["quantity"] for spec in slice_specs)
        if current_sum == 0 and total_quantity > 0:
            slice_specs[0]["quantity"] = total_quantity
            current_sum = total_quantity

        difference = total_quantity - current_sum
        if difference != 0 and slice_specs:
            if difference > 0:
                slice_specs[-1]["quantity"] += difference
            else:
                deficit = -difference
                for spec in reversed(slice_specs):
                    if deficit <= 0:
                        break
                    reducible = min(spec["quantity"], deficit)
                    spec["quantity"] -= reducible
                    deficit -= reducible
                if deficit > 0:
                    remaining_tail = sum(spec["quantity"] for spec in slice_specs[1:])
                    slice_specs[0]["quantity"] = max(0, total_quantity - remaining_tail)

        final_sum = sum(spec["quantity"] for spec in slice_specs)
        if final_sum != total_quantity:
            if slice_specs:
                tail_sum = sum(spec["quantity"] for spec in slice_specs[:-1])
                slice_specs[-1]["quantity"] = max(0, total_quantity - tail_sum)
            else:
                slice_specs.append({"quantity": total_quantity, "kwargs": {}})

        results: list[Any] = []
        shared_kwargs = dict(kwargs)

        for spec in slice_specs:
            slice_qty = int(spec.get("quantity", 0) or 0)
            if slice_qty <= 0:
                continue
            slice_kwargs = dict(shared_kwargs)
            slice_kwargs.update(spec.get("kwargs", {}))
            per_order_type = slice_kwargs.pop("order_type", order_type)
            per_asset_class = slice_kwargs.pop("asset_class", asset_class)
            result = self.execute_order(
                symbol,
                side,
                slice_qty,
                per_order_type,
                asset_class=per_asset_class,
                **slice_kwargs,
            )
            results.append(result)

        return results

    def _coerce_signal_weight(self, explicit_weight: Any, signal: Any) -> float | None:
        """Best-effort conversion of signal weight to ``float``."""

        for candidate in (explicit_weight, getattr(signal, "weight", None)):
            if candidate is None:
                continue
            try:
                weight = float(candidate)
            except (TypeError, ValueError):
                continue
            else:
                if math.isfinite(weight):
                    return weight
        return None

    def mark_fill_reported(self, order_id: str, quantity: int) -> None:
        """Record quantity already forwarded to risk engine to avoid double-counting."""

        meta = self._order_signal_meta.get(order_id)
        if meta is None:
            return
        try:
            qty = int(quantity)
        except (TypeError, ValueError):
            return
        if qty < 0:
            return
        meta.reported_fill_qty = max(meta.reported_fill_qty, qty)
        if meta.reported_fill_qty >= meta.requested_qty:
            self._order_signal_meta.pop(order_id, None)

    def _handle_execution_event(self, order: Order, event_type: str) -> None:
        """Propagate late fills back to the trading context."""

        order_id = getattr(order, "id", None)
        if not order_id:
            return
        meta = self._order_signal_meta.get(order_id)
        if meta is None:
            return
        filled_qty = self._safe_int(getattr(order, "filled_quantity", 0))
        if filled_qty is None:
            filled_qty = 0
        delta = filled_qty - meta.reported_fill_qty
        if delta > 0:
            self._persist_meta_trade(order, delta, getattr(meta, "signal", None))
            if self._forward_risk_fill(order, delta, meta):
                meta.reported_fill_qty = filled_qty
        status = ExecutionResult._normalize_status(getattr(order, "status", None))
        if (status is not None and status.is_terminal) or event_type in {
            "completed",
            "cancelled",
            "canceled",
            "expired",
        }:
            if delta <= 0 and filled_qty > meta.reported_fill_qty:
                meta.reported_fill_qty = filled_qty
            if status is None or status.is_terminal or event_type in {"cancelled", "canceled", "expired"}:
                self._order_signal_meta.pop(order_id, None)

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _persist_meta_trade(self, order: Order, delta_qty: int, signal: Any | None) -> None:
        """Persist fills for meta-learning trade history."""

        if delta_qty <= 0 or signal is None:
            return
        try:
            fill = order.fills[-1] if getattr(order, "fills", None) else None
            price_obj = None
            timestamp = None
            fill_id = None
            if isinstance(fill, dict):
                price_obj = fill.get("price")
                timestamp = fill.get("timestamp")
                fill_id = fill.get("fill_id")
            if price_obj is None:
                price_obj = getattr(order, "average_fill_price", None)
            if timestamp is None:
                timestamp = getattr(order, "updated_at", datetime.now(UTC))
            if isinstance(price_obj, Money):
                price = float(price_obj)
            else:
                try:
                    price = float(price_obj) if price_obj is not None else None
                except (TypeError, ValueError):
                    price = None
            side_val = getattr(order.side, "value", order.side)
            signal_tags = getattr(signal, "signal_tags", None) or getattr(signal, "tags", "")
            try:
                confidence = float(getattr(signal, "confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            record_trade_fill(
                {
                    "symbol": getattr(order, "symbol", ""),
                    "entry_time": timestamp,
                    "entry_price": price,
                    "qty": int(delta_qty),
                    "side": str(side_val).lower(),
                    "strategy": getattr(signal, "strategy", ""),
                    "signal_tags": signal_tags,
                    "confidence": confidence,
                    "order_id": getattr(order, "id", None),
                    "fill_id": fill_id,
                }
            )
            # Log realized slippage to CSV and update EWMA feedback (best-effort)
            try:
                from ai_trading.execution.slippage_log import record_fill as _log_slip

                _log_slip(
                    symbol=getattr(order, "symbol", ""),
                    side=str(side_val).lower(),
                    qty=int(delta_qty),
                    expected_price=float(getattr(order, "expected_price", None) or 0) or None,
                    fill_price=price,
                    timestamp=timestamp if isinstance(timestamp, datetime) else datetime.now(UTC),
                )
            except Exception:
                logger.debug("SLIPPAGE_EWMA_UPDATE_FAILED", exc_info=True)
        except Exception:  # pragma: no cover - persistence best effort
            logger.debug("META_TRADE_PERSIST_FAILED", exc_info=True)

    def _forward_risk_fill(self, order: Order, delta_qty: int, meta: _SignalMeta) -> bool:
        """Forward ``delta_qty`` fills to the risk engine when possible."""

        if delta_qty <= 0:
            return False
        ctx = getattr(self, "ctx", None)
        risk_engine = getattr(ctx, "risk_engine", None) if ctx is not None else None
        if risk_engine is None:
            return False
        signal = meta.signal
        if signal is None:
            return False
        weight_delta = self._weight_for_delta(delta_qty, meta)
        if weight_delta is None:
            return False
        try:
            if is_dataclass(signal):
                fill_signal = replace(signal, weight=weight_delta)
            else:
                fill_signal = signal.__class__(**{**getattr(signal, "__dict__", {}), "weight": weight_delta})
        except Exception:
            self.logger.debug(
                "RISK_FILL_SIGNAL_CLONE_FAILED",
                extra={"symbol": getattr(signal, "symbol", None)},
                exc_info=True,
            )
            return False
        risk_engine.register_fill(fill_signal)
        return True

    def _weight_for_delta(self, delta_qty: int, meta: _SignalMeta) -> float | None:
        """Return proportional weight for ``delta_qty`` fills."""

        if meta.requested_qty <= 0:
            return None
        weight = meta.signal_weight
        if weight is None:
            return None
        try:
            proportion = float(delta_qty) / float(meta.requested_qty)
        except ZeroDivisionError:
            return None
        if not math.isfinite(proportion) or proportion <= 0:
            return None
        return weight * proportion

    def _guess_price(self, symbol: str) -> float | None:
        """Best-effort to obtain a reasonable price for simulation.

        Tries (in order):
        - ai_trading.core.bot_engine.get_latest_price (lazy import)
        - Returns None if unavailable/fails.
        """
        try:
            # Lazy import to avoid heavy imports at startup
            import importlib

            be = importlib.import_module("ai_trading.core.bot_engine")
            if hasattr(be, "get_latest_price"):
                p = be.get_latest_price(symbol)
                try:
                    return float(p) if p is not None else None
                except (TypeError, ValueError):
                    return None
        except Exception:
            self.logger.debug("EXPECTED_PRICE_GUESS_FAILED", extra={"symbol": symbol}, exc_info=True)
            return None
        return None

    def _adaptive_slippage_threshold(self, symbol: str, base_bps: float) -> float:
        """Return slippage threshold scaled for symbol volatility."""
        try:
            if self.market_data_feed and hasattr(self.market_data_feed, "get_volatility"):
                vol = self.market_data_feed.get_volatility(symbol)
                if isinstance(vol, (int, float)) and vol > 0:
                    factor = max(0.5, min(3.0, vol / 0.02))
                    return base_bps * factor
        except Exception as exc:
            self.logger.debug("VOLATILITY_LOOKUP_FAILED", extra={"symbol": symbol}, exc_info=exc)
        return base_bps

    def _apply_slippage(
        self,
        order: Order,
        delayed_price: float,
        reference_price: float,
        threshold_bps: float,
    ) -> float:
        """Validate delayed price against reference price.

        Records slippage in basis points and raises ``AssertionError`` when the
        deviation between ``delayed_price`` and ``reference_price`` exceeds
        ``threshold_bps``.
        """
        try:
            diff_bps = ((delayed_price - reference_price) / reference_price) * 10000 if reference_price else 0.0
        except Exception:
            diff_bps = 0.0
        order.slippage_bps = diff_bps
        side = _normalize_order_side(getattr(order, "side", None))
        if side is None:
            adverse_bps = abs(diff_bps)
        else:
            directional = diff_bps if side == OrderSide.BUY else -diff_bps
            adverse_bps = max(directional, 0.0)
        if adverse_bps > threshold_bps:
            logger.warning(
                "SLIPPAGE_PRECHECK_THRESHOLD_EXCEEDED",
                extra={
                    "symbol": order.symbol,
                    "order_id": order.id,
                    "delayed_price": round(delayed_price, 6),
                    "reference_price": round(reference_price, 6),
                    "slippage_bps": round(diff_bps, 2),
                    "adverse_slippage_bps": round(adverse_bps, 2),
                    "threshold_bps": round(threshold_bps, 2),
                },
            )
            raise AssertionError(f"delayed price {delayed_price} deviates {diff_bps:.2f} bps from reference")
        return delayed_price

    def _simulate_market_execution(self, order: Order):
        """Simulate market order execution (demo purposes)."""
        try:
            had_manual_price = False
            # Prefer provided price; else try to guess; else final fallback
            if order.price is not None:
                try:
                    base_price = float(order.price)
                except Exception:
                    base_price = 100.0
                had_manual_price = True
            else:
                base_price = self._guess_price(order.symbol) or 100.0
            try:
                if getattr(order, "expected_price", None) is not None:
                    expected = float(order.expected_price)
                elif order.price is not None:
                    expected = float(order.price)
                else:
                    expected = float(base_price)
            except Exception:
                expected = float(base_price)

            price_source = (
                getattr(order, "expected_price_source", None) or getattr(order, "price_source", None) or "unknown"
            )
            base_threshold_env = get_env("MAX_SLIPPAGE_BPS", str(order.max_slippage_bps), cast=float)
            apply_slippage_controls = bool(
                had_manual_price or (isinstance(price_source, str) and price_source.startswith("alpaca"))
            )
            try:
                base_threshold_candidate = float(base_threshold_env)
            except Exception:
                base_threshold_candidate = float(order.max_slippage_bps)
            if not math.isfinite(base_threshold_candidate) or base_threshold_candidate <= 0:
                try:
                    base_threshold_candidate = float(EXECUTION_PARAMETERS.get("MAX_SLIPPAGE_BPS", 50))
                except Exception:
                    base_threshold_candidate = 50.0
            if apply_slippage_controls:
                base_threshold = base_threshold_candidate
                if had_manual_price:
                    try:
                        manual_threshold = float(order.max_slippage_bps)
                    except Exception:
                        manual_threshold = base_threshold
                    else:
                        if math.isfinite(manual_threshold) and manual_threshold > 0:
                            base_threshold = min(base_threshold, manual_threshold)
                threshold = self._adaptive_slippage_threshold(order.symbol, base_threshold)
                if isinstance(price_source, str) and price_source.startswith("alpaca"):
                    base_price = self._apply_slippage(order, base_price, expected, threshold)
                else:
                    order.slippage_bps = 0.0
            else:
                order.slippage_bps = 0.0
                threshold = float("inf")
                base_threshold = float("inf")
            if getattr(order, "expected_price", None) is None and base_price:
                try:
                    tick = TICK_BY_SYMBOL.get(order.symbol)
                    order.expected_price = Money(base_price, tick)
                except Exception as exc:
                    self.logger.debug(
                        "EXPECTED_PRICE_ASSIGN_FAILED",
                        extra={"symbol": order.symbol, "order_id": order.id},
                        exc_info=exc,
                    )
            try:
                if getattr(order, "expected_price", None) is not None:
                    expected = float(order.expected_price)
                else:
                    expected = float(base_price)
            except Exception:
                expected = float(base_price)
            if not math.isfinite(expected) or expected <= 0:
                expected = float(base_price or 1.0)
            jitter_ratio = _deterministic_fill_jitter_ratio(
                order.id,
                order.symbol,
                getattr(order.side, "value", order.side),
            )
            predicted_fill = base_price * (1 + jitter_ratio)
            if expected:
                predicted_slippage_bps = ((predicted_fill - expected) / expected) * 10000
            else:
                base_ref = float(base_price) if base_price else 1.0
                predicted_slippage_bps = ((predicted_fill - base_ref) / base_ref) * 10000
            side = _normalize_order_side(getattr(order, "side", None))
            if side is None:
                adverse_predicted_bps = abs(predicted_slippage_bps)
            else:
                directional_predicted = predicted_slippage_bps if side == OrderSide.BUY else -predicted_slippage_bps
                adverse_predicted_bps = max(directional_predicted, 0.0)
            testing_mode = _env_bool("TESTING", False)
            effective_threshold = threshold
            if testing_mode and (not math.isfinite(effective_threshold) or effective_threshold <= 0):
                effective_threshold = base_threshold_candidate
                if not math.isfinite(effective_threshold) or effective_threshold <= 0:
                    try:
                        effective_threshold = float(EXECUTION_PARAMETERS.get("MAX_SLIPPAGE_BPS", 50))
                    except Exception:
                        effective_threshold = 50.0
            if testing_mode and adverse_predicted_bps > effective_threshold and effective_threshold > 0:
                tolerance_default = EXECUTION_PARAMETERS.get("SLIPPAGE_LIMIT_TOLERANCE_BPS", 25.0)
                tolerance_env = get_env(
                    "SLIPPAGE_LIMIT_TOLERANCE_BPS", str(tolerance_default), cast=float
                )
                try:
                    tolerance_bps = float(tolerance_env)
                except Exception:
                    tolerance_bps = float(tolerance_default)
                if not math.isfinite(tolerance_bps) or tolerance_bps < 0:
                    tolerance_bps = float(tolerance_default)
                reference_price = expected if expected and expected > 0 else base_price
                if reference_price and reference_price > 0:
                    direction = 1.0
                    if side in (OrderSide.SELL, OrderSide.SELL_SHORT):
                        direction = -1.0
                    tolerance_ratio = tolerance_bps / 10000.0
                    limit_price = reference_price + (direction * reference_price * tolerance_ratio)
                    if limit_price <= 0:
                        limit_price = reference_price
                    tick = TICK_BY_SYMBOL.get(order.symbol)
                    order.price = Money(limit_price, tick)
                    base_price = float(order.price)
                    try:
                        order.expected_price = Money(base_price, tick)
                    except Exception:
                        order.expected_price = Money(base_price)
                order.order_type = OrderType.LIMIT
                logger.warning(
                    "SLIPPAGE_LIMIT_CONVERSION",
                    extra={
                        "order_id": order.id,
                        "symbol": order.symbol,
                        "side": getattr(side, "value", str(side)),
                        "predicted_slippage_bps": round(predicted_slippage_bps, 2),
                        "threshold_bps": round(effective_threshold, 2),
                        "tolerance_bps": round(tolerance_bps, 2),
                        "limit_price": round(float(order.price), 6) if order.price else None,
                    },
                )
                original_qty = getattr(order, "quantity", 0)
                if isinstance(original_qty, int) and original_qty > 1 and adverse_predicted_bps > 0:
                    scale = effective_threshold / adverse_predicted_bps
                    if not math.isfinite(scale) or scale <= 0:
                        scale = 1.0 / original_qty
                    scale = max(0.0, min(1.0, scale))
                    adjusted_qty = int(max(1, math.floor(original_qty * scale)))
                    if adjusted_qty < original_qty:
                        order.quantity = adjusted_qty
                        logger.warning(
                            "SLIPPAGE_QTY_REDUCED",
                            extra={
                                "order_id": order.id,
                                "symbol": order.symbol,
                                "original_qty": original_qty,
                                "adjusted_qty": adjusted_qty,
                                "reduction_scale": round(scale, 6),
                            },
                        )
            elif adverse_predicted_bps > threshold:
                payload = {
                    "order_id": order.id,
                    "adverse_slippage_bps": round(adverse_predicted_bps, 2),
                    "threshold_bps": round(threshold, 2) if math.isfinite(threshold) else None,
                }
                if had_manual_price:
                    logger.warning("MANUAL_SLIPPAGE_THRESHOLD_EXCEEDED", extra=payload)
                else:
                    logger.warning("SLIPPAGE_THRESHOLD_EXCEEDED", extra=payload)
                raise AssertionError(
                    f"predicted slippage {predicted_slippage_bps:.2f} bps exceeds threshold"
                )

            remaining = order.quantity
            while remaining > 0 and order.status != OrderStatus.CANCELED:
                fill_quantity = min(remaining, max(1, remaining // 3))
                fill_price = base_price * (1 + jitter_ratio)
                order.add_fill(fill_quantity, fill_price)
                self._update_position(order.symbol, order.side, fill_quantity)
                remaining -= fill_quantity
                if remaining > 0:
                    time.sleep(0.1)
            if order.is_filled:
                self.execution_stats["filled_orders"] += 1
                # Ensure float accumulation to avoid Decimal/float TypeError
                try:
                    nv = float(getattr(order, "notional_value", 0.0))
                except Exception:
                    try:
                        avg = getattr(order, "average_fill_price", None)
                        nv = (
                            (float(avg) * float(order.filled_quantity))
                            if avg is not None
                            else float(base_price) * float(order.quantity)
                        )
                    except Exception:
                        nv = float(order.quantity) * float(base_price)
                self.execution_stats["total_volume"] += nv
                fill_time = (order.executed_at - order.created_at).total_seconds()
                self.execution_stats["average_fill_time"] = (
                    self.execution_stats["average_fill_time"] * (self.execution_stats["filled_orders"] - 1) + fill_time
                ) / self.execution_stats["filled_orders"]

                # Compare expected vs actual fill price and check slippage
                try:
                    if getattr(order, "expected_price", None) is not None:
                        expected = float(order.expected_price)
                    elif order.price is not None:
                        expected = float(order.price)
                    else:
                        expected = float(base_price)
                except Exception:
                    expected = float(base_price)
                if not math.isfinite(expected) or expected <= 0:
                    expected = float(base_price or 1.0)
                try:
                    actual = float(order.average_fill_price) if order.average_fill_price is not None else expected
                except Exception:
                    actual = expected
                try:
                    limit_price_val = float(order.price) if order.price is not None else None
                except Exception:
                    limit_price_val = None
                if limit_price_val and limit_price_val > 0 and side is not None:
                    side_value = getattr(side, "value", str(side)).lower()
                    if side_value.startswith("buy"):
                        actual = min(actual, limit_price_val)
                    elif side_value.startswith("sell"):
                        actual = max(actual, limit_price_val)
                if expected:
                    slippage_bps = ((actual - expected) / expected) * 10000
                else:
                    ref = float(base_price) if base_price else 1.0
                    slippage_bps = ((actual - ref) / ref) * 10000
                order.slippage_bps = slippage_bps
                logger.info(
                    "SLIPPAGE_DIAGNOSTIC",
                    extra={
                        "symbol": order.symbol,
                        "expected_price": round(expected, 4),
                        "actual_price": round(actual, 4),
                        "slippage_bps": round(slippage_bps, 2),
                    },
                )
                threshold = self._adaptive_slippage_threshold(order.symbol, base_threshold)
                if side is None:
                    adverse_actual_bps = abs(slippage_bps)
                else:
                    directional_actual = slippage_bps if side == OrderSide.BUY else -slippage_bps
                    adverse_actual_bps = max(directional_actual, 0.0)
                if adverse_actual_bps > threshold:
                    logger.warning(
                        "SLIPPAGE_THRESHOLD_EXCEEDED",
                        extra={
                            "order_id": order.id,
                            "slippage_bps": round(slippage_bps, 2),
                            "adverse_slippage_bps": round(adverse_actual_bps, 2),
                            "threshold_bps": threshold,
                        },
                    )
                    testing_flag = os.getenv("TESTING", "").strip().lower()
                    exec_strict = _env_bool("EXECUTION_STRICT", False)
                    if exec_strict or testing_flag in {"1", "true", "yes"}:
                        raise AssertionError(
                            "SLIPPAGE_THRESHOLD_EXCEEDED: predicted slippage exceeds limit"
                        )
        except (KeyError, ValueError, TypeError, RuntimeError) as e:
            logger.error(
                "SIMULATION_FAILED", extra={"cause": e.__class__.__name__, "detail": str(e), "order_id": order.id}
            )

    def get_execution_stats(self) -> dict:
        """Get execution engine statistics."""
        stats = self.execution_stats.copy()
        stats["active_orders"] = len(self.order_manager.active_orders)
        stats["success_rate"] = stats["filled_orders"] / stats["total_orders"] if stats["total_orders"] > 0 else 0
        return stats
