"""OMS pre-trade controls for size, collars, duplicates, and throttles."""
from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, time as dt_time
import json
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Any, Mapping, cast
from zoneinfo import ZoneInfo

from ai_trading.config.management import get_env
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path
from ai_trading.utils.market_calendar import is_trading_day, session_info


@dataclass
class OrderIntent:
    symbol: str
    side: str
    qty: int
    notional: float
    limit_price: float | None
    bar_ts: datetime
    client_order_id: str
    last_price: float | None = None
    mid: float | None = None
    bid: float | None = None
    ask: float | None = None
    spread: float | None = None
    quote_ts: datetime | None = None
    quote_age_ms: float | None = None
    submit_quote_source: str | None = None
    sleeve: str | None = None
    liquidity_bucket: str | None = None
    avg_daily_volume: float | None = None
    minute_volume: float | None = None
    expected_slippage_bps: float | None = None
    expected_tca_bps: float | None = None
    fill_quality_score: float | None = None
    quote_quality_ok: bool | None = None
    sector: str | None = None
    factor_name: str | None = None
    factor_exposure: float | None = None
    event_risk: bool | None = None
    event_type: str | None = None
    execution_drift_bps: float | None = None
    reject_rate_pct: float | None = None
    session_regime: str | None = None
    opening_trade: bool | None = None
    require_realtime_nbbo: bool | None = None
    kill_switch_active: bool | None = None
    kill_switch_reason: str | None = None
    broker_ready: bool | None = None
    broker_ready_reason: str | None = None
    broker_cooldown_remaining_sec: float | None = None

    def to_contract(self):
        """Return the canonical order intent contract for downstream journals."""
        from ai_trading.contracts import OrderIntent as CanonicalOrderIntent

        return CanonicalOrderIntent.from_pretrade(self)


_LIVE_COST_MODEL_CACHE: dict[str, Any] = {
    "path": None,
    "mtime_ns": None,
    "payload": None,
}
_LIVE_COST_MODEL_LOCK = RLock()


class SlidingWindowRateLimiter:
    """Rate limiter for order and cancel message budgets."""

    def __init__(
        self,
        *,
        global_orders_per_min: int = 0,
        per_symbol_orders_per_min: int = 0,
        cancels_per_min: int = 0,
        cancel_loop_max_without_fill: int = 0,
        cancel_loop_block_bars: int = 0,
        state_path: str | Path | None = None,
    ) -> None:
        self.global_orders_per_min = max(0, int(global_orders_per_min))
        self.per_symbol_orders_per_min = max(0, int(per_symbol_orders_per_min))
        self.cancels_per_min = max(0, int(cancels_per_min))
        self.cancel_loop_max_without_fill = max(0, int(cancel_loop_max_without_fill))
        self.cancel_loop_block_bars = max(0, int(cancel_loop_block_bars))
        self.state_path = (
            Path(state_path).expanduser().resolve() if state_path is not None else None
        )
        self._lock = RLock()

        self._order_ts: deque[float] = deque()
        self._symbol_order_ts: dict[str, deque[float]] = {}
        self._cancel_ts: deque[float] = deque()
        self._cancel_without_fill: dict[str, int] = {}
        self._symbol_bar_index: dict[str, int] = {}
        self._symbol_last_bar_ts: dict[str, datetime] = {}
        self._symbol_block_until_bar: dict[str, int] = {}
        if self.state_path is not None:
            self._initialize_state_store()

    @staticmethod
    def _now() -> float:
        import time

        return time.time()

    @staticmethod
    def _prune(window: deque[float], now: float, seconds: float) -> None:
        while window and now - window[0] > seconds:
            window.popleft()

    @staticmethod
    def _coerce_bar_ts(bar_ts: datetime) -> datetime:
        return bar_ts if bar_ts.tzinfo is not None else bar_ts.replace(tzinfo=UTC)

    @staticmethod
    def _symbol_key(symbol: str) -> str:
        return str(symbol).strip().upper()

    def _initialize_state_store(self) -> None:
        assert self.state_path is not None
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with self._db_connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pretrade_rate_events (
                    kind TEXT NOT NULL,
                    symbol TEXT,
                    ts REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_pretrade_rate_events_kind_ts
                ON pretrade_rate_events (kind, ts)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_pretrade_rate_events_kind_symbol_ts
                ON pretrade_rate_events (kind, symbol, ts)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pretrade_symbol_state (
                    symbol TEXT PRIMARY KEY,
                    last_bar_ts TEXT,
                    bar_index INTEGER NOT NULL DEFAULT 0,
                    cancel_without_fill INTEGER NOT NULL DEFAULT 0,
                    block_until_bar INTEGER NOT NULL DEFAULT 0
                )
                """
            )

    def _db_connect(self) -> sqlite3.Connection:
        if self.state_path is None:
            raise RuntimeError("Durable pretrade state is not configured.")
        conn = sqlite3.connect(str(self.state_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _load_symbol_state_db(
        self,
        conn: sqlite3.Connection,
        symbol: str,
    ) -> tuple[datetime | None, int, int, int]:
        row = conn.execute(
            """
            SELECT last_bar_ts, bar_index, cancel_without_fill, block_until_bar
            FROM pretrade_symbol_state
            WHERE symbol = ?
            """,
            (self._symbol_key(symbol),),
        ).fetchone()
        if row is None:
            return None, 0, 0, 0
        raw_last_bar_ts, raw_bar_index, raw_cancel_without_fill, raw_block_until_bar = row
        last_bar_ts: datetime | None = None
        if isinstance(raw_last_bar_ts, str) and raw_last_bar_ts.strip():
            try:
                last_bar_ts = datetime.fromisoformat(raw_last_bar_ts)
            except ValueError:
                last_bar_ts = None
        return (
            self._coerce_bar_ts(last_bar_ts) if last_bar_ts is not None else None,
            int(raw_bar_index or 0),
            int(raw_cancel_without_fill or 0),
            int(raw_block_until_bar or 0),
        )

    def _persist_symbol_state_db(
        self,
        conn: sqlite3.Connection,
        *,
        symbol: str,
        last_bar_ts: datetime | None,
        bar_index: int,
        cancel_without_fill: int,
        block_until_bar: int,
    ) -> None:
        conn.execute(
            """
            INSERT INTO pretrade_symbol_state (
                symbol,
                last_bar_ts,
                bar_index,
                cancel_without_fill,
                block_until_bar
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                last_bar_ts = excluded.last_bar_ts,
                bar_index = excluded.bar_index,
                cancel_without_fill = excluded.cancel_without_fill,
                block_until_bar = excluded.block_until_bar
            """,
            (
                self._symbol_key(symbol),
                last_bar_ts.astimezone(UTC).isoformat() if last_bar_ts is not None else None,
                int(bar_index),
                int(cancel_without_fill),
                int(block_until_bar),
            ),
        )

    def _advance_bar_db(
        self,
        conn: sqlite3.Connection,
        symbol: str,
        bar_ts: datetime,
    ) -> tuple[int, int, int]:
        ts = self._coerce_bar_ts(bar_ts)
        previous, bar_index, cancel_without_fill, block_until_bar = self._load_symbol_state_db(
            conn,
            symbol,
        )
        if previous is None or ts > previous:
            previous = ts
            bar_index += 1
            self._persist_symbol_state_db(
                conn,
                symbol=symbol,
                last_bar_ts=previous,
                bar_index=bar_index,
                cancel_without_fill=cancel_without_fill,
                block_until_bar=block_until_bar,
            )
        return bar_index, cancel_without_fill, block_until_bar

    def _prune_db(self, conn: sqlite3.Connection, *, now: float) -> None:
        conn.execute(
            "DELETE FROM pretrade_rate_events WHERE ts < ?",
            (float(now - 60.0),),
        )

    def _advance_bar(self, symbol: str, bar_ts: datetime) -> int:
        symbol_key = self._symbol_key(symbol)
        ts = self._coerce_bar_ts(bar_ts)
        previous = self._symbol_last_bar_ts.get(symbol_key)
        if previous is None or ts > previous:
            self._symbol_last_bar_ts[symbol_key] = ts
            self._symbol_bar_index[symbol_key] = self._symbol_bar_index.get(symbol_key, 0) + 1
        return self._symbol_bar_index.get(symbol_key, 0)

    def _cancel_rate_ok_in_memory(self, now: float) -> bool:
        if self.cancels_per_min <= 0:
            return True
        self._prune(self._cancel_ts, now, 60.0)
        return len(self._cancel_ts) < self.cancels_per_min

    def _allow_order_in_memory(
        self,
        symbol: str,
        bar_ts: datetime,
        *,
        reserve_slot: bool,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        symbol_key = self._symbol_key(symbol)
        now = self._now()
        bar_idx = self._advance_bar(symbol_key, bar_ts)
        blocked_until = self._symbol_block_until_bar.get(symbol_key, 0)
        if blocked_until and bar_idx < blocked_until:
            return (
                False,
                "CANCEL_LOOP_BLOCK",
                {"symbol": symbol_key, "blocked_until_bar": blocked_until},
            )

        self._prune(self._order_ts, now, 60.0)
        if self.global_orders_per_min > 0 and len(self._order_ts) >= self.global_orders_per_min:
            return False, "RATE_THROTTLE_BLOCK", {"scope": "global", "limit": self.global_orders_per_min}

        symbol_window = self._symbol_order_ts.setdefault(symbol_key, deque())
        self._prune(symbol_window, now, 60.0)
        if (
            self.per_symbol_orders_per_min > 0
            and len(symbol_window) >= self.per_symbol_orders_per_min
        ):
            return (
                False,
                "RATE_THROTTLE_BLOCK",
                {"scope": "symbol", "symbol": symbol_key, "limit": self.per_symbol_orders_per_min},
            )

        if not self._cancel_rate_ok_in_memory(now):
            return False, "CANCEL_RATE_BLOCK", {"limit": self.cancels_per_min}

        if reserve_slot:
            self._order_ts.append(now)
            symbol_window.append(now)
        return True, None, {}

    def _allow_order_db(
        self,
        symbol: str,
        bar_ts: datetime,
        *,
        reserve_slot: bool,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        symbol_key = self._symbol_key(symbol)
        now = self._now()
        with self._db_connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._prune_db(conn, now=now)
            bar_idx, _cancel_without_fill, block_until_bar = self._advance_bar_db(
                conn,
                symbol_key,
                bar_ts,
            )
            if block_until_bar and bar_idx < block_until_bar:
                return (
                    False,
                    "CANCEL_LOOP_BLOCK",
                    {"symbol": symbol_key, "blocked_until_bar": block_until_bar},
                )
            if self.cancels_per_min > 0:
                cancel_count = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM pretrade_rate_events WHERE kind = 'cancel'",
                    ).fetchone()[0]
                )
                if cancel_count >= self.cancels_per_min:
                    return False, "CANCEL_RATE_BLOCK", {"limit": self.cancels_per_min}
            if self.global_orders_per_min > 0:
                global_count = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM pretrade_rate_events WHERE kind = 'order'",
                    ).fetchone()[0]
                )
                if global_count >= self.global_orders_per_min:
                    return (
                        False,
                        "RATE_THROTTLE_BLOCK",
                        {"scope": "global", "limit": self.global_orders_per_min},
                    )
            if self.per_symbol_orders_per_min > 0:
                symbol_count = int(
                    conn.execute(
                        """
                        SELECT COUNT(*)
                        FROM pretrade_rate_events
                        WHERE kind = 'order' AND symbol = ?
                        """,
                        (symbol_key,),
                    ).fetchone()[0]
                )
                if symbol_count >= self.per_symbol_orders_per_min:
                    return (
                        False,
                        "RATE_THROTTLE_BLOCK",
                        {
                            "scope": "symbol",
                            "symbol": symbol_key,
                            "limit": self.per_symbol_orders_per_min,
                        },
                    )
            if reserve_slot:
                conn.execute(
                    """
                    INSERT INTO pretrade_rate_events (kind, symbol, ts)
                    VALUES ('order', ?, ?)
                    """,
                    (symbol_key, float(now)),
                )
        return True, None, {}

    def allow_order(self, symbol: str, bar_ts: datetime) -> tuple[bool, str | None, dict[str, Any]]:
        with self._lock:
            if self.state_path is not None:
                return self._allow_order_db(symbol, bar_ts, reserve_slot=False)
            return self._allow_order_in_memory(symbol, bar_ts, reserve_slot=False)

    def allow_and_record_order(
        self,
        symbol: str,
        bar_ts: datetime,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        """Atomically validate and reserve an order pacing slot."""

        with self._lock:
            if self.state_path is not None:
                return self._allow_order_db(symbol, bar_ts, reserve_slot=True)
            return self._allow_order_in_memory(symbol, bar_ts, reserve_slot=True)

    def record_order(self, symbol: str, bar_ts: datetime) -> None:
        with self._lock:
            if self.state_path is not None:
                now = self._now()
                symbol_key = self._symbol_key(symbol)
                with self._db_connect() as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    self._prune_db(conn, now=now)
                    self._advance_bar_db(conn, symbol_key, bar_ts)
                    conn.execute(
                        """
                        INSERT INTO pretrade_rate_events (kind, symbol, ts)
                        VALUES ('order', ?, ?)
                        """,
                        (symbol_key, float(now)),
                    )
                return
            now = self._now()
            symbol_key = self._symbol_key(symbol)
            self._advance_bar(symbol_key, bar_ts)
            self._order_ts.append(now)
            symbol_window = self._symbol_order_ts.setdefault(symbol_key, deque())
            symbol_window.append(now)

    def record_cancel(self, symbol: str, *, bar_ts: datetime, filled: bool) -> None:
        with self._lock:
            now = self._now()
            symbol_key = self._symbol_key(symbol)
            if self.state_path is not None:
                with self._db_connect() as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    self._prune_db(conn, now=now)
                    conn.execute(
                        """
                        INSERT INTO pretrade_rate_events (kind, symbol, ts)
                        VALUES ('cancel', ?, ?)
                        """,
                        (symbol_key, float(now)),
                    )
                    last_bar_ts, bar_index, cancel_without_fill, block_until_bar = (
                        self._load_symbol_state_db(conn, symbol_key)
                    )
                    coerced_bar_ts = self._coerce_bar_ts(bar_ts)
                    if filled:
                        cancel_without_fill = 0
                    else:
                        cancel_without_fill += 1
                        if (
                            self.cancel_loop_max_without_fill > 0
                            and cancel_without_fill >= self.cancel_loop_max_without_fill
                            and self.cancel_loop_block_bars > 0
                        ):
                            bar_index, _current_without_fill, _current_block_until = self._advance_bar_db(
                                conn,
                                symbol_key,
                                bar_ts,
                            )
                            block_until_bar = bar_index + self.cancel_loop_block_bars
                    self._persist_symbol_state_db(
                        conn,
                        symbol=symbol_key,
                        last_bar_ts=max(
                            filter(None, [last_bar_ts, coerced_bar_ts]),
                            default=coerced_bar_ts,
                        ),
                        bar_index=bar_index,
                        cancel_without_fill=cancel_without_fill,
                        block_until_bar=block_until_bar,
                    )
                return

            self._cancel_ts.append(now)
            self._prune(self._cancel_ts, now, 60.0)

            if filled:
                self._cancel_without_fill[symbol_key] = 0
                return

            current = self._cancel_without_fill.get(symbol_key, 0) + 1
            self._cancel_without_fill[symbol_key] = current
            if (
                self.cancel_loop_max_without_fill > 0
                and current >= self.cancel_loop_max_without_fill
                and self.cancel_loop_block_bars > 0
            ):
                bar_idx = self._advance_bar(symbol_key, bar_ts)
                self._symbol_block_until_bar[symbol_key] = bar_idx + self.cancel_loop_block_bars

    def cancel_rate_ok(self) -> bool:
        with self._lock:
            if self.state_path is not None:
                if self.cancels_per_min <= 0:
                    return True
                now = self._now()
                with self._db_connect() as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    self._prune_db(conn, now=now)
                    cancel_count = int(
                        conn.execute(
                            "SELECT COUNT(*) FROM pretrade_rate_events WHERE kind = 'cancel'",
                        ).fetchone()[0]
                    )
                return cancel_count < self.cancels_per_min
            return self._cancel_rate_ok_in_memory(self._now())


def _cfg_value(
    cfg: Any,
    *,
    field: str,
    env_keys: tuple[str, ...],
    default: Any,
    cast: type[float] | type[int],
) -> float | int:
    raw = getattr(cfg, field, None) if cfg is not None else None
    if raw is not None:
        try:
            return cast(raw)
        except (TypeError, ValueError):
            pass
    for env_key in env_keys:
        raw_env = get_env(env_key, None)
        if raw_env is None:
            continue
        return cast(get_env(env_key, default, cast=cast))
    return cast(default)


def _coerce_utc_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _intent_regular_hours_open(intent: OrderIntent) -> bool:
    ts = _coerce_utc_datetime(intent.bar_ts)
    if ts is None:
        return False
    session_date = ts.astimezone(ZoneInfo("America/New_York")).date()
    if not is_trading_day(session_date):
        return False
    try:
        session = session_info(session_date)
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        ts_et = ts.astimezone(ZoneInfo("America/New_York"))
        minute_of_day = (ts_et.hour * 60) + ts_et.minute
        return ts_et.weekday() < 5 and ((9 * 60) + 30) <= minute_of_day < (16 * 60)
    return session.start_utc <= ts < session.end_utc


def _effective_quote_age_ms(intent: OrderIntent) -> float | None:
    if intent.quote_age_ms is not None:
        try:
            return max(0.0, float(intent.quote_age_ms))
        except (TypeError, ValueError):
            return None
    quote_ts = _coerce_utc_datetime(intent.quote_ts)
    if quote_ts is None:
        return None
    return max((datetime.now(UTC) - quote_ts.astimezone(UTC)).total_seconds() * 1000.0, 0.0)


def _ledger_fingerprints(ledger: Any) -> set[tuple[str, str, int, str]]:
    if ledger is None:
        return set()
    seen = getattr(ledger, "_pretrade_seen_fingerprints", None)
    if isinstance(seen, set):
        return seen
    seen = set()
    setattr(ledger, "_pretrade_seen_fingerprints", seen)
    return cast(set[tuple[str, str, int, str]], seen)


def _ledger_position_qty(ledger: Any, symbol: str) -> float | None:
    if ledger is None:
        return None
    symbol_norm = str(symbol).upper()
    method_names = ("position_qty", "get_position_qty", "current_position_qty")
    for method_name in method_names:
        method = getattr(ledger, method_name, None)
        if callable(method):
            try:
                return float(method(symbol_norm))
            except (TypeError, ValueError):
                continue
    positions = getattr(ledger, "positions", None)
    if isinstance(positions, dict):
        raw_position = positions.get(symbol_norm)
        if raw_position is None:
            return 0.0
        if isinstance(raw_position, (int, float)):
            return float(raw_position)
        raw_qty = getattr(raw_position, "qty", None)
        if raw_qty is not None:
            try:
                return float(raw_qty)
            except (TypeError, ValueError):
                return None
    return None


def _ledger_gross_notional(ledger: Any) -> float | None:
    if ledger is None:
        return None
    method_names = (
        "gross_notional",
        "get_gross_notional",
        "current_gross_notional",
    )
    for method_name in method_names:
        method = getattr(ledger, method_name, None)
        if callable(method):
            try:
                return float(method())
            except (TypeError, ValueError):
                continue
    attr_names = ("gross_notional", "gross_exposure_notional")
    for attr_name in attr_names:
        value = getattr(ledger, attr_name, None)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _json_env_dict(name: str, default: dict[str, float]) -> dict[str, float]:
    raw = get_env(name, None)
    if raw is None:
        return dict(default)
    parsed: Any = raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return dict(default)
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError, json.JSONDecodeError):
            return dict(default)
    if not isinstance(parsed, dict):
        return dict(default)
    out: dict[str, float] = {}
    for key, value in parsed.items():
        try:
            out[str(key).upper()] = float(value)
        except (TypeError, ValueError):
            continue
    if not out:
        return dict(default)
    return out


def _parse_utc_datetime_text(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
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


def _live_cost_model_payload() -> Mapping[str, Any] | None:
    if not _live_cost_model_gate_enabled():
        return None
    configured_path = str(
        get_env(
            "AI_TRADING_LIVE_COST_MODEL_PATH",
            "runtime/live_cost_model_latest.json",
            cast=str,
            resolve_aliases=False,
        )
        or "runtime/live_cost_model_latest.json"
    ).strip()
    path = resolve_runtime_artifact_path(
        configured_path,
        default_relative="runtime/live_cost_model_latest.json",
    )
    try:
        stat_result = path.stat()
    except OSError:
        return None
    with _LIVE_COST_MODEL_LOCK:
        if (
            _LIVE_COST_MODEL_CACHE.get("path") == str(path)
            and _LIVE_COST_MODEL_CACHE.get("mtime_ns") == stat_result.st_mtime_ns
        ):
            cached = _LIVE_COST_MODEL_CACHE.get("payload")
            return cached if isinstance(cached, Mapping) else None
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            parsed = None
        payload = parsed if isinstance(parsed, Mapping) else None
        _LIVE_COST_MODEL_CACHE.update(
            {
                "path": str(path),
                "mtime_ns": stat_result.st_mtime_ns,
                "payload": payload,
            }
        )
    return payload


def _live_cost_model_gate_enabled() -> bool:
    return bool(get_env("AI_TRADING_PRETRADE_LIVE_COST_MODEL_GATE_ENABLED", True, cast=bool))


_SYMBOL_UNIVERSE_SCORECARD_LOCK = RLock()
_SYMBOL_UNIVERSE_SCORECARD_CACHE: dict[str, Any] = {}

_RUNTIME_DECAY_CONTROLS_LOCK = RLock()
_RUNTIME_DECAY_CONTROLS_CACHE: dict[str, Any] = {}


def _symbol_universe_scorecard_payload() -> Mapping[str, Any] | None:
    if not bool(get_env("AI_TRADING_PRETRADE_SYMBOL_UNIVERSE_GATE_ENABLED", False, cast=bool)):
        return None
    configured_path = str(
        get_env(
            "AI_TRADING_SYMBOL_UNIVERSE_SCORECARD_PATH",
            "runtime/symbol_universe_scorecard_latest.json",
            cast=str,
            resolve_aliases=False,
        )
        or "runtime/symbol_universe_scorecard_latest.json"
    ).strip()
    path = resolve_runtime_artifact_path(
        configured_path,
        default_relative="runtime/symbol_universe_scorecard_latest.json",
    )
    try:
        stat_result = path.stat()
    except OSError:
        return None
    with _SYMBOL_UNIVERSE_SCORECARD_LOCK:
        if (
            _SYMBOL_UNIVERSE_SCORECARD_CACHE.get("path") == str(path)
            and _SYMBOL_UNIVERSE_SCORECARD_CACHE.get("mtime_ns") == stat_result.st_mtime_ns
        ):
            cached = _SYMBOL_UNIVERSE_SCORECARD_CACHE.get("payload")
            return cached if isinstance(cached, Mapping) else None
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            parsed = None
        payload = parsed if isinstance(parsed, Mapping) else None
        _SYMBOL_UNIVERSE_SCORECARD_CACHE.update(
            {
                "path": str(path),
                "mtime_ns": stat_result.st_mtime_ns,
                "payload": payload,
            }
        )
    return payload


def _symbol_universe_scorecard_usable(payload: Mapping[str, Any]) -> bool:
    if str(payload.get("artifact_type") or "") != "symbol_universe_scorecard":
        return False
    generated_at = _parse_utc_datetime_text(payload.get("generated_at"))
    if generated_at is None:
        return False
    max_age_minutes = _finite_float(
        get_env("AI_TRADING_SYMBOL_UNIVERSE_SCORECARD_MAX_AGE_MINUTES", 1440.0, cast=float)
    )
    if max_age_minutes is not None and max_age_minutes > 0.0:
        if (datetime.now(UTC) - generated_at).total_seconds() > max_age_minutes * 60.0:
            return False
    status = payload.get("status")
    return isinstance(status, Mapping) and bool(status.get("available"))


def _symbol_universe_mode(symbol: str) -> tuple[str, Mapping[str, Any] | None]:
    payload = _symbol_universe_scorecard_payload()
    if payload is None or not _symbol_universe_scorecard_usable(payload):
        return "allow", None
    rows = payload.get("symbols")
    if not isinstance(rows, list):
        return "allow", None
    symbol_token = str(symbol or "").strip().upper()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("symbol") or "").strip().upper() != symbol_token:
            continue
        mode = str(row.get("effective_mode") or "allow").strip().lower()
        if mode in {"allow", "shadow_only", "disabled"}:
            return mode, row
    return "allow", None


def _symbol_universe_pretrade_gate(
    intent: OrderIntent,
    *,
    ledger: Any,
) -> tuple[bool, str, dict[str, Any]] | None:
    if _intent_reduces_position(intent, ledger):
        return None
    mode, row = _symbol_universe_mode(intent.symbol)
    if mode == "allow":
        return None
    if mode == "shadow_only" and not bool(
        get_env("AI_TRADING_PRETRADE_SYMBOL_UNIVERSE_BLOCK_SHADOW_ONLY", True, cast=bool)
    ):
        return None
    return (
        False,
        "SYMBOL_UNIVERSE_MODE_BLOCK",
        {
            "symbol": str(intent.symbol or "").strip().upper(),
            "mode": mode,
            "sample_count": row.get("sample_count") if isinstance(row, Mapping) else None,
            "persistence_count": (
                row.get("persistence_count") if isinstance(row, Mapping) else None
            ),
            "reasons": row.get("reasons") if isinstance(row, Mapping) else None,
        },
    )


def _runtime_decay_controls_payload() -> Mapping[str, Any] | None:
    if not bool(get_env("AI_TRADING_PRETRADE_RUNTIME_DECAY_GATE_ENABLED", True, cast=bool)):
        return None
    configured_path = str(
        get_env(
            "AI_TRADING_RUNTIME_DECAY_CONTROLS_PATH",
            "runtime/runtime_decay_controls_latest.json",
            cast=str,
            resolve_aliases=False,
        )
        or "runtime/runtime_decay_controls_latest.json"
    ).strip()
    path = resolve_runtime_artifact_path(
        configured_path,
        default_relative="runtime/runtime_decay_controls_latest.json",
    )
    try:
        stat_result = path.stat()
    except OSError:
        return None
    with _RUNTIME_DECAY_CONTROLS_LOCK:
        if (
            _RUNTIME_DECAY_CONTROLS_CACHE.get("path") == str(path)
            and _RUNTIME_DECAY_CONTROLS_CACHE.get("mtime_ns") == stat_result.st_mtime_ns
        ):
            cached = _RUNTIME_DECAY_CONTROLS_CACHE.get("payload")
            return cached if isinstance(cached, Mapping) else None
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            parsed = None
        payload = parsed if isinstance(parsed, Mapping) else None
        _RUNTIME_DECAY_CONTROLS_CACHE.update(
            {"path": str(path), "mtime_ns": stat_result.st_mtime_ns, "payload": payload}
        )
    return payload


def _runtime_decay_controls_usable(payload: Mapping[str, Any]) -> bool:
    if str(payload.get("artifact_type") or "") != "runtime_decay_controls":
        return False
    generated_at = _parse_utc_datetime_text(payload.get("generated_at"))
    if generated_at is None:
        return False
    max_age_minutes = _finite_float(
        get_env("AI_TRADING_RUNTIME_DECAY_CONTROLS_MAX_AGE_MINUTES", 1440.0, cast=float)
    )
    if max_age_minutes is not None and max_age_minutes > 0.0:
        if (datetime.now(UTC) - generated_at).total_seconds() > max_age_minutes * 60.0:
            return False
    status = payload.get("status")
    return isinstance(status, Mapping) and bool(status.get("available", True))


def _live_or_canary_opening(intent: OrderIntent, cfg: Any) -> bool:
    if not bool(intent.opening_trade):
        return False
    execution_mode = str(getattr(cfg, "execution_mode", "") or "").strip().lower()
    launch_profile = str(
        getattr(
            cfg,
            "launch_profile",
            get_env("AI_TRADING_LAUNCH_PROFILE", "", cast=str),
        )
        or ""
    ).strip().lower()
    return execution_mode == "live" or launch_profile == "live_canary" or launch_profile.startswith("live_")


def _runtime_decay_pretrade_gate(
    intent: OrderIntent,
    *,
    ledger: Any,
    cfg: Any,
) -> tuple[bool, str, dict[str, Any]] | None:
    if _intent_reduces_position(intent, ledger):
        return None
    payload = _runtime_decay_controls_payload()
    if payload is None:
        return None
    if not _runtime_decay_controls_usable(payload):
        if _live_or_canary_opening(intent, cfg):
            return (
                False,
                "RUNTIME_DECAY_ARTIFACT_STALE_BLOCK",
                {
                    "artifact_type": payload.get("artifact_type"),
                    "generated_at": payload.get("generated_at"),
                    "opening_trade": bool(intent.opening_trade),
                },
            )
        return None
    actions = payload.get("actions")
    if not isinstance(actions, Mapping):
        return None
    size_scale = _finite_float(actions.get("size_scale"))
    max_action = str(actions.get("max_action") or "normal").strip().lower()
    entries_allowed = bool(actions.get("entries_allowed", True))
    reduce_size_unconsumed = (
        max_action == "reduce_size"
        and size_scale is not None
        and 0.0 <= float(size_scale) < 1.0
    )
    if entries_allowed and not reduce_size_unconsumed:
        return None
    reasons_raw = actions.get("reasons")
    reasons = list(reasons_raw) if isinstance(reasons_raw, list) else []
    return (
        False,
        "RUNTIME_DECAY_CONTROL_BLOCK",
        {
            "action": max_action or "disable_new_entries",
            "size_scale": size_scale,
            "reasons": reasons,
            "generated_at": payload.get("generated_at"),
            "fail_safe": bool(entries_allowed and reduce_size_unconsumed),
        },
    )


def _live_cost_model_usable(payload: Mapping[str, Any]) -> bool:
    if str(payload.get("artifact_type") or "") != "live_cost_model":
        return False
    max_age_minutes = _finite_float(
        get_env(
            "AI_TRADING_PRETRADE_LIVE_COST_MODEL_MAX_AGE_MINUTES",
            1440.0,
            cast=float,
        )
    )
    generated_at = _parse_utc_datetime_text(payload.get("generated_at"))
    if generated_at is None:
        return False
    if max_age_minutes is not None and max_age_minutes > 0.0:
        age_seconds = (datetime.now(UTC) - generated_at).total_seconds()
        if age_seconds > max_age_minutes * 60.0:
            return False
    status = payload.get("status")
    if not isinstance(status, Mapping):
        return False
    require_ready = bool(
        get_env(
            "AI_TRADING_PRETRADE_LIVE_COST_MODEL_REQUIRE_READY",
            True,
            cast=bool,
        )
    )
    if require_ready and str(status.get("status") or "").lower() != "ready":
        return False
    return bool(status.get("available"))


def _live_cost_artifact_pretrade_gate(
    intent: OrderIntent,
    *,
    cfg: Any,
) -> tuple[bool, str, dict[str, Any]] | None:
    if not _live_cost_model_gate_enabled():
        return None
    payload = _live_cost_model_payload()
    if payload is None:
        if not _live_or_canary_opening(intent, cfg):
            return None
        return (
            False,
            "LIVE_COST_ARTIFACT_MISSING_BLOCK",
            {
                "opening_trade": bool(intent.opening_trade),
            },
        )
    if _live_cost_model_usable(payload):
        return None
    if not _live_or_canary_opening(intent, cfg):
        return None
    return (
        False,
        "LIVE_COST_ARTIFACT_STALE_BLOCK",
        {
            "artifact_type": payload.get("artifact_type"),
            "generated_at": payload.get("generated_at"),
            "opening_trade": bool(intent.opening_trade),
        },
    )


def _normalized_live_cost_side(side: str | None) -> str:
    token = str(side or "").strip().lower()
    if token in {"short", "sell_short", "sellshort"}:
        return "sell_short"
    if token in {"cover", "buy_to_cover", "buytocover"}:
        return "buy"
    if token in {"sell_long", "sell"}:
        return "sell"
    return token or "unknown"


def _live_cost_model_bucket(
    payload: Mapping[str, Any],
    *,
    symbol: str,
    side: str,
    session_regime: str,
    min_samples: int,
) -> tuple[Mapping[str, Any] | None, str | None]:
    rows = payload.get("by_symbol_side_session")
    if not isinstance(rows, list):
        return None, None
    symbol_token = str(symbol or "").strip().upper()
    side_token = _normalized_live_cost_side(side)
    session_token = str(session_regime or "").strip().lower()
    side_candidates = [side_token]
    if side_token == "sell_short":
        side_candidates.append("sell")
    elif side_token == "sell":
        side_candidates.append("sell_short")
    candidates: list[tuple[Mapping[str, Any], str]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("symbol") or "").strip().upper() != symbol_token:
            continue
        row_session = str(row.get("session_regime") or "").strip().lower()
        if row_session != session_token:
            continue
        sample_count = _finite_float(row.get("sample_count")) or 0.0
        if sample_count < max(1, int(min_samples)):
            continue
        if not bool(row.get("sufficient_samples")):
            continue
        row_side = _normalized_live_cost_side(str(row.get("side") or ""))
        if row_side == side_token:
            return row, f"LIVE_COST_MODEL:{symbol_token}:{row_side}:{session_token}"
        if row_side in side_candidates:
            candidates.append(
                (row, f"LIVE_COST_MODEL:{symbol_token}:{row_side}:{session_token}")
            )
        else:
            candidates.append(
                (row, f"LIVE_COST_MODEL:{symbol_token}:ANY:{session_token}")
            )
    if candidates:
        return candidates[0]
    return None, None


def _live_cost_model_thresholds(
    intent: OrderIntent,
    *,
    symbol_token: str,
    session_regime: str,
) -> dict[str, Any]:
    payload = _live_cost_model_payload()
    if payload is None or not _live_cost_model_usable(payload):
        return {}
    min_samples = int(
        max(
            1,
            float(
                get_env(
                    "AI_TRADING_PRETRADE_LIVE_COST_MODEL_MIN_SAMPLES",
                    5,
                    cast=float,
                )
            ),
        )
    )
    row, source = _live_cost_model_bucket(
        payload,
        symbol=symbol_token,
        side=intent.side,
        session_regime=session_regime,
        min_samples=min_samples,
    )
    if row is None:
        return {}
    spread_multiplier = max(
        1.0,
        float(
            get_env(
                "AI_TRADING_PRETRADE_LIVE_COST_MODEL_SPREAD_MULTIPLIER",
                1.25,
                cast=float,
            )
        ),
    )
    quote_age_multiplier = max(
        1.0,
        float(
            get_env(
                "AI_TRADING_PRETRADE_LIVE_COST_MODEL_QUOTE_AGE_MULTIPLIER",
                1.25,
                cast=float,
            )
        ),
    )
    max_total_cost_bps = max(
        0.0,
        float(
            get_env(
                "AI_TRADING_PRETRADE_LIVE_COST_MODEL_MAX_TOTAL_COST_BPS",
                25.0,
                cast=float,
            )
        ),
    )
    spread_basis = _finite_float(row.get("p90_spread_bps"))
    if spread_basis is None:
        spread_basis = _finite_float(row.get("mean_spread_bps"))
    quote_age_basis = _finite_float(row.get("p90_quote_age_ms"))
    if quote_age_basis is None:
        quote_age_basis = _finite_float(row.get("mean_quote_age_ms"))
    p90_total_cost_bps = _finite_float(row.get("p90_total_cost_bps"))
    mean_total_cost_bps = _finite_float(row.get("mean_total_cost_bps"))
    return {
        "source": source,
        "sample_count": int(_finite_float(row.get("sample_count")) or 0.0),
        "max_spread_bps": (
            float(spread_basis) * spread_multiplier
            if spread_basis is not None and spread_basis >= 0.0
            else None
        ),
        "spread_basis_bps": spread_basis,
        "max_quote_age_ms": (
            float(quote_age_basis) * quote_age_multiplier
            if quote_age_basis is not None and quote_age_basis >= 0.0
            else None
        ),
        "quote_age_basis_ms": quote_age_basis,
        "p90_adverse_slippage_bps": _finite_float(row.get("p90_adverse_slippage_bps")),
        "mean_slippage_bps": _finite_float(row.get("mean_slippage_bps")),
        "p90_total_cost_bps": p90_total_cost_bps,
        "mean_total_cost_bps": mean_total_cost_bps,
        "max_total_cost_bps": max_total_cost_bps,
    }


def _parse_hhmm(token: str) -> dt_time | None:
    parts = token.strip().split(":")
    if len(parts) != 2:
        return None
    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except (TypeError, ValueError):
        return None
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return None
    return dt_time(hour=hour, minute=minute)


def _parse_et_window(token: str) -> tuple[dt_time, dt_time] | None:
    parts = token.strip().split("-")
    if len(parts) != 2:
        return None
    start = _parse_hhmm(parts[0])
    end = _parse_hhmm(parts[1])
    if start is None or end is None:
        return None
    return start, end


def _time_in_window(now_val: dt_time, start: dt_time, end: dt_time) -> bool:
    if start <= end:
        return start <= now_val <= end
    # Overnight window, e.g. 23:30-00:30.
    return now_val >= start or now_val <= end


def _event_blackout_window_match(intent: OrderIntent) -> str | None:
    if not bool(get_env("AI_TRADING_EVENT_RISK_BLACKOUT_ENABLED", True, cast=bool)):
        return None

    ts = intent.bar_ts if intent.bar_ts.tzinfo else intent.bar_ts.replace(tzinfo=UTC)
    ts_et = ts.astimezone(ZoneInfo("America/New_York"))
    now_tod = ts_et.time()

    min_after_open = max(
        0,
        int(get_env("AI_TRADING_EVENT_BLACKOUT_MIN_AFTER_OPEN", 0, cast=int)),
    )
    min_before_close = max(
        0,
        int(get_env("AI_TRADING_EVENT_BLACKOUT_MIN_BEFORE_CLOSE", 0, cast=int)),
    )
    if min_after_open > 0:
        cutoff_open_minutes = (9 * 60 + 30) + min_after_open
        if (now_tod.hour * 60 + now_tod.minute) <= cutoff_open_minutes:
            return "open_auction_blackout"
    if min_before_close > 0:
        cutoff_close_minutes = (16 * 60) - min_before_close
        if (now_tod.hour * 60 + now_tod.minute) >= cutoff_close_minutes:
            return "close_auction_blackout"

    custom_windows_raw = str(get_env("AI_TRADING_EVENT_BLACKOUT_WINDOWS_ET", "", cast=str) or "").strip()
    if not custom_windows_raw:
        return None
    for token in custom_windows_raw.split(","):
        parsed = _parse_et_window(token)
        if parsed is None:
            continue
        if _time_in_window(now_tod, parsed[0], parsed[1]):
            return f"custom_window:{token.strip()}"
    return None


def _intent_session_regime(intent: OrderIntent) -> str:
    explicit = str(intent.session_regime or "").strip().lower()
    if explicit in {"opening", "midday", "closing", "offhours"}:
        return explicit
    ts = intent.bar_ts if intent.bar_ts.tzinfo else intent.bar_ts.replace(tzinfo=UTC)
    ts_et = ts.astimezone(ZoneInfo("America/New_York"))
    weekday = int(ts_et.weekday())
    minute_of_day = int((ts_et.hour * 60) + ts_et.minute)
    open_minute = (9 * 60) + 30
    close_minute = 16 * 60
    if weekday >= 5 or minute_of_day < open_minute or minute_of_day >= close_minute:
        return "offhours"
    minutes_from_open = minute_of_day - open_minute
    minutes_to_close = close_minute - minute_of_day
    if minutes_from_open < 45:
        return "opening"
    if minutes_to_close <= 45:
        return "closing"
    return "midday"


def _effective_expected_slippage_bps(intent: OrderIntent) -> tuple[float | None, str | None]:
    if intent.expected_slippage_bps is not None:
        try:
            return float(intent.expected_slippage_bps), "explicit"
        except (TypeError, ValueError):
            return None, None
    symbol_token = str(intent.symbol or "").strip().upper()
    if symbol_token:
        live_thresholds = _live_cost_model_thresholds(
            intent,
            symbol_token=symbol_token,
            session_regime=_intent_session_regime(intent),
        )
        live_slippage = _finite_float(live_thresholds.get("p90_adverse_slippage_bps"))
        live_slippage_source = "p90_adverse_slippage_bps"
        if live_slippage is None:
            live_slippage = _finite_float(live_thresholds.get("mean_slippage_bps"))
            live_slippage_source = "mean_slippage_bps"
        if live_slippage is not None and live_slippage >= 0.0:
            source = str(live_thresholds.get("source") or "LIVE_COST_MODEL")
            return float(live_slippage), f"{source}:{live_slippage_source}"
    reference = None
    if intent.mid is not None:
        try:
            reference = float(intent.mid)
        except (TypeError, ValueError):
            reference = None
    if (reference is None or reference <= 0.0) and intent.last_price is not None:
        try:
            reference = float(intent.last_price)
        except (TypeError, ValueError):
            reference = None
    if (
        intent.spread is not None
        and reference is not None
        and reference > 0.0
    ):
        try:
            spread_bps = max(0.0, (float(intent.spread) / float(reference)) * 10_000.0)
        except (TypeError, ValueError, ZeroDivisionError):
            return None, None
        return spread_bps, "derived_from_spread"
    return None, None


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return None
    return parsed


def _intent_reference_price(intent: OrderIntent) -> float | None:
    for value in (intent.mid, intent.last_price, intent.limit_price):
        parsed = _finite_float(value)
        if parsed is not None and parsed > 0.0:
            return parsed
    bid = _finite_float(intent.bid)
    ask = _finite_float(intent.ask)
    if bid is not None and ask is not None and bid > 0.0 and ask > 0.0:
        return (bid + ask) / 2.0
    return None


def _intent_spread_bps(intent: OrderIntent) -> tuple[float | None, str | None]:
    reference = _intent_reference_price(intent)
    bid = _finite_float(intent.bid)
    ask = _finite_float(intent.ask)
    if bid is not None and ask is not None and bid > 0.0 and ask >= bid:
        mid = (bid + ask) / 2.0
        if mid > 0.0:
            return ((ask - bid) / mid) * 10_000.0, "bid_ask"
    spread = _finite_float(intent.spread)
    if spread is not None and spread >= 0.0 and reference is not None and reference > 0.0:
        return (spread / reference) * 10_000.0, "spread"
    return None, None


def _intent_reduces_position(intent: OrderIntent, ledger: Any) -> bool:
    current_qty = _ledger_position_qty(ledger, intent.symbol)
    if current_qty is None:
        return False
    side = str(intent.side or "").strip().lower()
    order_qty = abs(int(intent.qty))
    if order_qty <= 0:
        return False
    if current_qty > 0.0 and side in {"sell", "sell_long"}:
        return order_qty <= abs(current_qty)
    if current_qty < 0.0 and side in {"buy", "cover", "buy_to_cover"}:
        return order_qty <= abs(current_qty)
    return False


def _lookup_threshold(
    values: dict[str, float],
    *,
    symbol: str,
    session_regime: str,
    liquidity_bucket: str,
) -> tuple[float | None, str | None]:
    keys = (
        f"{symbol}:{session_regime.upper()}",
        f"{symbol}:{session_regime.lower()}",
        symbol,
        f"BUCKET:{liquidity_bucket}",
        liquidity_bucket,
    )
    for key in keys:
        value = values.get(str(key).upper())
        if value is not None and value > 0.0:
            return float(value), str(key).upper()
    return None, None


def _execution_quality_gate(
    intent: OrderIntent,
    *,
    cfg: Any,
    ledger: Any,
    effective_quote_age_ms: float | None,
    quote_max_age_ms: int,
) -> tuple[bool, str, dict[str, Any]] | None:
    enabled = get_env(
        "AI_TRADING_PRETRADE_EXECUTION_QUALITY_GATE_ENABLED",
        True,
        cast=bool,
    )
    if not bool(enabled) or _intent_reduces_position(intent, ledger):
        return None

    symbol_token = str(intent.symbol or "").strip().upper()
    session_regime = _intent_session_regime(intent).upper()
    liquidity_bucket = str(intent.liquidity_bucket or "UNKNOWN").strip().upper() or "UNKNOWN"
    live_cost_thresholds = _live_cost_model_thresholds(
        intent,
        symbol_token=symbol_token,
        session_regime=session_regime,
    )
    live_total_cost_bps = _finite_float(live_cost_thresholds.get("p90_total_cost_bps"))
    if live_total_cost_bps is None:
        live_total_cost_bps = _finite_float(live_cost_thresholds.get("mean_total_cost_bps"))
    live_max_total_cost_bps = _finite_float(
        live_cost_thresholds.get("max_total_cost_bps")
    )
    if (
        live_total_cost_bps is not None
        and live_max_total_cost_bps is not None
        and live_max_total_cost_bps > 0.0
        and live_total_cost_bps > live_max_total_cost_bps
    ):
        return (
            False,
            "EXECUTION_QUALITY_LIVE_COST_BLOCK",
            {
                "symbol": symbol_token,
                "session_regime": session_regime.lower(),
                "liquidity_bucket": liquidity_bucket,
                "p90_total_cost_bps": (
                    _finite_float(live_cost_thresholds.get("p90_total_cost_bps"))
                ),
                "mean_total_cost_bps": (
                    _finite_float(live_cost_thresholds.get("mean_total_cost_bps"))
                ),
                "max_total_cost_bps": float(live_max_total_cost_bps),
                "threshold_source": live_cost_thresholds.get("source"),
                "sample_count": live_cost_thresholds.get("sample_count"),
                "action": "block",
                "block_reason": "live_cost_too_high",
            },
        )

    global_max_spread_bps = _finite_float(
        _cfg_value(
            cfg,
            field="execution_max_spread_bps",
            env_keys=("AI_TRADING_EXEC_MAX_SPREAD_BPS", "AI_TRADING_POLICY_EXEC_MAX_SPREAD_BPS"),
            default=0.0,
            cast=float,
        )
    )
    spread_thresholds = _json_env_dict(
        "AI_TRADING_EXEC_MAX_SPREAD_BPS_BY_SYMBOL",
        {},
    )
    spread_thresholds.update(
        _json_env_dict("AI_TRADING_EXEC_MAX_SPREAD_BPS_BY_SYMBOL_SESSION", {})
    )
    spread_thresholds.update(
        _json_env_dict("AI_TRADING_EXEC_MAX_SPREAD_BPS_BY_BUCKET", {})
    )
    max_spread_bps, spread_threshold_source = _lookup_threshold(
        spread_thresholds,
        symbol=symbol_token,
        session_regime=session_regime,
        liquidity_bucket=liquidity_bucket,
    )
    if max_spread_bps is None and global_max_spread_bps is not None and global_max_spread_bps > 0.0:
        max_spread_bps = float(global_max_spread_bps)
        spread_threshold_source = "GLOBAL"
    live_max_spread_bps = _finite_float(live_cost_thresholds.get("max_spread_bps"))
    if live_max_spread_bps is not None and live_max_spread_bps > 0.0:
        if max_spread_bps is None or live_max_spread_bps < max_spread_bps:
            max_spread_bps = float(live_max_spread_bps)
            spread_threshold_source = str(live_cost_thresholds.get("source") or "LIVE_COST_MODEL")

    spread_bps, spread_source = _intent_spread_bps(intent)
    if (
        spread_bps is not None
        and max_spread_bps is not None
        and spread_bps > max_spread_bps
    ):
        return (
            False,
            "EXECUTION_QUALITY_SPREAD_BLOCK",
            {
                "symbol": symbol_token,
                "session_regime": session_regime.lower(),
                "liquidity_bucket": liquidity_bucket,
                "spread_bps": round(float(spread_bps), 6),
                "spread_source": spread_source,
                "max_spread_bps": float(max_spread_bps),
                "live_spread_basis_bps": live_cost_thresholds.get("spread_basis_bps"),
                "live_cost_sample_count": live_cost_thresholds.get("sample_count"),
                "threshold_source": spread_threshold_source,
                "bid": intent.bid,
                "ask": intent.ask,
                "quote_source": intent.submit_quote_source,
                "action": "block",
                "block_reason": "spread_bps_too_wide",
            },
        )

    quote_age_thresholds = _json_env_dict(
        "AI_TRADING_EXEC_MAX_QUOTE_AGE_MS_BY_SYMBOL",
        {},
    )
    quote_age_thresholds.update(
        _json_env_dict("AI_TRADING_EXEC_MAX_QUOTE_AGE_MS_BY_SYMBOL_SESSION", {})
    )
    quote_age_thresholds.update(
        _json_env_dict("AI_TRADING_EXEC_MAX_QUOTE_AGE_MS_BY_BUCKET", {})
    )
    max_quote_age_ms, quote_age_threshold_source = _lookup_threshold(
        quote_age_thresholds,
        symbol=symbol_token,
        session_regime=session_regime,
        liquidity_bucket=liquidity_bucket,
    )
    if max_quote_age_ms is None and quote_max_age_ms > 0:
        max_quote_age_ms = float(quote_max_age_ms)
        quote_age_threshold_source = "GLOBAL"
    live_max_quote_age_ms = _finite_float(
        live_cost_thresholds.get("max_quote_age_ms")
    )
    if live_max_quote_age_ms is not None and live_max_quote_age_ms > 0.0:
        if max_quote_age_ms is None or live_max_quote_age_ms < max_quote_age_ms:
            max_quote_age_ms = float(live_max_quote_age_ms)
            quote_age_threshold_source = str(
                live_cost_thresholds.get("source") or "LIVE_COST_MODEL"
            )
    if (
        effective_quote_age_ms is not None
        and max_quote_age_ms is not None
        and max_quote_age_ms > 0.0
        and effective_quote_age_ms > max_quote_age_ms
    ):
        return (
            False,
            "EXECUTION_QUALITY_STALE_QUOTE_BLOCK",
            {
                "symbol": symbol_token,
                "session_regime": session_regime.lower(),
                "liquidity_bucket": liquidity_bucket,
                "quote_age_ms": round(float(effective_quote_age_ms), 3),
                "max_quote_age_ms": float(max_quote_age_ms),
                "live_quote_age_basis_ms": live_cost_thresholds.get("quote_age_basis_ms"),
                "live_cost_sample_count": live_cost_thresholds.get("sample_count"),
                "threshold_source": quote_age_threshold_source,
                "quote_source": intent.submit_quote_source,
                "quote_ts": (
                    _coerce_utc_datetime(intent.quote_ts).isoformat()
                    if _coerce_utc_datetime(intent.quote_ts) is not None
                    else None
                ),
                "action": "block",
                "block_reason": "quote_age_too_stale",
            },
        )
    return None


def _ledger_sector_notional(ledger: Any, sector: str | None) -> float | None:
    if ledger is None or not sector:
        return None
    method_names = (
        "sector_notional",
        "get_sector_notional",
        "current_sector_notional",
    )
    for method_name in method_names:
        method = getattr(ledger, method_name, None)
        if callable(method):
            try:
                return float(method(str(sector)))
            except (TypeError, ValueError):
                continue
    sector_map = getattr(ledger, "sector_notional_map", None)
    if isinstance(sector_map, dict):
        try:
            return float(sector_map.get(str(sector), 0.0))
        except (TypeError, ValueError):
            return None
    return None


def _ledger_factor_exposure(ledger: Any, factor_name: str | None) -> float | None:
    if ledger is None or not factor_name:
        return None
    method_names = (
        "factor_exposure",
        "get_factor_exposure",
        "current_factor_exposure",
    )
    for method_name in method_names:
        method = getattr(ledger, method_name, None)
        if callable(method):
            try:
                return float(method(str(factor_name)))
            except (TypeError, ValueError):
                continue
    factor_map = getattr(ledger, "factor_exposure_map", None)
    if isinstance(factor_map, dict):
        try:
            return float(factor_map.get(str(factor_name), 0.0))
        except (TypeError, ValueError):
            return None
    return None


def _ledger_metric_value(ledger: Any, metric: str) -> float | None:
    if ledger is None:
        return None
    metric_key = str(metric).strip().lower()
    metric_method_names: dict[str, tuple[str, ...]] = {
        "var": ("intraday_var", "var_95", "get_intraday_var"),
        "cvar": ("intraday_cvar", "cvar_95", "get_intraday_cvar"),
        "drawdown": ("current_drawdown", "intraday_drawdown", "get_current_drawdown"),
        "daily_loss_pct": ("daily_loss_pct", "get_daily_loss_pct"),
        "daily_loss_abs": ("daily_loss_abs", "daily_loss_amount", "get_daily_loss_abs"),
        "execution_drift_bps": ("execution_drift_bps", "get_execution_drift_bps"),
        "reject_rate_pct": ("reject_rate_pct", "get_reject_rate_pct"),
    }
    for method_name in metric_method_names.get(metric_key, ()):
        method = getattr(ledger, method_name, None)
        if callable(method):
            try:
                return float(method())
            except (TypeError, ValueError):
                continue
    attr_names: dict[str, tuple[str, ...]] = {
        "var": ("intraday_var", "var_95"),
        "cvar": ("intraday_cvar", "cvar_95"),
        "drawdown": ("current_drawdown", "intraday_drawdown"),
        "daily_loss_pct": ("daily_loss_pct",),
        "daily_loss_abs": ("daily_loss_abs", "daily_loss_amount"),
        "execution_drift_bps": ("execution_drift_bps",),
        "reject_rate_pct": ("reject_rate_pct",),
    }
    for attr_name in attr_names.get(metric_key, ()):
        value = getattr(ledger, attr_name, None)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def validate_pretrade(
    intent: OrderIntent,
    *,
    cfg: Any,
    ledger: Any,
    rate_limiter: SlidingWindowRateLimiter,
) -> tuple[bool, str, dict[str, Any]]:
    """Validate pre-trade controls and return allow/deny decision."""

    max_order_dollars = float(
        _cfg_value(
            cfg,
            field="max_order_dollars",
            env_keys=("MAX_ORDER_DOLLARS", "AI_TRADING_MAX_ORDER_DOLLARS"),
            default=0.0,
            cast=float,
        )
    )
    max_order_shares = int(
        _cfg_value(
            cfg,
            field="max_order_shares",
            env_keys=("MAX_ORDER_SHARES", "AI_TRADING_MAX_ORDER_SHARES"),
            default=0,
            cast=int,
        )
    )
    collar_pct = float(
        _cfg_value(
            cfg,
            field="price_collar_pct",
            env_keys=("PRICE_COLLAR_PCT", "AI_TRADING_PRICE_COLLAR_PCT"),
            default=0.03,
            cast=float,
        )
    )
    max_symbol_notional = float(
        _cfg_value(
            cfg,
            field="max_symbol_notional",
            env_keys=("MAX_SYMBOL_NOTIONAL", "AI_TRADING_MAX_SYMBOL_NOTIONAL"),
            default=0.0,
            cast=float,
        )
    )
    max_gross_notional = float(
        _cfg_value(
            cfg,
            field="max_gross_notional",
            env_keys=("MAX_GROSS_NOTIONAL", "AI_TRADING_MAX_GROSS_NOTIONAL"),
            default=0.0,
            cast=float,
        )
    )
    max_sector_notional = float(
        _cfg_value(
            cfg,
            field="max_sector_notional",
            env_keys=("AI_TRADING_MAX_SECTOR_NOTIONAL",),
            default=0.0,
            cast=float,
        )
    )
    max_factor_exposure = float(
        _cfg_value(
            cfg,
            field="max_factor_exposure",
            env_keys=("AI_TRADING_MAX_FACTOR_EXPOSURE",),
            default=0.0,
            cast=float,
        )
    )
    intraday_var_limit = float(
        _cfg_value(
            cfg,
            field="intraday_var_limit",
            env_keys=("AI_TRADING_INTRADAY_VAR_LIMIT",),
            default=0.0,
            cast=float,
        )
    )
    intraday_cvar_limit = float(
        _cfg_value(
            cfg,
            field="intraday_cvar_limit",
            env_keys=("AI_TRADING_INTRADAY_CVAR_LIMIT",),
            default=0.0,
            cast=float,
        )
    )
    intraday_drawdown_limit = float(
        _cfg_value(
            cfg,
            field="intraday_drawdown_limit",
            env_keys=("AI_TRADING_INTRADAY_DRAWDOWN_LIMIT",),
            default=0.0,
            cast=float,
        )
    )
    quote_max_age_ms = int(
        _cfg_value(
            cfg,
            field="quote_max_age_ms",
            env_keys=("QUOTE_MAX_AGE_MS",),
            default=0,
            cast=int,
        )
    )
    if quote_max_age_ms <= 0:
        quote_max_age_ms = int(
            _cfg_value(
                cfg,
                field="min_quote_freshness_ms",
                env_keys=("TRADING__MIN_QUOTE_FRESHNESS_MS",),
                default=0,
                cast=int,
            )
        )
    rth_only = _coerce_bool(getattr(cfg, "rth_only", get_env("RTH_ONLY", True, cast=bool)))
    allow_extended = _coerce_bool(
        getattr(cfg, "allow_extended", get_env("ALLOW_EXTENDED", False, cast=bool))
    )

    if bool(intent.kill_switch_active):
        return (
            False,
            "KILL_SWITCH_BLOCK",
            {
                "reason": str(intent.kill_switch_reason or "kill_switch"),
            },
        )

    if intent.broker_ready is False:
        details: dict[str, Any] = {
            "reason": str(intent.broker_ready_reason or "broker_not_ready"),
        }
        if intent.broker_cooldown_remaining_sec is not None:
            details["auth_forbidden_retry_after_sec"] = round(
                float(intent.broker_cooldown_remaining_sec),
                3,
            )
        return False, str(intent.broker_ready_reason or "BROKER_NOT_READY_BLOCK"), details

    try:
        qty_value = int(intent.qty)
    except (TypeError, ValueError):
        return False, "INVALID_QTY_BLOCK", {"qty": intent.qty}
    if qty_value <= 0:
        return False, "INVALID_QTY_BLOCK", {"qty": qty_value}

    runtime_decay_decision = _runtime_decay_pretrade_gate(intent, ledger=ledger, cfg=cfg)
    if runtime_decay_decision is not None:
        return runtime_decay_decision

    symbol_universe_decision = _symbol_universe_pretrade_gate(intent, ledger=ledger)
    if symbol_universe_decision is not None:
        return symbol_universe_decision

    if (rth_only or not allow_extended) and not _intent_regular_hours_open(intent):
        return (
            False,
            "MARKET_HOURS_BLOCK",
            {
                "bar_ts": intent.bar_ts.isoformat(),
                "rth_only": bool(rth_only),
                "allow_extended": bool(allow_extended),
            },
        )

    if intent.bid is not None and intent.ask is not None:
        try:
            bid = float(intent.bid)
            ask = float(intent.ask)
        except (TypeError, ValueError):
            bid = 0.0
            ask = 0.0
        if bid <= 0.0 or ask <= 0.0 or ask < bid:
            return (
                False,
                "QUOTE_SANITY_BLOCK",
                {
                    "bid": intent.bid,
                    "ask": intent.ask,
                    "source": intent.submit_quote_source,
                },
            )

    effective_quote_age_ms = _effective_quote_age_ms(intent)
    if quote_max_age_ms > 0 and effective_quote_age_ms is not None:
        if effective_quote_age_ms > float(quote_max_age_ms):
            return (
                False,
                "STALE_QUOTE_BLOCK",
                {
                    "quote_age_ms": round(float(effective_quote_age_ms), 3),
                    "max_quote_age_ms": int(quote_max_age_ms),
                    "quote_source": intent.submit_quote_source,
                    "quote_ts": (
                        _coerce_utc_datetime(intent.quote_ts).isoformat()
                        if _coerce_utc_datetime(intent.quote_ts) is not None
                        else None
                    ),
                },
            )

    live_cost_artifact_decision = _live_cost_artifact_pretrade_gate(intent, cfg=cfg)
    if live_cost_artifact_decision is not None:
        return live_cost_artifact_decision

    execution_quality_decision = _execution_quality_gate(
        intent,
        cfg=cfg,
        ledger=ledger,
        effective_quote_age_ms=effective_quote_age_ms,
        quote_max_age_ms=quote_max_age_ms,
    )
    if execution_quality_decision is not None:
        return execution_quality_decision

    require_realtime_nbbo = bool(intent.require_realtime_nbbo)
    if (
        require_realtime_nbbo
        and bool(intent.opening_trade)
        and str(intent.submit_quote_source or "").strip().lower() != "broker_nbbo"
    ):
        return (
            False,
            "NBBO_REQUIRED_OPENING_SKIP",
            {
                "required": True,
                "opening_trade": True,
                "quote_source": intent.submit_quote_source,
                "quote_age_ms": effective_quote_age_ms,
            },
        )

    qty_abs = qty_value
    notional_abs = abs(float(intent.notional))
    if (max_order_shares > 0 and qty_abs > max_order_shares) or (
        max_order_dollars > 0 and notional_abs > max_order_dollars
    ):
        return False, "ORDER_SIZE_BLOCK", {"qty": qty_abs, "notional": notional_abs}

    reference = intent.mid if intent.mid and intent.mid > 0 else intent.last_price
    current_symbol_notional: float | None = None
    projected_symbol_notional: float | None = None
    if reference is not None and reference > 0:
        side_token = str(intent.side).strip().lower()
        signed_qty = qty_abs if side_token in {"buy", "buy_to_cover", "cover"} else -qty_abs
        current_symbol_qty = _ledger_position_qty(ledger, intent.symbol)
        if current_symbol_qty is not None:
            projected_symbol_notional = abs((current_symbol_qty + signed_qty) * float(reference))
            if max_symbol_notional > 0 and projected_symbol_notional > max_symbol_notional:
                return (
                    False,
                    "SYMBOL_NOTIONAL_BLOCK",
                    {
                        "symbol": str(intent.symbol).upper(),
                        "projected_symbol_notional": projected_symbol_notional,
                        "max_symbol_notional": max_symbol_notional,
                    },
                )
            current_gross_notional = _ledger_gross_notional(ledger)
            if current_gross_notional is not None and max_gross_notional > 0:
                current_symbol_notional = abs(current_symbol_qty * float(reference))
                projected_gross_notional = max(
                    0.0,
                    float(current_gross_notional) - current_symbol_notional + projected_symbol_notional,
                )
                if projected_gross_notional > max_gross_notional:
                    return (
                        False,
                        "GROSS_NOTIONAL_BLOCK",
                        {
                            "projected_gross_notional": projected_gross_notional,
                            "max_gross_notional": max_gross_notional,
                            "symbol": str(intent.symbol).upper(),
                        },
                    )

    if max_sector_notional > 0 and reference and reference > 0 and intent.sector:
        sector_notional = _ledger_sector_notional(ledger, intent.sector)
        if sector_notional is not None:
            if current_symbol_notional is not None and projected_symbol_notional is not None:
                projected_sector_notional = max(
                    0.0,
                    float(sector_notional) - current_symbol_notional + projected_symbol_notional,
                )
            elif str(intent.side).strip().lower() in {"buy", "buy_to_cover", "cover", "sell_short", "short"}:
                projected_sector_notional = float(sector_notional) + notional_abs
            else:
                projected_sector_notional = max(0.0, float(sector_notional) - notional_abs)
            if projected_sector_notional > max_sector_notional:
                return (
                    False,
                    "SECTOR_CONCENTRATION_BLOCK",
                    {
                        "sector": str(intent.sector),
                        "projected_sector_notional": projected_sector_notional,
                        "max_sector_notional": max_sector_notional,
                    },
                )

    if max_factor_exposure > 0 and intent.factor_name and intent.factor_exposure is not None:
        current_factor = _ledger_factor_exposure(ledger, intent.factor_name)
        projected_factor = float(intent.factor_exposure)
        if current_factor is not None:
            projected_factor += float(current_factor)
        if abs(projected_factor) > max_factor_exposure:
            return (
                False,
                "FACTOR_CONCENTRATION_BLOCK",
                {
                    "factor": str(intent.factor_name),
                    "projected_factor_exposure": projected_factor,
                    "max_factor_exposure": max_factor_exposure,
                },
            )

    if intraday_var_limit > 0:
        current_var = _ledger_metric_value(ledger, "var")
        if current_var is not None and current_var > intraday_var_limit:
            return (
                False,
                "INTRADAY_VAR_BLOCK",
                {"intraday_var": current_var, "max_intraday_var": intraday_var_limit},
            )

    if intraday_cvar_limit > 0:
        current_cvar = _ledger_metric_value(ledger, "cvar")
        if current_cvar is not None and current_cvar > intraday_cvar_limit:
            return (
                False,
                "INTRADAY_CVAR_BLOCK",
                {"intraday_cvar": current_cvar, "max_intraday_cvar": intraday_cvar_limit},
            )

    if intraday_drawdown_limit > 0:
        current_drawdown = _ledger_metric_value(ledger, "drawdown")
        if current_drawdown is not None and current_drawdown > intraday_drawdown_limit:
            return (
                False,
                "INTRADAY_DRAWDOWN_BLOCK",
                {
                    "intraday_drawdown": current_drawdown,
                    "max_intraday_drawdown": intraday_drawdown_limit,
                },
            )

    daily_risk_budget_enabled = bool(get_env("AI_TRADING_DAILY_RISK_BUDGET_ENABLED", True, cast=bool))
    daily_loss_limit_pct = float(
        _cfg_value(
            cfg,
            field="daily_loss_limit_pct",
            env_keys=("AI_TRADING_DAILY_LOSS_LIMIT_PCT",),
            default=0.0,
            cast=float,
        )
    )
    if daily_loss_limit_pct <= 0:
        daily_loss_limit_pct = float(
            _cfg_value(
                cfg,
                field="daily_loss_limit",
                env_keys=("AI_TRADING_DAILY_LOSS_LIMIT",),
                default=0.0,
                cast=float,
            )
        )
    daily_loss_limit_abs = float(
        _cfg_value(
            cfg,
            field="daily_loss_limit_abs",
            env_keys=("AI_TRADING_DAILY_LOSS_LIMIT_ABS",),
            default=0.0,
            cast=float,
        )
    )
    if daily_risk_budget_enabled and daily_loss_limit_pct > 0:
        current_daily_loss_pct = _ledger_metric_value(ledger, "daily_loss_pct")
        if current_daily_loss_pct is not None and current_daily_loss_pct >= daily_loss_limit_pct:
            return (
                False,
                "DAILY_RISK_BUDGET_BLOCK",
                {
                    "daily_loss_pct": current_daily_loss_pct,
                    "max_daily_loss_pct": daily_loss_limit_pct,
                },
            )
    if daily_risk_budget_enabled and daily_loss_limit_abs > 0:
        current_daily_loss_abs = _ledger_metric_value(ledger, "daily_loss_abs")
        if current_daily_loss_abs is not None and current_daily_loss_abs >= daily_loss_limit_abs:
            return (
                False,
                "DAILY_RISK_BUDGET_BLOCK",
                {
                    "daily_loss_abs": current_daily_loss_abs,
                    "max_daily_loss_abs": daily_loss_limit_abs,
                },
            )

    if bool(get_env("AI_TRADING_EVENT_RISK_BLACKOUT_ENABLED", True, cast=bool)):
        if bool(intent.event_risk):
            return (
                False,
                "EVENT_RISK_BLACKOUT_BLOCK",
                {
                    "event_type": str(intent.event_type or "event_risk"),
                },
            )
        blackout_reason = _event_blackout_window_match(intent)
        if blackout_reason is not None:
            return (
                False,
                "EVENT_RISK_BLACKOUT_BLOCK",
                {"reason": blackout_reason},
            )

    derisk_enabled = bool(get_env("AI_TRADING_DERISK_ON_DATA_DEGRADED", True, cast=bool))
    derisk_mode = str(get_env("AI_TRADING_DERISK_MODE", "block", cast=str) or "block").strip().lower()
    data_degraded = bool(get_env("AI_TRADING_DATA_DEGRADED", False, cast=bool))
    quote_quality_bad = intent.quote_quality_ok is False
    if derisk_enabled and (data_degraded or quote_quality_bad) and derisk_mode == "block":
        return (
            False,
            "DERISK_DATA_QUALITY_BLOCK",
            {
                "data_degraded": data_degraded,
                "quote_quality_ok": intent.quote_quality_ok,
                "mode": derisk_mode,
            },
        )

    derisk_slo_enabled = bool(get_env("AI_TRADING_DERISK_ON_SLO_BREACH_ENABLED", True, cast=bool))
    derisk_slo_mode = str(get_env("AI_TRADING_DERISK_SLO_MODE", "block", cast=str) or "block").strip().lower()
    max_reject_rate_pct = float(get_env("AI_TRADING_DERISK_SLO_MAX_REJECT_RATE_PCT", 5.0, cast=float))
    max_execution_drift_bps = float(get_env("AI_TRADING_DERISK_SLO_MAX_EXEC_DRIFT_BPS", 35.0, cast=float))
    reject_rate_pct = (
        float(intent.reject_rate_pct)
        if intent.reject_rate_pct is not None
        else _ledger_metric_value(ledger, "reject_rate_pct")
    )
    execution_drift_bps = (
        float(intent.execution_drift_bps)
        if intent.execution_drift_bps is not None
        else _ledger_metric_value(ledger, "execution_drift_bps")
    )
    reject_breached = (
        reject_rate_pct is not None
        and max_reject_rate_pct > 0
        and reject_rate_pct >= max_reject_rate_pct
    )
    drift_breached = (
        execution_drift_bps is not None
        and max_execution_drift_bps > 0
        and execution_drift_bps >= max_execution_drift_bps
    )
    if derisk_slo_enabled and derisk_slo_mode == "block" and (reject_breached or drift_breached):
        return (
            False,
            "DERISK_SLO_BREACH_BLOCK",
            {
                "reject_rate_pct": reject_rate_pct,
                "execution_drift_bps": execution_drift_bps,
                "max_reject_rate_pct": max_reject_rate_pct,
                "max_execution_drift_bps": max_execution_drift_bps,
            },
        )

    bucket_default = {"THIN": 45.0, "NORMAL": 30.0, "THICK": 20.0}
    symbol_default = {
        "BA": 8.0,
        "QCOM": 10.0,
        "ADBE": 10.0,
        "ABBV": 12.0,
        "V": 12.0,
        "PFE": 10.0,
        "EXC": 10.0,
    }
    symbol_session_default = {
        "BA:MIDDAY": 6.0,
        "QCOM:MIDDAY": 8.0,
        "ADBE:MIDDAY": 8.0,
        "ABBV:MIDDAY": 10.0,
        "V:OPENING": 10.0,
        "ADBE:OPENING": 8.0,
        "PFE:OPENING": 8.0,
        "EXC:OPENING": 8.0,
    }
    bucket_map = _json_env_dict("AI_TRADING_EXEC_SLIPPAGE_CEILING_BPS_BY_BUCKET", bucket_default)
    symbol_map = _json_env_dict("AI_TRADING_EXEC_SLIPPAGE_CEILING_BPS_BY_SYMBOL", symbol_default)
    symbol_session_map = _json_env_dict(
        "AI_TRADING_EXEC_SLIPPAGE_CEILING_BPS_BY_SYMBOL_SESSION",
        symbol_session_default,
    )
    liquidity_bucket = str(intent.liquidity_bucket or "NORMAL").upper()
    symbol_token = str(intent.symbol).upper()
    session_regime = _intent_session_regime(intent)
    symbol_session_ceiling = symbol_session_map.get(f"{symbol_token}:{session_regime.upper()}")
    symbol_ceiling = symbol_map.get(symbol_token)
    bucket_ceiling = bucket_map.get(liquidity_bucket, bucket_map.get("NORMAL", 30.0))
    slippage_ceiling_bps = (
        symbol_session_ceiling
        if symbol_session_ceiling is not None
        else symbol_ceiling
        if symbol_ceiling is not None
        else bucket_ceiling
    )
    effective_expected_slippage_bps, slippage_source = _effective_expected_slippage_bps(intent)
    if (
        effective_expected_slippage_bps is not None
        and slippage_ceiling_bps is not None
        and float(effective_expected_slippage_bps) > float(slippage_ceiling_bps)
    ):
        return (
            False,
            "SLIPPAGE_CEILING_BLOCK",
            {
                "symbol": symbol_token,
                "bucket": liquidity_bucket,
                "session_regime": session_regime,
                "expected_slippage_bps": float(effective_expected_slippage_bps),
                "expected_slippage_source": slippage_source,
                "ceiling_bps": float(slippage_ceiling_bps),
            },
        )

    max_adv_participation = float(get_env("AI_TRADING_EXEC_MAX_PARTICIPATION_PCT_ADV", 0.05, cast=float))
    max_minute_participation = float(
        get_env("AI_TRADING_EXEC_MAX_PARTICIPATION_PCT_MINUTE", 0.20, cast=float)
    )
    if bool(get_env("AI_TRADING_EXEC_CAPACITY_AWARE_THROTTLE_ENABLED", True, cast=bool)):
        bucket_multiplier = _json_env_dict(
            "AI_TRADING_EXEC_CAPACITY_BUCKET_MULTIPLIER",
            {"THIN": 0.50, "NORMAL": 1.0, "THICK": 1.15},
        )
        liquidity_multiplier = float(bucket_multiplier.get(liquidity_bucket, 1.0))
        liquidity_multiplier = max(0.05, min(liquidity_multiplier, 1.5))
        spread_soft_bps = float(get_env("AI_TRADING_EXEC_CAPACITY_SPREAD_SOFT_BPS", 12.0, cast=float))
        spread_hard_bps = float(get_env("AI_TRADING_EXEC_CAPACITY_SPREAD_HARD_BPS", 30.0, cast=float))
        spread_multiplier = 1.0
        if (
            intent.spread is not None
            and reference is not None
            and reference > 0
            and spread_hard_bps > spread_soft_bps
        ):
            spread_bps = max(0.0, float(intent.spread) / float(reference) * 10_000.0)
            if spread_bps > spread_soft_bps:
                if spread_bps >= spread_hard_bps:
                    spread_multiplier = 0.25
                else:
                    progress = (spread_bps - spread_soft_bps) / (spread_hard_bps - spread_soft_bps)
                    spread_multiplier = max(0.25, 1.0 - progress * 0.75)
        effective_multiplier = max(0.05, min(liquidity_multiplier * spread_multiplier, 1.0))
        max_adv_participation *= effective_multiplier
        max_minute_participation *= effective_multiplier
    if intent.avg_daily_volume is not None and intent.avg_daily_volume > 0 and max_adv_participation > 0:
        adv_limit = float(intent.avg_daily_volume) * max_adv_participation
        if qty_abs > adv_limit:
            return (
                False,
                "PARTICIPATION_CAP_BLOCK",
                {
                    "scope": "adv",
                    "qty": qty_abs,
                    "adv_limit_qty": adv_limit,
                    "max_participation_pct_adv": max_adv_participation,
                },
            )
    if intent.minute_volume is not None and intent.minute_volume > 0 and max_minute_participation > 0:
        minute_limit = float(intent.minute_volume) * max_minute_participation
        if qty_abs > minute_limit:
            return (
                False,
                "PARTICIPATION_CAP_BLOCK",
                {
                    "scope": "minute",
                    "qty": qty_abs,
                    "minute_limit_qty": minute_limit,
                    "max_participation_pct_minute": max_minute_participation,
                },
            )

    tca_gate_enabled = bool(get_env("AI_TRADING_EXEC_TCA_GATE_ENABLED", True, cast=bool))
    max_expected_tca_bps = float(get_env("AI_TRADING_EXEC_TCA_MAX_EXPECTED_BPS", 35.0, cast=float))
    min_fill_quality = float(get_env("AI_TRADING_EXEC_MIN_FILL_QUALITY_SCORE", 0.50, cast=float))
    if tca_gate_enabled and intent.expected_tca_bps is not None:
        if float(intent.expected_tca_bps) > max_expected_tca_bps:
            return (
                False,
                "TCA_GATE_BLOCK",
                {
                    "expected_tca_bps": float(intent.expected_tca_bps),
                    "max_expected_tca_bps": max_expected_tca_bps,
                },
            )
    if tca_gate_enabled and intent.fill_quality_score is not None:
        if float(intent.fill_quality_score) < min_fill_quality:
            return (
                False,
                "FILL_QUALITY_GATE_BLOCK",
                {
                    "fill_quality_score": float(intent.fill_quality_score),
                    "min_fill_quality_score": min_fill_quality,
                },
            )

    if intent.limit_price is not None and reference and reference > 0:
        deviation = abs(float(intent.limit_price) - float(reference)) / float(reference)
        if deviation > max(0.0, collar_pct):
            return False, "PRICE_COLLAR_BLOCK", {"reference": reference, "limit_price": intent.limit_price, "deviation": deviation}

    if ledger is not None and intent.client_order_id:
        seen_fn = getattr(ledger, "seen_client_order_id", None)
        if callable(seen_fn) and bool(seen_fn(intent.client_order_id)):
            return False, "DUPLICATE_ORDER_BLOCK", {"client_order_id": intent.client_order_id}

    fingerprint = (
        str(intent.symbol).upper(),
        str(intent.side).lower(),
        qty_abs,
        intent.bar_ts.isoformat(),
    )
    fingerprints = _ledger_fingerprints(ledger)
    if fingerprint in fingerprints:
        return False, "DUPLICATE_ORDER_BLOCK", {"fingerprint": fingerprint}

    allow_and_record = getattr(rate_limiter, "allow_and_record_order", None)
    if callable(allow_and_record):
        allowed, reason, details = allow_and_record(intent.symbol, intent.bar_ts)
    else:
        allowed, reason, details = rate_limiter.allow_order(intent.symbol, intent.bar_ts)
        if allowed and not rate_limiter.cancel_rate_ok():
            return False, "RATE_THROTTLE_BLOCK", {"scope": "cancel"}
        if allowed:
            rate_limiter.record_order(intent.symbol, intent.bar_ts)
    if not allowed:
        return False, reason or "RATE_THROTTLE_BLOCK", details
    if ledger is not None:
        fingerprints.add(fingerprint)
    return True, "OK", {}


def safe_validate_pretrade(
    intent: OrderIntent,
    *,
    cfg: Any,
    ledger: Any,
    rate_limiter: SlidingWindowRateLimiter,
) -> tuple[bool, str, dict[str, Any]]:
    """Validate pre-trade controls with fail-closed handling on unexpected errors."""

    fail_closed = bool(get_env("AI_TRADING_PRETRADE_FAIL_CLOSED", True, cast=bool))
    try:
        return validate_pretrade(
            intent,
            cfg=cfg,
            ledger=ledger,
            rate_limiter=rate_limiter,
        )
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        details = {
            "error": str(exc),
            "symbol": str(intent.symbol).upper(),
            "client_order_id": intent.client_order_id,
            "fail_closed": fail_closed,
        }
        if fail_closed:
            return False, "PRETRADE_VALIDATION_ERROR", details
        return True, "PRETRADE_VALIDATION_FAIL_OPEN", details
