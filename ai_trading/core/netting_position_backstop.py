"""Position backstop for the netting cycle: stop loss, maximum hold, position cap.

These are safety backstops, not strategy rules. They act on positions the
netting cycle already holds and only through the normal netting submission:

* stop loss: exit when a position is ``POSITION_STOP_LOSS_PCT`` against the
  broker's average entry price (shorts symmetric);
* maximum hold: exit after ``POSITION_MAX_HOLD_SESSIONS`` NYSE session closes;
* position cap: block new openings once the occupied slots reach
  ``cfg.max_portfolio_positions``; never touches reductions, adds or flips.

A fired backstop locks the symbol until the next session opens. While locked
the exit keeps being forced; once flat, openings are blocked. Episodes (entry
times) and locks live in a bot-owned JSON store written atomically only when
they change. Evaluation is pure (:func:`evaluate`); logging and the store sit
in :class:`PositionBackstop`.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from zoneinfo import ZoneInfo

from ai_trading.runtime.atomic_io import atomic_write_text
from ai_trading.utils import market_calendar

POSITION_STOP_LOSS_PCT = 0.05
POSITION_MAX_HOLD_SESSIONS = 5
POSITION_BACKSTOP_PRICE_SANITY_PCT = 0.40
# A broker-sync snapshot older than this cannot arm the stop loss.
POSITION_BACKSTOP_BASIS_MAX_AGE_SECONDS = 600.0
# A locked position still held this many bars after arming is reported stalled.
POSITION_BACKSTOP_STALL_BARS = 2
POSITION_BACKSTOP_STATE_PATH = "runtime/position_backstop_state.json"

GATE_STOP_LOSS_EXIT = "POSITION_STOP_LOSS_EXIT"
GATE_MAX_HOLD_EXIT = "POSITION_MAX_HOLD_EXIT"
GATE_REENTRY_BLOCK = "POSITION_BACKSTOP_REENTRY_BLOCK"
GATE_CAP_BLOCK = "POSITION_CAP_BLOCK"
GATE_PROTECTIVE_EXIT_OVERRIDE = "PROTECTIVE_EXIT_OVERRIDE"

REASON_STOP_LOSS = "stop_loss"
REASON_MAX_HOLD = "max_hold"
_EXIT_GATES = {REASON_STOP_LOSS: GATE_STOP_LOSS_EXIT, REASON_MAX_HOLD: GATE_MAX_HOLD_EXIT}

SOURCE_OBSERVED = "observed"
SOURCE_FIRST_SEEN = "first_seen"

# Floating-point slack so a return of exactly -5% fires (95/100 - 1 is not exact).
_THRESHOLD_EPS = 1e-9
_ET = ZoneInfo("America/New_York")
_CALENDAR_SCAN_DAYS = 15


@dataclass(frozen=True, slots=True)
class PositionBasis:
    """Broker average entry price for a position the cycle holds."""

    avg_entry_price: float
    qty: float


@dataclass(frozen=True, slots=True)
class Episode:
    opened_at: datetime
    opened_at_source: str
    side: str


@dataclass(frozen=True, slots=True)
class Lock:
    until: datetime
    reason: str
    armed_at: datetime
    entry_price: float | None
    trigger_price: float | None
    side: str


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Outcome of :func:`evaluate` for one symbol on one bar."""

    action: str = "none"  # "none" | "exit" | "reentry_block"
    reason: str | None = None
    gate: str | None = None
    breach: bool = False
    new_lock: Lock | None = None
    loss_pct: float | None = None
    sessions_held: int | None = None
    price_suspect: bool = False


def _side_of(qty: float) -> str:
    return "long" if qty > 0 else "short"


def _positive_int(value: Any) -> int | None:
    """``value`` as a positive int; anything else (missing, zero, bad) disables the cap."""

    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


@lru_cache(maxsize=1024)
def _session_bounds(day: date) -> tuple[datetime, datetime] | None:
    if not market_calendar.is_trading_day(day):
        return None
    try:
        session = market_calendar.session_info(day)
    except (ValueError, RuntimeError):
        return None
    return _as_utc(session.start_utc), _as_utc(session.end_utc)


def sessions_held(opened_at: datetime, now: datetime, *, limit: int | None = None) -> int:
    """Count NYSE session closes in ``(opened_at, now]``; stops counting at ``limit``."""

    start, end = _as_utc(opened_at), _as_utc(now)
    if end <= start:
        return 0
    day = start.astimezone(_ET).date()
    last_day = end.astimezone(_ET).date()
    count = 0
    while day <= last_day:
        bounds = _session_bounds(day)
        if bounds is not None and start < bounds[1] <= end:
            count += 1
            if limit is not None and count >= limit:
                return count
        day += timedelta(days=1)
    return count


def next_session_open(now: datetime) -> datetime:
    """Open of the first NYSE session that starts after ``now``."""

    current = _as_utc(now)
    day = current.astimezone(_ET).date()
    for offset in range(_CALENDAR_SCAN_DAYS):
        bounds = _session_bounds(day + timedelta(days=offset))
        if bounds is not None and bounds[0] > current:
            return bounds[0]
    return current + timedelta(days=1)


def evaluate(
    *,
    current_shares: int,
    delta_shares: int,
    price: float,
    now: datetime,
    basis: PositionBasis | None,
    episode: Episode | None,
    lock: Lock | None,
    stop_loss_pct: float | None = None,
    max_hold_sessions: int | None = None,
    price_sanity_pct: float | None = None,
    sessions_held_func: Callable[..., int] = sessions_held,
    next_open_func: Callable[[datetime], datetime] = next_session_open,
) -> Evaluation:
    """Decide the backstop action for one symbol on one new bar.

    ``lock`` must already be an active (unexpired) lock. A held position with
    an active lock keeps being exited; a flat one cannot reopen. Otherwise the
    stop loss (needs ``basis``) and the maximum hold (needs ``episode``) are
    checked; the stop loss wins when both breach. Thresholds default to the
    module constants, read at call time.
    """

    stop_loss_pct = POSITION_STOP_LOSS_PCT if stop_loss_pct is None else stop_loss_pct
    max_hold_sessions = POSITION_MAX_HOLD_SESSIONS if max_hold_sessions is None else max_hold_sessions
    price_sanity_pct = POSITION_BACKSTOP_PRICE_SANITY_PCT if price_sanity_pct is None else price_sanity_pct
    if lock is not None:
        if current_shares != 0:
            return Evaluation(action="exit", reason=lock.reason, gate=_EXIT_GATES.get(lock.reason))
        if delta_shares != 0:
            return Evaluation(action="reentry_block", reason=lock.reason, gate=GATE_REENTRY_BLOCK)
        return Evaluation()
    if current_shares == 0:
        return Evaluation()

    sign = 1.0 if current_shares > 0 else -1.0
    loss_pct: float | None = None
    price_suspect = False
    reason: str | None = None
    if basis is not None and math.isfinite(price) and price > 0:
        move = price / basis.avg_entry_price - 1.0
        if abs(move) > price_sanity_pct:
            price_suspect = True
        else:
            loss_pct = move * sign
            if loss_pct <= -stop_loss_pct + _THRESHOLD_EPS:
                reason = REASON_STOP_LOSS
    held: int | None = None
    if episode is not None:
        held = int(sessions_held_func(episode.opened_at, now, limit=max_hold_sessions))
        if reason is None and held >= max_hold_sessions:
            reason = REASON_MAX_HOLD
    if reason is None:
        return Evaluation(loss_pct=loss_pct, sessions_held=held, price_suspect=price_suspect)
    new_lock = Lock(
        until=next_open_func(now),
        reason=reason,
        armed_at=_as_utc(now),
        entry_price=basis.avg_entry_price if basis is not None else None,
        trigger_price=float(price),
        side=_side_of(current_shares),
    )
    return Evaluation(
        action="exit",
        reason=reason,
        gate=_EXIT_GATES[reason],
        breach=True,
        new_lock=new_lock,
        loss_pct=loss_pct,
        sessions_held=held,
        price_suspect=price_suspect,
    )


# ---- durable store ------------------------------------------------------


def _parse_ts(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    return _as_utc(parsed)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


@dataclass(slots=True)
class BackstopStore:
    episodes: dict[str, Episode] = field(default_factory=dict)
    locks: dict[str, Lock] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Any) -> BackstopStore:
        if not isinstance(payload, Mapping):
            raise ValueError("payload is not an object")
        episodes_raw = payload.get("episodes", {})
        locks_raw = payload.get("locks", {})
        if not isinstance(episodes_raw, Mapping) or not isinstance(locks_raw, Mapping):
            raise ValueError("episodes/locks are not objects")
        store = cls()
        for symbol, row in episodes_raw.items():
            store.episodes[str(symbol).upper()] = Episode(
                opened_at=_parse_ts(row["opened_at"]),
                opened_at_source=str(row["opened_at_source"]),
                side=str(row["side"]),
            )
        for symbol, row in locks_raw.items():
            store.locks[str(symbol).upper()] = Lock(
                until=_parse_ts(row["until"]),
                reason=str(row["reason"]),
                armed_at=_parse_ts(row["armed_at"]),
                entry_price=_optional_float(row.get("entry_price")),
                trigger_price=_optional_float(row.get("trigger_price")),
                side=str(row["side"]),
            )
        return store

    def to_payload(self) -> dict[str, Any]:
        return {
            "episodes": {
                symbol: {
                    "opened_at": episode.opened_at.isoformat(),
                    "opened_at_source": episode.opened_at_source,
                    "side": episode.side,
                }
                for symbol, episode in sorted(self.episodes.items())
            },
            "locks": {
                symbol: {
                    "until": lock.until.isoformat(),
                    "reason": lock.reason,
                    "armed_at": lock.armed_at.isoformat(),
                    "entry_price": lock.entry_price,
                    "trigger_price": lock.trigger_price,
                    "side": lock.side,
                }
                for symbol, lock in sorted(self.locks.items())
            },
        }

    def reconcile(
        self,
        positions: Mapping[str, float],
        now: datetime,
        *,
        previously_seen: Iterable[str] | None,
    ) -> bool:
        """Update episodes from the cycle's positions; return whether anything changed.

        ``previously_seen`` is the set of symbols held at the previous
        reconciliation in this process, or ``None`` before the first one.
        """

        seen = None if previously_seen is None else set(previously_seen)
        changed = False
        held = {symbol: qty for symbol, qty in positions.items() if qty != 0}
        for symbol in list(self.episodes):
            if symbol not in held:
                del self.episodes[symbol]
                changed = True
        for symbol, qty in held.items():
            side = _side_of(qty)
            episode = self.episodes.get(symbol)
            if episode is not None and episode.side == side:
                continue
            if episode is not None or (seen is not None and symbol not in seen):
                source = SOURCE_OBSERVED
            else:
                source = SOURCE_FIRST_SEEN
            self.episodes[symbol] = Episode(opened_at=_as_utc(now), opened_at_source=source, side=side)
            changed = True
        return changed


def load_store(path: Path) -> BackstopStore:
    """Read the store; raises ``OSError`` / ``ValueError`` / ``KeyError`` / ``TypeError``."""

    return BackstopStore.from_payload(json.loads(path.read_text(encoding="utf-8")))


def save_store(path: Path, store: BackstopStore) -> None:
    atomic_write_text(path, json.dumps(store.to_payload(), sort_keys=True, indent=1) + "\n")


# ---- per-process runtime ------------------------------------------------


@dataclass(frozen=True, slots=True)
class SymbolStep:
    """What :meth:`PositionBackstop.evaluate_symbol` asks the symbol loop to do."""

    action: str = "none"  # "none" | "exit" | "reentry_block"
    gate: str | None = None
    snapshot: dict[str, Any] | None = None


class PositionBackstop:
    """Store, per-cycle inputs and once-per-session log keys for one process."""

    def __init__(self, path: Path, logger: Any) -> None:
        self.path = path
        self.logger = logger
        self.store = BackstopStore()
        self._dirty = False
        self._logged: set[tuple[str, ...]] = set()
        self._previously_held: set[str] | None = None
        self._bars_since_arming: dict[str, tuple[datetime, Any, int]] = {}
        self._stall_logged: set[tuple[str, str]] = set()
        self._last_gates: dict[str, list[str]] = {}
        self.now = datetime.now(UTC)
        self.bases: dict[str, PositionBasis] = {}
        self.positions: dict[str, float] = {}
        self.cap_limit: int | None = None
        self.pending_symbols: set[str] = set()
        self.cycle_openings: set[str] = set()
        self._load()

    # -- persistence --
    def _load(self) -> None:
        try:
            self.store = load_store(self.path)
        except FileNotFoundError:
            self.logger.warning(
                "POSITION_BACKSTOP_STATE_LOAD_FAILED",
                extra={"path": str(self.path), "reason": "missing"},
            )
            self.store = BackstopStore()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.logger.warning(
                "POSITION_BACKSTOP_STATE_LOAD_FAILED",
                extra={"path": str(self.path), "reason": "corrupt", "error": str(exc)},
            )
            self.store = BackstopStore()

    def _save(self) -> None:
        if not self._dirty:
            return
        try:
            save_store(self.path, self.store)
        except OSError as exc:
            # Decisions keep the in-memory state; the write retries next change.
            self.logger.warning(
                "POSITION_BACKSTOP_STATE_WRITE_FAILED",
                extra={"path": str(self.path), "error": str(exc)},
            )
            return
        self._dirty = False

    def _once(self, *key: str) -> bool:
        session_key = (self.now.astimezone(_ET).date().isoformat(), *key)
        if session_key in self._logged:
            return False
        self._logged.add(session_key)
        return True

    # -- cycle --
    def begin_cycle(
        self,
        *,
        now: datetime,
        positions: Mapping[str, Any],
        bases: Mapping[str, PositionBasis],
        unarmed: Mapping[str, str],
        cap_limit: Any,
        pending_symbols: Iterable[str],
    ) -> None:
        self.now = _as_utc(now)
        self.positions = {}
        for raw_symbol, raw_qty in positions.items():
            try:
                qty = float(raw_qty or 0.0)
            except (TypeError, ValueError):
                continue
            if math.isfinite(qty) and qty != 0:
                self.positions[str(raw_symbol).strip().upper()] = qty
        self.bases = dict(bases)
        self.cap_limit = _positive_int(cap_limit)
        self.pending_symbols = {str(sym).strip().upper() for sym in pending_symbols if str(sym).strip()}
        self.cycle_openings = set()

        if self.store.reconcile(self.positions, self.now, previously_seen=self._previously_held):
            self._dirty = True
        self._previously_held = set(self.positions)
        for symbol, lock in sorted(self.store.locks.items()):
            if self.now >= lock.until:
                del self.store.locks[symbol]
                self._dirty = True
                self._bars_since_arming.pop(symbol, None)
                self.logger.info(
                    "POSITION_BACKSTOP_LOCK_EXPIRED",
                    extra={
                        "symbol": symbol,
                        "reason": lock.reason,
                        "lock_until": lock.until.isoformat(),
                        "still_held": symbol in self.positions,
                    },
                )
        self._save()
        for symbol, reason in sorted(unarmed.items()):
            if symbol in self.positions and self._once("unarmed", symbol, reason):
                self.logger.info(
                    "POSITION_BACKSTOP_UNARMED",
                    extra={"symbol": symbol, "reason": reason, "qty": self.positions[symbol]},
                )
        if self.cap_limit is not None and len(self.positions) > self.cap_limit and self._once("cap_exceeded"):
            self.logger.warning(
                "POSITION_CAP_EXCEEDED",
                extra={"positions": len(self.positions), "cap": self.cap_limit, "symbols": sorted(self.positions)},
            )

    def evaluate_symbol(
        self,
        *,
        symbol: str,
        current_shares: int,
        delta_shares: int,
        price: float,
        bar_ts: Any,
    ) -> SymbolStep:
        lock = self.store.locks.get(symbol)
        if lock is not None and self.now >= lock.until:
            lock = None
        result = evaluate(
            current_shares=int(current_shares),
            delta_shares=int(delta_shares),
            price=float(price),
            now=self.now,
            basis=self.bases.get(symbol),
            episode=self.store.episodes.get(symbol),
            lock=lock,
        )
        if result.price_suspect and self._once("price_suspect", symbol):
            basis = self.bases.get(symbol)
            self.logger.warning(
                "POSITION_BACKSTOP_PRICE_SUSPECT",
                extra={
                    "symbol": symbol,
                    "price": float(price),
                    "avg_entry": basis.avg_entry_price if basis is not None else None,
                    "bar_ts": str(bar_ts),
                    "sanity_pct": POSITION_BACKSTOP_PRICE_SANITY_PCT,
                },
            )
        if result.action == "none":
            return SymbolStep()
        if result.breach and result.new_lock is not None:
            lock = result.new_lock
            self.store.locks[symbol] = lock
            self._dirty = True
            self._save()
            self._bars_since_arming[symbol] = (lock.armed_at, bar_ts, 0)
            self._log_breach(symbol, int(current_shares), float(price), bar_ts, result, lock)
        assert lock is not None
        snapshot = {
            "action": result.action,
            "reason": lock.reason,
            "lock_until": lock.until.isoformat(),
            "armed_at": lock.armed_at.isoformat(),
            "entry_price": lock.entry_price,
            "trigger_price": lock.trigger_price,
            "side": lock.side,
            "breach": bool(result.breach),
        }
        if result.action == "exit" and not result.breach:
            self._track_stall(symbol, int(current_shares), lock, bar_ts)
        return SymbolStep(action=result.action, gate=result.gate, snapshot=snapshot)

    def _log_breach(
        self,
        symbol: str,
        current_shares: int,
        price: float,
        bar_ts: Any,
        result: Evaluation,
        lock: Lock,
    ) -> None:
        episode = self.store.episodes.get(symbol)
        extra: dict[str, Any] = {
            "symbol": symbol,
            "side": lock.side,
            "qty": int(current_shares),
            "avg_entry": lock.entry_price,
            "price": float(price),
            "bar_ts": str(bar_ts),
            "lock_until": lock.until.isoformat(),
            "opened_at": episode.opened_at.isoformat() if episode is not None else None,
            "source": episode.opened_at_source if episode is not None else None,
        }
        if result.reason == REASON_STOP_LOSS:
            extra.update({"loss_pct": result.loss_pct, "threshold": -POSITION_STOP_LOSS_PCT})
            self.logger.warning("POSITION_STOP_LOSS_BREACH", extra=extra)
        else:
            extra.update({"sessions_held": result.sessions_held, "threshold": POSITION_MAX_HOLD_SESSIONS})
            self.logger.warning("POSITION_MAX_HOLD_BREACH", extra=extra)

    def _track_stall(self, symbol: str, current_shares: int, lock: Lock, bar_ts: Any) -> None:
        armed_at, last_bar, bars = self._bars_since_arming.get(symbol, (lock.armed_at, None, 0))
        if armed_at != lock.armed_at:
            armed_at, last_bar, bars = lock.armed_at, None, 0
        if bar_ts != last_bar:
            bars += 1
        self._bars_since_arming[symbol] = (armed_at, bar_ts, bars)
        stall_key = (symbol, armed_at.isoformat())
        if bars >= POSITION_BACKSTOP_STALL_BARS and stall_key not in self._stall_logged:
            self._stall_logged.add(stall_key)
            self.logger.warning(
                "POSITION_BACKSTOP_EXIT_STALLED",
                extra={
                    "symbol": symbol,
                    "side": _side_of(current_shares),
                    "qty": int(current_shares),
                    "reason": lock.reason,
                    "armed_at": armed_at.isoformat(),
                    "bars_since_arming": bars,
                    "bar_ts": str(bar_ts),
                    "lock_until": lock.until.isoformat(),
                    "last_gates": list(self._last_gates.get(symbol, [])),
                },
            )

    def note_decision(self, symbol: str, gates: Iterable[Any]) -> None:
        if symbol in self.store.locks:
            self._last_gates[symbol] = [str(gate) for gate in gates]

    def cap_block(self, *, symbol: str, current_shares: int, delta_shares: int) -> dict[str, Any] | None:
        """Return the cap context when a new opening must be blocked, else ``None``."""

        if self.cap_limit is None or current_shares != 0 or delta_shares == 0:
            return None
        occupied = set(self.positions) | self.pending_symbols | self.cycle_openings
        if symbol in occupied or len(occupied) < self.cap_limit:
            return None
        context = {
            "cap": self.cap_limit,
            "occupied": len(occupied),
            "positions": len(self.positions),
            "pending": len(self.pending_symbols - set(self.positions)),
            "cycle_openings": len(self.cycle_openings),
        }
        if self._once("cap_block", symbol):
            self.logger.info("POSITION_CAP_BLOCK", extra={"symbol": symbol, **context})
        return context

    def note_opening_submitted(self, symbol: str) -> None:
        self.cycle_openings.add(symbol)


__all__ = [
    "BackstopStore",
    "Episode",
    "Evaluation",
    "GATE_CAP_BLOCK",
    "GATE_MAX_HOLD_EXIT",
    "GATE_PROTECTIVE_EXIT_OVERRIDE",
    "GATE_REENTRY_BLOCK",
    "GATE_STOP_LOSS_EXIT",
    "Lock",
    "POSITION_BACKSTOP_PRICE_SANITY_PCT",
    "POSITION_MAX_HOLD_SESSIONS",
    "POSITION_STOP_LOSS_PCT",
    "PositionBackstop",
    "PositionBasis",
    "SymbolStep",
    "evaluate",
    "load_store",
    "next_session_open",
    "save_store",
    "sessions_held",
]
