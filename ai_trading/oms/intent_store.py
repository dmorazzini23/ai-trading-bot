"""Durable OMS intent store with transactional idempotency semantics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any

from ai_trading.config.management import get_env
from ai_trading.logging import get_logger

logger = get_logger(__name__)


_TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"FILLED", "CANCELED", "CANCELLED", "REJECTED", "CLOSED"}
)


@dataclass(frozen=True, slots=True)
class IntentRecord:
    """Persistent intent row."""

    intent_id: str
    idempotency_key: str
    symbol: str
    side: str
    quantity: float
    status: str
    broker_order_id: str | None
    decision_ts: str
    created_at: str
    updated_at: str
    submit_attempts: int
    strategy_id: str | None
    expected_edge_bps: float | None
    regime: str | None
    last_error: str | None
    metadata_json: str | None


@dataclass(frozen=True, slots=True)
class FillRecord:
    """Persistent fill row tied to an intent."""

    fill_id: int
    intent_id: str
    fill_qty: float
    fill_price: float | None
    fee: float
    liquidity_flag: str | None
    fill_ts: str
    created_at: str


class IntentStore:
    """SQLite-backed intent store used for exactly-once order intent tracking."""

    def __init__(self, path: str | None = None) -> None:
        resolved_path = path or get_env(
            "AI_TRADING_OMS_INTENT_STORE_PATH",
            "runtime/oms_intents.db",
        )
        self._path = Path(str(resolved_path)).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self._path),
            timeout=30.0,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._bootstrap()

    @property
    def path(self) -> Path:
        """Return intent store path."""

        return self._path

    @staticmethod
    def _utcnow_iso() -> str:
        return datetime.now(UTC).isoformat()

    def _bootstrap(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS intents (
                    intent_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    status TEXT NOT NULL,
                    broker_order_id TEXT NULL,
                    decision_ts TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    submit_attempts INTEGER NOT NULL DEFAULT 0,
                    strategy_id TEXT NULL,
                    expected_edge_bps REAL NULL,
                    regime TEXT NULL,
                    last_error TEXT NULL,
                    metadata_json TEXT NULL
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS intent_fills (
                    fill_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    intent_id TEXT NOT NULL,
                    fill_qty REAL NOT NULL,
                    fill_price REAL NULL,
                    fee REAL NOT NULL DEFAULT 0.0,
                    liquidity_flag TEXT NULL,
                    fill_ts TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(intent_id) REFERENCES intents(intent_id)
                );
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_intents_status ON intents(status);"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_fills_intent_id ON intent_fills(intent_id);"
            )

    @staticmethod
    def _row_to_intent(row: sqlite3.Row) -> IntentRecord:
        return IntentRecord(
            intent_id=str(row["intent_id"]),
            idempotency_key=str(row["idempotency_key"]),
            symbol=str(row["symbol"]),
            side=str(row["side"]),
            quantity=float(row["quantity"]),
            status=str(row["status"]),
            broker_order_id=row["broker_order_id"],
            decision_ts=str(row["decision_ts"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            submit_attempts=int(row["submit_attempts"]),
            strategy_id=row["strategy_id"],
            expected_edge_bps=(
                float(row["expected_edge_bps"])
                if row["expected_edge_bps"] is not None
                else None
            ),
            regime=row["regime"],
            last_error=row["last_error"],
            metadata_json=row["metadata_json"],
        )

    @staticmethod
    def _row_to_fill(row: sqlite3.Row) -> FillRecord:
        return FillRecord(
            fill_id=int(row["fill_id"]),
            intent_id=str(row["intent_id"]),
            fill_qty=float(row["fill_qty"]),
            fill_price=(
                float(row["fill_price"]) if row["fill_price"] is not None else None
            ),
            fee=float(row["fee"]),
            liquidity_flag=row["liquidity_flag"],
            fill_ts=str(row["fill_ts"]),
            created_at=str(row["created_at"]),
        )

    def create_intent(
        self,
        *,
        intent_id: str,
        idempotency_key: str,
        symbol: str,
        side: str,
        quantity: float,
        decision_ts: str | None = None,
        strategy_id: str | None = None,
        expected_edge_bps: float | None = None,
        regime: str | None = None,
        metadata: dict[str, Any] | None = None,
        status: str = "PENDING_SUBMIT",
    ) -> tuple[IntentRecord, bool]:
        """Insert or return existing intent keyed by idempotency key."""

        now = self._utcnow_iso()
        record_decision_ts = decision_ts or now
        meta_payload = json.dumps(metadata or {}, sort_keys=True)
        normalized_status = str(status).strip().upper() or "PENDING_SUBMIT"
        with self._lock:
            try:
                with self._conn:
                    self._conn.execute(
                        """
                        INSERT INTO intents (
                            intent_id,
                            idempotency_key,
                            symbol,
                            side,
                            quantity,
                            status,
                            broker_order_id,
                            decision_ts,
                            created_at,
                            updated_at,
                            submit_attempts,
                            strategy_id,
                            expected_edge_bps,
                            regime,
                            last_error,
                            metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, 0, ?, ?, ?, NULL, ?)
                        """,
                        (
                            intent_id,
                            idempotency_key,
                            symbol.upper(),
                            side.lower(),
                            float(quantity),
                            normalized_status,
                            record_decision_ts,
                            now,
                            now,
                            strategy_id,
                            expected_edge_bps,
                            regime,
                            meta_payload,
                        ),
                    )
                row = self._conn.execute(
                    "SELECT * FROM intents WHERE intent_id = ?",
                    (intent_id,),
                ).fetchone()
                if row is None:  # pragma: no cover - defensive
                    raise RuntimeError("intent_insert_missing_row")
                return (self._row_to_intent(row), True)
            except sqlite3.IntegrityError:
                existing = self._conn.execute(
                    "SELECT * FROM intents WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                if existing is None:  # pragma: no cover - defensive
                    raise
                return (self._row_to_intent(existing), False)

    def get_intent(self, intent_id: str) -> IntentRecord | None:
        """Return intent by ID."""

        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM intents WHERE intent_id = ?",
                (intent_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_intent(row)

    def get_intent_by_key(self, idempotency_key: str) -> IntentRecord | None:
        """Return intent by idempotency key."""

        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM intents WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_intent(row)

    def get_open_intents(self) -> list[IntentRecord]:
        """Return all non-terminal intents."""

        placeholders = ",".join(["?"] * len(_TERMINAL_STATUSES))
        query = (
            "SELECT * FROM intents WHERE status NOT IN "
            f"({placeholders}) ORDER BY updated_at ASC"
        )
        with self._lock:
            rows = self._conn.execute(query, tuple(sorted(_TERMINAL_STATUSES))).fetchall()
        return [self._row_to_intent(row) for row in rows]

    def claim_for_submit(
        self,
        intent_id: str,
        *,
        stale_after_seconds: int = 180,
    ) -> bool:
        """Acquire submit lease for an intent exactly once.

        Returns ``True`` only when the caller has an active claim and should
        proceed to submit a broker order.
        """

        stale_after = max(1, int(stale_after_seconds))
        now = datetime.now(UTC)
        now_iso = now.isoformat()
        stale_cutoff_iso = (now - timedelta(seconds=stale_after)).isoformat()
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                UPDATE intents
                   SET status = 'SUBMITTING',
                       submit_attempts = submit_attempts + 1,
                       updated_at = ?,
                       last_error = NULL
                 WHERE intent_id = ?
                   AND (
                        status = 'PENDING_SUBMIT'
                        OR (status = 'SUBMITTING' AND updated_at <= ?)
                   )
                """,
                (now_iso, intent_id, stale_cutoff_iso),
            )
        return cur.rowcount > 0

    def mark_submitted(self, intent_id: str, broker_order_id: str) -> None:
        """Mark intent as submitted with broker order id."""

        now = self._utcnow_iso()
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE intents
                   SET status = 'SUBMITTED',
                       broker_order_id = ?,
                       updated_at = ?
                 WHERE intent_id = ?
                """,
                (broker_order_id, now, intent_id),
            )

    def record_submit_error(self, intent_id: str, error: str) -> None:
        """Record submit failure while keeping intent retryable."""

        now = self._utcnow_iso()
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE intents
                   SET status = 'PENDING_SUBMIT',
                       last_error = ?,
                       updated_at = ?
                 WHERE intent_id = ?
                """,
                (error[:500], now, intent_id),
            )

    def record_fill(
        self,
        intent_id: str,
        *,
        fill_qty: float,
        fill_price: float | None,
        fee: float = 0.0,
        liquidity_flag: str | None = None,
        fill_ts: str | None = None,
    ) -> None:
        """Persist a fill observation for an intent."""

        now = self._utcnow_iso()
        ts = fill_ts or now
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO intent_fills (
                    intent_id,
                    fill_qty,
                    fill_price,
                    fee,
                    liquidity_flag,
                    fill_ts,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    intent_id,
                    float(fill_qty),
                    float(fill_price) if fill_price is not None else None,
                    float(fee),
                    liquidity_flag,
                    ts,
                    now,
                ),
            )
            self._conn.execute(
                """
                UPDATE intents
                   SET status = CASE
                       WHEN status IN ('FILLED', 'CLOSED') THEN status
                       ELSE 'PARTIALLY_FILLED'
                   END,
                       updated_at = ?
                 WHERE intent_id = ?
                """,
                (now, intent_id),
            )

    def list_fills(self, intent_id: str) -> list[FillRecord]:
        """Return fills for intent in insertion order."""

        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM intent_fills WHERE intent_id = ? ORDER BY fill_id ASC",
                (intent_id,),
            ).fetchall()
        return [self._row_to_fill(row) for row in rows]

    def close_intent(
        self,
        intent_id: str,
        *,
        final_status: str,
        last_error: str | None = None,
    ) -> None:
        """Close intent with terminal status."""

        normalized = str(final_status).strip().upper() or "CLOSED"
        now = self._utcnow_iso()
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE intents
                   SET status = ?,
                       last_error = ?,
                       updated_at = ?
                 WHERE intent_id = ?
                """,
                (
                    normalized,
                    (last_error[:500] if last_error else None),
                    now,
                    intent_id,
                ),
            )

    def close(self) -> None:
        """Close DB connection."""

        with self._lock:
            try:
                self._conn.close()
            except Exception:
                logger.debug("INTENT_STORE_CLOSE_FAILED", exc_info=True)

