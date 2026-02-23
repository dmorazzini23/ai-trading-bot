"""Durable OMS intent store with transactional idempotency semantics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import threading
from typing import Any

from ai_trading.config.management import get_env
from ai_trading.logging import get_logger
from ai_trading.oms.statuses import (
    TERMINAL_INTENT_STATUSES,
    is_terminal_intent_status,
    normalize_intent_status,
)

logger = get_logger(__name__)

try:
    from sqlalchemy import (
        Column,
        Float,
        ForeignKey,
        Integer,
        MetaData,
        String,
        Table,
        Text,
        case,
        create_engine,
        insert,
        select,
        text,
        update,
    )
    from sqlalchemy.engine import Engine
    from sqlalchemy.exc import IntegrityError
    from sqlalchemy.orm import sessionmaker

    _SQLALCHEMY_AVAILABLE = True
    _SQLALCHEMY_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised in environments missing sqlalchemy
    _SQLALCHEMY_AVAILABLE = False
    _SQLALCHEMY_IMPORT_ERROR = exc
    MetaData = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]
    Engine = Any  # type: ignore[assignment,misc]
    sessionmaker = None  # type: ignore[assignment]
    IntegrityError = Exception  # type: ignore[assignment]


_TERMINAL_STATUSES: frozenset[str] = TERMINAL_INTENT_STATUSES


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


if _SQLALCHEMY_AVAILABLE:
    _METADATA = MetaData()
    _INTENTS_TABLE = Table(
        "intents",
        _METADATA,
        # Core identity and dedupe
        Column("intent_id", String(128), primary_key=True),
        Column("idempotency_key", String(128), nullable=False, unique=True, index=True),
        # Decision payload
        Column("symbol", String(32), nullable=False),
        Column("side", String(16), nullable=False),
        Column("quantity", Float, nullable=False),
        Column("status", String(32), nullable=False, index=True),
        Column("broker_order_id", String(128), nullable=True),
        Column("decision_ts", String(64), nullable=False),
        Column("created_at", String(64), nullable=False),
        Column("updated_at", String(64), nullable=False),
        Column("submit_attempts", Integer, nullable=False, default=0, server_default="0"),
        Column("strategy_id", String(128), nullable=True),
        Column("expected_edge_bps", Float, nullable=True),
        Column("regime", String(64), nullable=True),
        Column("last_error", Text, nullable=True),
        Column("metadata_json", Text, nullable=True),
    )
    _INTENT_FILLS_TABLE = Table(
        "intent_fills",
        _METADATA,
        Column("fill_id", Integer, primary_key=True, autoincrement=True),
        Column(
            "intent_id",
            String(128),
            ForeignKey("intents.intent_id"),
            nullable=False,
            index=True,
        ),
        Column("fill_qty", Float, nullable=False),
        Column("fill_price", Float, nullable=True),
        Column("fee", Float, nullable=False, default=0.0, server_default="0"),
        Column("liquidity_flag", String(32), nullable=True),
        Column("fill_ts", String(64), nullable=False),
        Column("created_at", String(64), nullable=False),
    )
else:  # pragma: no cover - fallback only
    _METADATA = None
    _INTENTS_TABLE = None
    _INTENT_FILLS_TABLE = None


def _normalize_database_url(
    *,
    path: str | None = None,
    url: str | None = None,
) -> tuple[str, Path]:
    raw_url = str(url or get_env("DATABASE_URL", "") or "").strip()
    if raw_url:
        if raw_url.startswith("postgres://"):
            raw_url = f"postgresql+psycopg://{raw_url[len('postgres://') :]}"
        elif raw_url.startswith("postgresql://") and "+" not in raw_url.split("://", 1)[0]:
            raw_url = f"postgresql+psycopg://{raw_url[len('postgresql://') :]}"
        if raw_url.startswith("sqlite:///"):
            sqlite_path = Path(raw_url[len("sqlite:///") :]).expanduser()
            return raw_url, sqlite_path
        return raw_url, Path("database-url")

    resolved_path_raw = path or get_env(
        "AI_TRADING_OMS_INTENT_STORE_PATH",
        "runtime/oms_intents.db",
    )
    resolved_path_text = str(resolved_path_raw).strip()
    if "://" in resolved_path_text:
        return resolved_path_text, Path("database-url")
    resolved_path = Path(resolved_path_text).expanduser()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{resolved_path}", resolved_path


class IntentStore:
    """SQLAlchemy-backed durable intent store used for exactly-once semantics."""

    def __init__(
        self,
        path: str | None = None,
        *,
        url: str | None = None,
    ) -> None:
        if not _SQLALCHEMY_AVAILABLE:
            detail = str(_SQLALCHEMY_IMPORT_ERROR) if _SQLALCHEMY_IMPORT_ERROR else "unknown"
            raise RuntimeError(
                f"SQLAlchemy backend unavailable for IntentStore: {detail}. "
                "Install SQLAlchemy>=2.0 and psycopg[binary] to enable durable intent storage."
            )

        database_url, locator_path = _normalize_database_url(path=path, url=url)
        self._database_url = database_url
        self._path = locator_path
        self._lock = threading.RLock()
        if database_url.startswith("sqlite:///"):
            self._path.parent.mkdir(parents=True, exist_ok=True)
        connect_args: dict[str, Any] = {}
        if database_url.startswith("sqlite:"):
            connect_args["check_same_thread"] = False
        self._engine: Engine = create_engine(
            database_url,
            future=True,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
        self._session_factory = sessionmaker(
            bind=self._engine,
            autoflush=False,
            expire_on_commit=False,
            future=True,
        )
        self._bootstrap()

    @property
    def path(self) -> Path:
        """Return storage locator path (for compatibility with existing logs/tests)."""

        return self._path

    @property
    def database_url(self) -> str:
        """Return SQLAlchemy database URL backing this store."""

        return self._database_url

    @staticmethod
    def _utcnow_iso() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _row_to_intent(row: Mapping[str, Any]) -> IntentRecord:
        return IntentRecord(
            intent_id=str(row["intent_id"]),
            idempotency_key=str(row["idempotency_key"]),
            symbol=str(row["symbol"]),
            side=str(row["side"]),
            quantity=float(row["quantity"]),
            status=str(row["status"]),
            broker_order_id=(str(row["broker_order_id"]) if row["broker_order_id"] else None),
            decision_ts=str(row["decision_ts"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            submit_attempts=int(row["submit_attempts"] or 0),
            strategy_id=(str(row["strategy_id"]) if row["strategy_id"] else None),
            expected_edge_bps=(
                float(row["expected_edge_bps"])
                if row["expected_edge_bps"] is not None
                else None
            ),
            regime=(str(row["regime"]) if row["regime"] else None),
            last_error=(str(row["last_error"]) if row["last_error"] else None),
            metadata_json=(str(row["metadata_json"]) if row["metadata_json"] else None),
        )

    @staticmethod
    def _row_to_fill(row: Mapping[str, Any]) -> FillRecord:
        return FillRecord(
            fill_id=int(row["fill_id"]),
            intent_id=str(row["intent_id"]),
            fill_qty=float(row["fill_qty"]),
            fill_price=(
                float(row["fill_price"]) if row["fill_price"] is not None else None
            ),
            fee=float(row["fee"] or 0.0),
            liquidity_flag=(str(row["liquidity_flag"]) if row["liquidity_flag"] else None),
            fill_ts=str(row["fill_ts"]),
            created_at=str(row["created_at"]),
        )

    def _bootstrap(self) -> None:
        assert _METADATA is not None
        with self._engine.begin() as conn:
            if self._database_url.startswith("sqlite:"):
                conn.execute(text("PRAGMA journal_mode=WAL;"))
                conn.execute(text("PRAGMA synchronous=NORMAL;"))
            _METADATA.create_all(conn)

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

        assert _INTENTS_TABLE is not None
        now = self._utcnow_iso()
        record_decision_ts = decision_ts or now
        meta_payload = json.dumps(metadata or {}, sort_keys=True)
        normalized_status = str(status).strip().upper() or "PENDING_SUBMIT"
        payload = {
            "intent_id": intent_id,
            "idempotency_key": idempotency_key,
            "symbol": symbol.upper(),
            "side": side.lower(),
            "quantity": float(quantity),
            "status": normalized_status,
            "broker_order_id": None,
            "decision_ts": record_decision_ts,
            "created_at": now,
            "updated_at": now,
            "submit_attempts": 0,
            "strategy_id": strategy_id,
            "expected_edge_bps": expected_edge_bps,
            "regime": regime,
            "last_error": None,
            "metadata_json": meta_payload,
        }
        with self._lock:
            try:
                with self._session_factory.begin() as session:
                    session.execute(insert(_INTENTS_TABLE).values(**payload))
            except IntegrityError:
                with self._session_factory() as session:
                    row = (
                        session.execute(
                            select(_INTENTS_TABLE).where(
                                _INTENTS_TABLE.c.idempotency_key == idempotency_key
                            )
                        )
                        .mappings()
                        .first()
                    )
                if row is None:  # pragma: no cover - defensive
                    raise
                return (self._row_to_intent(row), False)

            with self._session_factory() as session:
                row = (
                    session.execute(
                        select(_INTENTS_TABLE).where(_INTENTS_TABLE.c.intent_id == intent_id)
                    )
                    .mappings()
                    .first()
                )
            if row is None:  # pragma: no cover - defensive
                raise RuntimeError("intent_insert_missing_row")
            return (self._row_to_intent(row), True)

    def get_intent(self, intent_id: str) -> IntentRecord | None:
        """Return intent by ID."""

        assert _INTENTS_TABLE is not None
        with self._lock, self._session_factory() as session:
            row = (
                session.execute(
                    select(_INTENTS_TABLE).where(_INTENTS_TABLE.c.intent_id == intent_id)
                )
                .mappings()
                .first()
            )
        if row is None:
            return None
        return self._row_to_intent(row)

    def get_intent_by_key(self, idempotency_key: str) -> IntentRecord | None:
        """Return intent by idempotency key."""

        assert _INTENTS_TABLE is not None
        with self._lock, self._session_factory() as session:
            row = (
                session.execute(
                    select(_INTENTS_TABLE).where(
                        _INTENTS_TABLE.c.idempotency_key == idempotency_key
                    )
                )
                .mappings()
                .first()
            )
        if row is None:
            return None
        return self._row_to_intent(row)

    def get_open_intents(self) -> list[IntentRecord]:
        """Return all non-terminal intents."""

        assert _INTENTS_TABLE is not None
        ordered_terminal = tuple(sorted(_TERMINAL_STATUSES))
        stmt = (
            select(_INTENTS_TABLE)
            .where(_INTENTS_TABLE.c.status.not_in(ordered_terminal))
            .order_by(_INTENTS_TABLE.c.updated_at.asc())
        )
        with self._lock, self._session_factory() as session:
            rows = session.execute(stmt).mappings().all()
        return [self._row_to_intent(row) for row in rows]

    def claim_for_submit(
        self,
        intent_id: str,
        *,
        stale_after_seconds: int = 180,
    ) -> bool:
        """Acquire submit lease for an intent exactly once."""

        assert _INTENTS_TABLE is not None
        stale_after = max(1, int(stale_after_seconds))
        now = datetime.now(UTC)
        now_iso = now.isoformat()
        stale_cutoff_iso = (now - timedelta(seconds=stale_after)).isoformat()
        stmt = (
            update(_INTENTS_TABLE)
            .where(_INTENTS_TABLE.c.intent_id == intent_id)
            .where(
                (
                    _INTENTS_TABLE.c.status == "PENDING_SUBMIT"
                )
                | (
                    (_INTENTS_TABLE.c.status == "SUBMITTING")
                    & (_INTENTS_TABLE.c.updated_at <= stale_cutoff_iso)
                )
            )
            .values(
                status="SUBMITTING",
                submit_attempts=_INTENTS_TABLE.c.submit_attempts + 1,
                updated_at=now_iso,
                last_error=None,
            )
        )
        with self._lock, self._session_factory.begin() as session:
            result = session.execute(stmt)
        rowcount = int(result.rowcount or 0)
        return rowcount > 0

    def mark_submitted(self, intent_id: str, broker_order_id: str) -> None:
        """Mark intent as submitted with broker order id."""

        assert _INTENTS_TABLE is not None
        now = self._utcnow_iso()
        stmt = (
            update(_INTENTS_TABLE)
            .where(_INTENTS_TABLE.c.intent_id == intent_id)
            .values(
                status="SUBMITTED",
                broker_order_id=str(broker_order_id),
                updated_at=now,
            )
        )
        with self._lock, self._session_factory.begin() as session:
            session.execute(stmt)

    def record_submit_error(self, intent_id: str, error: str) -> None:
        """Record submit failure while keeping intent retryable."""

        assert _INTENTS_TABLE is not None
        now = self._utcnow_iso()
        stmt = (
            update(_INTENTS_TABLE)
            .where(_INTENTS_TABLE.c.intent_id == intent_id)
            .values(
                status="PENDING_SUBMIT",
                last_error=str(error)[:500],
                updated_at=now,
            )
        )
        with self._lock, self._session_factory.begin() as session:
            session.execute(stmt)

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

        assert _INTENTS_TABLE is not None
        assert _INTENT_FILLS_TABLE is not None
        now = self._utcnow_iso()
        ts = fill_ts or now
        fill_stmt = insert(_INTENT_FILLS_TABLE).values(
            intent_id=intent_id,
            fill_qty=float(fill_qty),
            fill_price=float(fill_price) if fill_price is not None else None,
            fee=float(fee),
            liquidity_flag=liquidity_flag,
            fill_ts=ts,
            created_at=now,
        )
        status_case = case(
            (_INTENTS_TABLE.c.status.in_(("FILLED", "CLOSED")), _INTENTS_TABLE.c.status),
            else_="PARTIALLY_FILLED",
        )
        update_stmt = (
            update(_INTENTS_TABLE)
            .where(_INTENTS_TABLE.c.intent_id == intent_id)
            .values(
                status=status_case,
                updated_at=now,
            )
        )
        with self._lock, self._session_factory.begin() as session:
            session.execute(fill_stmt)
            session.execute(update_stmt)

    def list_fills(self, intent_id: str) -> list[FillRecord]:
        """Return fills for intent in insertion order."""

        assert _INTENT_FILLS_TABLE is not None
        stmt = (
            select(_INTENT_FILLS_TABLE)
            .where(_INTENT_FILLS_TABLE.c.intent_id == intent_id)
            .order_by(_INTENT_FILLS_TABLE.c.fill_id.asc())
        )
        with self._lock, self._session_factory() as session:
            rows = session.execute(stmt).mappings().all()
        return [self._row_to_fill(row) for row in rows]

    def close_intent(
        self,
        intent_id: str,
        *,
        final_status: str,
        last_error: str | None = None,
    ) -> None:
        """Close intent with terminal status."""

        assert _INTENTS_TABLE is not None
        normalized = normalize_intent_status(final_status, default="CLOSED")
        if not is_terminal_intent_status(normalized):
            raise ValueError(f"close_intent requires terminal status, got: {normalized}")
        now = self._utcnow_iso()
        stmt = (
            update(_INTENTS_TABLE)
            .where(_INTENTS_TABLE.c.intent_id == intent_id)
            .values(
                status=normalized,
                last_error=(str(last_error)[:500] if last_error else None),
                updated_at=now,
            )
        )
        with self._lock, self._session_factory.begin() as session:
            session.execute(stmt)

    def close(self) -> None:
        """Dispose DB engine resources."""

        with self._lock:
            try:
                self._engine.dispose()
            except Exception:
                logger.debug("INTENT_STORE_CLOSE_FAILED", exc_info=True)
