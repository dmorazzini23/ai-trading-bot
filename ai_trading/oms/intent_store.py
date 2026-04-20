"""Durable OMS intent store with transactional idempotency semantics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import json
from pathlib import Path
import threading
from typing import Any, cast

from ai_trading.config.management import get_env
from ai_trading.logging import get_logger
from ai_trading.oms.engine_registry import resolve_shared_engine
from ai_trading.oms.lifecycle import (
    PENDING_SUBMIT_STATUS,
    SUBMITTING_STATUS,
    normalize_terminal_status,
    status_for_fill,
    status_for_submit_ack,
    status_for_submit_claim,
    status_for_submit_error,
    terminal_event_type,
)
from ai_trading.oms.statuses import (
    TERMINAL_INTENT_STATUSES,
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
    MetaData = None  # type: ignore[assignment,misc]
    Table = None  # type: ignore[assignment,misc]
    Engine = Any  # type: ignore[assignment,misc]
    sessionmaker = None  # type: ignore[assignment,misc]
    IntegrityError = Exception  # type: ignore[assignment,misc]


_TERMINAL_STATUSES: frozenset[str] = TERMINAL_INTENT_STATUSES
_BOOTSTRAPPED_DATABASE_URLS: set[str] = set()
_BOOTSTRAP_LOCK = threading.RLock()


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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
        event_dual_write_enabled: bool | None = None,
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
        engine_kwargs: dict[str, Any] = {
            "future": True,
            "pool_pre_ping": True,
        }
        if database_url.startswith("sqlite:"):
            connect_args["check_same_thread"] = False
            shared_registry_key: str | None = None
        else:
            pool_size = max(1, int(get_env("AI_TRADING_OMS_DB_POOL_SIZE", 5, cast=int)))
            max_overflow = max(0, int(get_env("AI_TRADING_OMS_DB_MAX_OVERFLOW", 10, cast=int)))
            pool_timeout = max(
                1.0,
                float(get_env("AI_TRADING_OMS_DB_POOL_TIMEOUT_SEC", 30, cast=float)),
            )
            pool_recycle = max(
                30,
                int(get_env("AI_TRADING_OMS_DB_POOL_RECYCLE_SEC", 1800, cast=int)),
            )
            connect_timeout = max(
                1,
                int(get_env("AI_TRADING_OMS_DB_CONNECT_TIMEOUT_SEC", 10, cast=int)),
            )
            app_name = str(get_env("AI_TRADING_OMS_DB_APP_NAME", "ai_trading_oms") or "").strip()
            connect_args["connect_timeout"] = connect_timeout
            if app_name:
                connect_args["application_name"] = app_name
            engine_kwargs.update(
                {
                    "pool_size": pool_size,
                    "max_overflow": max_overflow,
                    "pool_timeout": pool_timeout,
                    "pool_recycle": pool_recycle,
                }
            )
            shared_registry_key = database_url
        self._engine, self._owns_engine = resolve_shared_engine(
            registry_key=shared_registry_key,
            factory=lambda: create_engine(
                database_url,
                connect_args=connect_args,
                **engine_kwargs,
            ),
        )
        self._session_factory = sessionmaker(
            bind=self._engine,
            autoflush=False,
            expire_on_commit=False,
            future=True,
        )
        if event_dual_write_enabled is None:
            self._event_dual_write_enabled = bool(
                get_env("AI_TRADING_OMS_EVENT_DUAL_WRITE_ENABLED", False, cast=bool)
            )
        else:
            self._event_dual_write_enabled = bool(event_dual_write_enabled)
        self._event_store: Any | None = None
        self._event_store_init_failed = False
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
        shared_bootstrap_key = None if self._database_url.startswith("sqlite:") else self._database_url
        if shared_bootstrap_key:
            with _BOOTSTRAP_LOCK:
                if shared_bootstrap_key in _BOOTSTRAPPED_DATABASE_URLS:
                    return
                with self._engine.begin() as conn:
                    _METADATA.create_all(conn, checkfirst=True)
                _BOOTSTRAPPED_DATABASE_URLS.add(shared_bootstrap_key)
            return
        with self._engine.begin() as conn:
            if self._database_url.startswith("sqlite:"):
                conn.execute(text("PRAGMA journal_mode=WAL;"))
                conn.execute(text("PRAGMA synchronous=NORMAL;"))
            _METADATA.create_all(conn, checkfirst=True)

    @staticmethod
    def _event_idempotency_key(*parts: Any) -> str:
        material = "|".join(str(part) for part in parts if part not in (None, ""))
        if not material:
            material = "intent-store-event"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    @staticmethod
    def _result_rowcount(result: Any) -> int:
        rowcount = getattr(result, "rowcount", None)
        return int(rowcount or 0)

    @staticmethod
    def _result_inserted_primary_key(result: Any) -> list[Any]:
        inserted_primary_key = getattr(result, "inserted_primary_key", None)
        if not inserted_primary_key:
            return []
        return list(inserted_primary_key)

    def _resolve_event_store(self) -> Any | None:
        if not self._event_dual_write_enabled:
            return None
        if self._event_store_init_failed:
            return None
        if self._event_store is not None:
            return self._event_store
        with self._lock:
            if self._event_store is not None:
                return self._event_store
            try:
                from ai_trading.oms.event_store import EventStore

                self._event_store = EventStore(url=self._database_url)
            except Exception as exc:
                self._event_store_init_failed = True
                logger.warning(
                    "OMS_EVENT_STORE_INIT_FAILED",
                    extra={"database_url": self._database_url, "error": str(exc)},
                )
                return None
            return self._event_store

    def _append_oms_event(
        self,
        *,
        event_type: str,
        intent_id: str,
        event_idempotency_key: str,
        payload: Mapping[str, Any] | None = None,
        broker_order_id: str | None = None,
        fill_id: str | None = None,
        error_code: str | None = None,
    ) -> None:
        store = self._resolve_event_store()
        if store is None:
            return
        try:
            store.append_oms_event_payload(
                event_type=event_type,
                event_source="intent_store",
                idempotency_key=event_idempotency_key,
                intent_id=intent_id,
                payload=dict(payload or {}),
                broker_order_id=broker_order_id,
                fill_id=fill_id,
                error_code=error_code,
            )
        except Exception as exc:
            logger.warning(
                "OMS_EVENT_APPEND_FAILED",
                extra={
                    "intent_id": intent_id,
                    "event_type": event_type,
                    "error": str(exc),
                },
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
        status: str = PENDING_SUBMIT_STATUS,
    ) -> tuple[IntentRecord, bool]:
        """Insert or return existing intent keyed by idempotency key."""

        assert _INTENTS_TABLE is not None
        now = self._utcnow_iso()
        record_decision_ts = decision_ts or now
        meta_payload = json.dumps(metadata or {}, sort_keys=True)
        normalized_status = normalize_intent_status(
            status,
            default=PENDING_SUBMIT_STATUS,
        )
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
                with self._engine.connect() as conn:
                    result = conn.execute(
                        select(_INTENTS_TABLE).where(
                            _INTENTS_TABLE.c.idempotency_key == idempotency_key
                        )
                    )
                    try:
                        row = result.mappings().first()
                    finally:
                        result.close()
                        conn.rollback()
                if row is None:  # pragma: no cover - defensive
                    raise
                return (self._row_to_intent(cast(Mapping[str, Any], row)), False)

            with self._engine.connect() as conn:
                result = conn.execute(
                    select(_INTENTS_TABLE).where(_INTENTS_TABLE.c.intent_id == intent_id)
                )
                try:
                    row = result.mappings().first()
                finally:
                    result.close()
                    conn.rollback()
            if row is None:  # pragma: no cover - defensive
                raise RuntimeError("intent_insert_missing_row")
            inserted_record = self._row_to_intent(cast(Mapping[str, Any], row))
            self._append_oms_event(
                event_type="INTENT_CREATED",
                intent_id=inserted_record.intent_id,
                event_idempotency_key=self._event_idempotency_key(
                    "INTENT_CREATED",
                    inserted_record.intent_id,
                    inserted_record.idempotency_key,
                ),
                payload={
                    "symbol": inserted_record.symbol,
                    "side": inserted_record.side,
                    "quantity": inserted_record.quantity,
                    "status": inserted_record.status,
                    "strategy_id": inserted_record.strategy_id,
                    "expected_edge_bps": inserted_record.expected_edge_bps,
                    "regime": inserted_record.regime,
                    "decision_ts": inserted_record.decision_ts,
                },
            )
            return (inserted_record, True)

    def get_intent(self, intent_id: str) -> IntentRecord | None:
        """Return intent by ID."""

        assert _INTENTS_TABLE is not None
        with self._lock, self._engine.connect() as conn:
            result = conn.execute(
                select(_INTENTS_TABLE).where(_INTENTS_TABLE.c.intent_id == intent_id)
            )
            try:
                row = result.mappings().first()
            finally:
                result.close()
                conn.rollback()
        if row is None:
            return None
        return self._row_to_intent(cast(Mapping[str, Any], row))

    def get_intent_by_key(self, idempotency_key: str) -> IntentRecord | None:
        """Return intent by idempotency key."""

        assert _INTENTS_TABLE is not None
        with self._lock, self._engine.connect() as conn:
            result = conn.execute(
                select(_INTENTS_TABLE).where(
                    _INTENTS_TABLE.c.idempotency_key == idempotency_key
                )
            )
            try:
                row = result.mappings().first()
            finally:
                result.close()
                conn.rollback()
        if row is None:
            return None
        return self._row_to_intent(cast(Mapping[str, Any], row))

    def get_open_intents(self) -> list[IntentRecord]:
        """Return all non-terminal intents."""

        assert _INTENTS_TABLE is not None
        ordered_terminal = tuple(sorted(_TERMINAL_STATUSES))
        stmt = (
            select(_INTENTS_TABLE)
            .where(_INTENTS_TABLE.c.status.not_in(ordered_terminal))
            .order_by(_INTENTS_TABLE.c.updated_at.asc())
        )
        with self._lock, self._engine.connect() as conn:
            result = conn.execute(stmt)
            try:
                rows = result.mappings().all()
            finally:
                result.close()
                conn.rollback()
        return [self._row_to_intent(cast(Mapping[str, Any], row)) for row in rows]

    def list_intents(self, *, limit: int = 5000) -> list[IntentRecord]:
        """Return intents ordered by creation time."""

        assert _INTENTS_TABLE is not None
        stmt = select(_INTENTS_TABLE).order_by(_INTENTS_TABLE.c.created_at.asc()).limit(
            max(1, int(limit))
        )
        with self._lock, self._engine.connect() as conn:
            result = conn.execute(stmt)
            try:
                rows = result.mappings().all()
            finally:
                result.close()
                conn.rollback()
        return [self._row_to_intent(cast(Mapping[str, Any], row)) for row in rows]

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
                    _INTENTS_TABLE.c.status == PENDING_SUBMIT_STATUS
                )
                | (
                    (_INTENTS_TABLE.c.status == SUBMITTING_STATUS)
                    & (_INTENTS_TABLE.c.updated_at <= stale_cutoff_iso)
                )
            )
            .values(
                status=status_for_submit_claim(),
                submit_attempts=_INTENTS_TABLE.c.submit_attempts + 1,
                updated_at=now_iso,
                last_error=None,
            )
        )
        with self._lock, self._session_factory.begin() as session:
            result = session.execute(stmt)
        rowcount = self._result_rowcount(result)
        if rowcount > 0:
            claimed_record = self.get_intent(intent_id)
            submit_attempts = int(getattr(claimed_record, "submit_attempts", 0) or 0)
            self._append_oms_event(
                event_type="SUBMIT_CLAIMED",
                intent_id=intent_id,
                event_idempotency_key=self._event_idempotency_key(
                    "SUBMIT_CLAIMED",
                    intent_id,
                    submit_attempts,
                ),
                payload={
                    "stale_after_seconds": stale_after,
                    "submit_attempts": submit_attempts,
                },
            )
            self._append_oms_event(
                event_type="SUBMIT_ATTEMPTED",
                intent_id=intent_id,
                event_idempotency_key=self._event_idempotency_key(
                    "SUBMIT_ATTEMPTED",
                    intent_id,
                    submit_attempts,
                ),
                payload={
                    "submit_attempts": submit_attempts,
                    "attempted_at": now_iso,
                },
            )
        return rowcount > 0

    def mark_submitted(self, intent_id: str, broker_order_id: str) -> None:
        """Mark intent as submitted with broker order id."""

        assert _INTENTS_TABLE is not None
        now = self._utcnow_iso()
        stmt = (
            update(_INTENTS_TABLE)
            .where(_INTENTS_TABLE.c.intent_id == intent_id)
            .values(
                status=status_for_submit_ack(),
                broker_order_id=str(broker_order_id),
                updated_at=now,
            )
        )
        with self._lock, self._session_factory.begin() as session:
            result = session.execute(stmt)
        if self._result_rowcount(result) > 0:
            self._append_oms_event(
                event_type="SUBMIT_ACK",
                intent_id=intent_id,
                event_idempotency_key=self._event_idempotency_key(
                    "SUBMIT_ACK",
                    intent_id,
                    broker_order_id,
                ),
                payload={
                    "broker_order_id": str(broker_order_id),
                    "status": status_for_submit_ack(),
                },
                broker_order_id=str(broker_order_id),
            )

    def record_submit_error(self, intent_id: str, error: str) -> None:
        """Record submit failure while keeping intent retryable."""

        assert _INTENTS_TABLE is not None
        now = self._utcnow_iso()
        stmt = (
            update(_INTENTS_TABLE)
            .where(_INTENTS_TABLE.c.intent_id == intent_id)
            .values(
                status=status_for_submit_error(),
                last_error=str(error)[:500],
                updated_at=now,
            )
        )
        with self._lock, self._session_factory.begin() as session:
            result = session.execute(stmt)
        if self._result_rowcount(result) > 0:
            normalized_error = str(error or "").strip()[:500]
            self._append_oms_event(
                event_type="SUBMIT_REJECT",
                intent_id=intent_id,
                event_idempotency_key=self._event_idempotency_key(
                    "SUBMIT_REJECT",
                    intent_id,
                    normalized_error,
                ),
                payload={
                    "error": normalized_error,
                    "status": status_for_submit_error(),
                },
                error_code=normalized_error[:64] if normalized_error else None,
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
            else_=status_for_fill(None),
        )
        update_stmt = (
            update(_INTENTS_TABLE)
            .where(_INTENTS_TABLE.c.intent_id == intent_id)
            .values(
                status=status_case,
                updated_at=now,
            )
        )
        fill_id_text: str | None = None
        with self._lock, self._session_factory.begin() as session:
            fill_result = session.execute(fill_stmt)
            session.execute(update_stmt)
            inserted_primary = self._result_inserted_primary_key(fill_result)
            if inserted_primary:
                first = inserted_primary[0]
                fill_id_text = str(first) if first is not None else None
        self._append_oms_event(
            event_type="ORDER_PARTIALLY_FILLED",
            intent_id=intent_id,
            event_idempotency_key=self._event_idempotency_key(
                "ORDER_PARTIALLY_FILLED",
                intent_id,
                fill_id_text or now,
            ),
            payload={
                "fill_qty": float(fill_qty),
                "fill_price": (float(fill_price) if fill_price is not None else None),
                "fee": float(fee),
                "liquidity_flag": liquidity_flag,
                "fill_ts": ts,
                "status": status_for_fill(None),
            },
            fill_id=fill_id_text,
        )

    def list_fills(self, intent_id: str) -> list[FillRecord]:
        """Return fills for intent in insertion order."""

        assert _INTENT_FILLS_TABLE is not None
        stmt = (
            select(_INTENT_FILLS_TABLE)
            .where(_INTENT_FILLS_TABLE.c.intent_id == intent_id)
            .order_by(_INTENT_FILLS_TABLE.c.fill_id.asc())
        )
        with self._lock, self._engine.connect() as conn:
            result = conn.execute(stmt)
            try:
                rows = result.mappings().all()
            finally:
                result.close()
                conn.rollback()
        return [self._row_to_fill(cast(Mapping[str, Any], row)) for row in rows]

    def close_intent(
        self,
        intent_id: str,
        *,
        final_status: str,
        last_error: str | None = None,
    ) -> None:
        """Close intent with terminal status."""

        assert _INTENTS_TABLE is not None
        normalized = normalize_terminal_status(final_status)
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
            result = session.execute(stmt)
        if self._result_rowcount(result) > 0:
            mapped_terminal = terminal_event_type(normalized)
            self._append_oms_event(
                event_type=mapped_terminal,
                intent_id=intent_id,
                event_idempotency_key=self._event_idempotency_key(
                    mapped_terminal,
                    intent_id,
                    normalized,
                ),
                payload={
                    "final_status": normalized,
                    "last_error": (str(last_error)[:500] if last_error else None),
                },
                error_code=(str(last_error)[:64] if last_error else None),
            )
            self._append_oms_event(
                event_type="INTENT_CLOSED",
                intent_id=intent_id,
                event_idempotency_key=self._event_idempotency_key(
                    "INTENT_CLOSED",
                    intent_id,
                    normalized,
                ),
                payload={
                    "final_status": normalized,
                    "last_error": (str(last_error)[:500] if last_error else None),
                },
                error_code=(str(last_error)[:64] if last_error else None),
            )

    def close(self) -> None:
        """Dispose DB engine resources."""

        with self._lock:
            if self._event_store is not None:
                try:
                    self._event_store.close()
                except Exception:
                    logger.debug("INTENT_STORE_EVENT_STORE_CLOSE_FAILED", exc_info=True)
                finally:
                    self._event_store = None
            if not self._owns_engine:
                return
            try:
                self._engine.dispose()
            except Exception:
                logger.debug("INTENT_STORE_CLOSE_FAILED", exc_info=True)
