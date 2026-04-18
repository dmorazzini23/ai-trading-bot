"""Immutable OMS/decision event store with opt-in JSONL fallback writes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict
from datetime import UTC, datetime
import json
from pathlib import Path
from threading import RLock
from typing import Any

from ai_trading.config.management import get_env
from ai_trading.logging import get_logger

from .event_types import DecisionEvent, OmsEvent, new_event_uuid

logger = get_logger(__name__)

# Serialize Postgres append-only guard DDL across concurrent bootstrap callers.
_POSTGRES_APPEND_ONLY_GUARD_LOCK_KEY = 346017647123854219

try:
    from sqlalchemy import (
        Column,
        Float,
        Integer,
        MetaData,
        String,
        Table,
        Text,
        UniqueConstraint,
        create_engine,
        func,
        insert,
        select,
        text,
    )
    from sqlalchemy.engine import Engine
    from sqlalchemy.exc import IntegrityError
    from sqlalchemy.orm import sessionmaker

    _SQLALCHEMY_AVAILABLE = True
    _SQLALCHEMY_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover
    _SQLALCHEMY_AVAILABLE = False
    _SQLALCHEMY_IMPORT_ERROR = exc
    MetaData = None  # type: ignore[assignment,misc]
    Table = None  # type: ignore[assignment,misc]
    Engine = Any  # type: ignore[assignment,misc]
    sessionmaker = None  # type: ignore[assignment,misc]
    IntegrityError = Exception  # type: ignore[assignment,misc]

if _SQLALCHEMY_AVAILABLE:
    _EVENT_METADATA = MetaData()
    _OMS_EVENTS_TABLE = Table(
        "oms_events",
        _EVENT_METADATA,
        Column("event_id", Integer, primary_key=True, autoincrement=True),
        Column("event_uuid", String(128), nullable=False, unique=True),
        Column("intent_id", String(128), nullable=True, index=True),
        Column("event_type", String(64), nullable=False, index=True),
        Column("event_ts", String(64), nullable=False, index=True),
        Column("event_source", String(64), nullable=False, index=True),
        Column("idempotency_key", String(128), nullable=False),
        Column("sequence_no", Integer, nullable=False, default=0, server_default="0"),
        Column("payload_json", Text, nullable=False),
        Column("policy_hash", String(128), nullable=True),
        Column("model_hash", String(128), nullable=True),
        Column("error_code", String(64), nullable=True),
        Column("broker_order_id", String(128), nullable=True),
        Column("fill_id", String(128), nullable=True),
        Column("created_at", String(64), nullable=False),
        UniqueConstraint(
            "event_source",
            "idempotency_key",
            name="uq_oms_events_source_idempotency",
        ),
    )
    _DECISION_EVENTS_TABLE = Table(
        "decision_events",
        _EVENT_METADATA,
        Column("decision_id", Integer, primary_key=True, autoincrement=True),
        Column("decision_uuid", String(128), nullable=False, unique=True),
        Column("decision_ts", String(64), nullable=False, index=True),
        Column("symbol", String(32), nullable=False, index=True),
        Column("strategy_id", String(128), nullable=True, index=True),
        Column("decision_action", String(32), nullable=False),
        Column("confidence", Float, nullable=True),
        Column("expected_edge_bps", Float, nullable=True),
        Column("policy_hash", String(128), nullable=True),
        Column("model_hash", String(128), nullable=True),
        Column("config_hash", String(128), nullable=True),
        Column("idempotency_key", String(128), nullable=True),
        Column("features_json", Text, nullable=True),
        Column("context_json", Text, nullable=True),
        Column("created_at", String(64), nullable=False),
        UniqueConstraint(
            "idempotency_key",
            name="uq_decision_events_idempotency_key",
        ),
    )
    _POSITION_SNAPSHOTS_TABLE = Table(
        "position_snapshots",
        _EVENT_METADATA,
        Column("snapshot_id", Integer, primary_key=True, autoincrement=True),
        Column("snapshot_uuid", String(128), nullable=False, unique=True),
        Column("snapshot_ts", String(64), nullable=False, index=True),
        Column("snapshot_source", String(64), nullable=False, index=True),
        Column("symbol", String(32), nullable=False, index=True),
        Column("quantity", Float, nullable=False),
        Column("side", String(16), nullable=True),
        Column("avg_entry_price", Float, nullable=True),
        Column("market_price", Float, nullable=True),
        Column("market_value", Float, nullable=True),
        Column("unrealized_pnl", Float, nullable=True),
        Column("policy_hash", String(128), nullable=True),
        Column("model_hash", String(128), nullable=True),
        Column("idempotency_key", String(128), nullable=False),
        Column("payload_json", Text, nullable=False),
        Column("created_at", String(64), nullable=False),
        UniqueConstraint(
            "snapshot_source",
            "idempotency_key",
            name="uq_position_snapshots_source_idempotency",
        ),
    )
    _RISK_SNAPSHOTS_TABLE = Table(
        "risk_snapshots",
        _EVENT_METADATA,
        Column("risk_snapshot_id", Integer, primary_key=True, autoincrement=True),
        Column("snapshot_uuid", String(128), nullable=False, unique=True),
        Column("snapshot_ts", String(64), nullable=False, index=True),
        Column("snapshot_source", String(64), nullable=False, index=True),
        Column("idempotency_key", String(128), nullable=False),
        Column("policy_hash", String(128), nullable=True),
        Column("model_hash", String(128), nullable=True),
        Column("config_hash", String(128), nullable=True),
        Column("exposure_pct", Float, nullable=True),
        Column("drawdown_pct", Float, nullable=True),
        Column("var_95", Float, nullable=True),
        Column("var_99", Float, nullable=True),
        Column("positions_count", Integer, nullable=True),
        Column("open_orders_count", Integer, nullable=True),
        Column("payload_json", Text, nullable=False),
        Column("created_at", String(64), nullable=False),
        UniqueConstraint(
            "snapshot_source",
            "idempotency_key",
            name="uq_risk_snapshots_source_idempotency",
        ),
    )
else:
    _EVENT_METADATA = None
    _OMS_EVENTS_TABLE = None
    _DECISION_EVENTS_TABLE = None
    _POSITION_SNAPSHOTS_TABLE = None
    _RISK_SNAPSHOTS_TABLE = None


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


class EventStore:
    """Persistent append-only event store."""

    def __init__(
        self,
        path: str | None = None,
        *,
        url: str | None = None,
        engine: Engine | None = None,
    ) -> None:
        if not _SQLALCHEMY_AVAILABLE:
            detail = str(_SQLALCHEMY_IMPORT_ERROR) if _SQLALCHEMY_IMPORT_ERROR else "unknown"
            raise RuntimeError(
                f"SQLAlchemy backend unavailable for EventStore: {detail}. "
                "Install SQLAlchemy>=2.0 and psycopg[binary] to enable durable event storage."
            )

        database_url, locator_path = _normalize_database_url(path=path, url=url)
        self._database_url = database_url
        self._path = locator_path
        self._lock = RLock()
        self._owns_engine = engine is None
        self._jsonl_enabled = bool(
            get_env("AI_TRADING_OMS_EVENT_JSONL_ENABLED", False, cast=bool)
        )
        self._jsonl_path = Path(
            str(
                get_env(
                    "AI_TRADING_OMS_EVENT_JSONL_PATH",
                    "runtime/oms_events.jsonl",
                    cast=str,
                )
            )
        )

        if engine is not None:
            self._engine = engine
        else:
            connect_args: dict[str, Any] = {}
            engine_kwargs: dict[str, Any] = {
                "future": True,
                "pool_pre_ping": True,
            }
            if database_url.startswith("sqlite:"):
                connect_args["check_same_thread"] = False
            else:
                pool_size = max(1, int(get_env("AI_TRADING_OMS_DB_POOL_SIZE", 5, cast=int)))
                max_overflow = max(
                    0,
                    int(get_env("AI_TRADING_OMS_DB_MAX_OVERFLOW", 10, cast=int)),
                )
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
            self._engine = create_engine(
                database_url,
                connect_args=connect_args,
                **engine_kwargs,
            )

        self._session_factory = sessionmaker(
            bind=self._engine,
            autoflush=False,
            expire_on_commit=False,
            future=True,
        )
        self._bootstrap()

    @property
    def database_url(self) -> str:
        """Return backing SQLAlchemy database URL."""

        return self._database_url

    @property
    def path(self) -> Path:
        """Return locator path for sqlite-backed stores."""

        return self._path

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(UTC).isoformat()

    def _bootstrap(self) -> None:
        assert _EVENT_METADATA is not None
        with self._engine.begin() as conn:
            if self._database_url.startswith("sqlite:"):
                conn.execute(text("PRAGMA journal_mode=WAL;"))
                conn.execute(text("PRAGMA synchronous=NORMAL;"))
            _EVENT_METADATA.create_all(conn, checkfirst=True)
            self._ensure_append_only_guards(conn)

    def _ensure_append_only_guards(self, conn: Any) -> None:
        """Install DB-side append-only guards for immutable event tables."""

        dialect_name = str(getattr(conn.dialect, "name", "") or "").strip().lower()
        statements: list[str] = []
        if dialect_name == "sqlite":
            statements = [
                """
                CREATE TRIGGER IF NOT EXISTS trg_oms_events_no_update
                BEFORE UPDATE ON oms_events
                BEGIN
                  SELECT RAISE(ABORT, 'oms_events is append-only');
                END;
                """,
                """
                CREATE TRIGGER IF NOT EXISTS trg_oms_events_no_delete
                BEFORE DELETE ON oms_events
                BEGIN
                  SELECT RAISE(ABORT, 'oms_events is append-only');
                END;
                """,
                """
                CREATE TRIGGER IF NOT EXISTS trg_decision_events_no_update
                BEFORE UPDATE ON decision_events
                BEGIN
                  SELECT RAISE(ABORT, 'decision_events is append-only');
                END;
                """,
                """
                CREATE TRIGGER IF NOT EXISTS trg_decision_events_no_delete
                BEFORE DELETE ON decision_events
                BEGIN
                  SELECT RAISE(ABORT, 'decision_events is append-only');
                END;
                """,
                """
                CREATE TRIGGER IF NOT EXISTS trg_position_snapshots_no_update
                BEFORE UPDATE ON position_snapshots
                BEGIN
                  SELECT RAISE(ABORT, 'position_snapshots is append-only');
                END;
                """,
                """
                CREATE TRIGGER IF NOT EXISTS trg_position_snapshots_no_delete
                BEFORE DELETE ON position_snapshots
                BEGIN
                  SELECT RAISE(ABORT, 'position_snapshots is append-only');
                END;
                """,
                """
                CREATE TRIGGER IF NOT EXISTS trg_risk_snapshots_no_update
                BEFORE UPDATE ON risk_snapshots
                BEGIN
                  SELECT RAISE(ABORT, 'risk_snapshots is append-only');
                END;
                """,
                """
                CREATE TRIGGER IF NOT EXISTS trg_risk_snapshots_no_delete
                BEFORE DELETE ON risk_snapshots
                BEGIN
                  SELECT RAISE(ABORT, 'risk_snapshots is append-only');
                END;
                """,
            ]
        elif dialect_name in {"postgresql", "postgres"}:
            statements = [
                f"SELECT pg_advisory_xact_lock({_POSTGRES_APPEND_ONLY_GUARD_LOCK_KEY});",
                """
                CREATE OR REPLACE FUNCTION ai_trading_prevent_append_only_mutation()
                RETURNS trigger AS $$
                BEGIN
                  RAISE EXCEPTION '% is append-only', TG_TABLE_NAME;
                END;
                $$ LANGUAGE plpgsql;
                """,
                """
                DO $$
                BEGIN
                  IF NOT EXISTS (
                    SELECT 1 FROM pg_trigger WHERE tgname = 'trg_oms_events_no_update'
                  ) THEN
                    CREATE TRIGGER trg_oms_events_no_update
                    BEFORE UPDATE ON oms_events
                    FOR EACH ROW
                    EXECUTE FUNCTION ai_trading_prevent_append_only_mutation();
                  END IF;
                END $$;
                """,
                """
                DO $$
                BEGIN
                  IF NOT EXISTS (
                    SELECT 1 FROM pg_trigger WHERE tgname = 'trg_oms_events_no_delete'
                  ) THEN
                    CREATE TRIGGER trg_oms_events_no_delete
                    BEFORE DELETE ON oms_events
                    FOR EACH ROW
                    EXECUTE FUNCTION ai_trading_prevent_append_only_mutation();
                  END IF;
                END $$;
                """,
                """
                DO $$
                BEGIN
                  IF NOT EXISTS (
                    SELECT 1 FROM pg_trigger WHERE tgname = 'trg_decision_events_no_update'
                  ) THEN
                    CREATE TRIGGER trg_decision_events_no_update
                    BEFORE UPDATE ON decision_events
                    FOR EACH ROW
                    EXECUTE FUNCTION ai_trading_prevent_append_only_mutation();
                  END IF;
                END $$;
                """,
                """
                DO $$
                BEGIN
                  IF NOT EXISTS (
                    SELECT 1 FROM pg_trigger WHERE tgname = 'trg_decision_events_no_delete'
                  ) THEN
                    CREATE TRIGGER trg_decision_events_no_delete
                    BEFORE DELETE ON decision_events
                    FOR EACH ROW
                    EXECUTE FUNCTION ai_trading_prevent_append_only_mutation();
                  END IF;
                END $$;
                """,
                """
                DO $$
                BEGIN
                  IF NOT EXISTS (
                    SELECT 1 FROM pg_trigger WHERE tgname = 'trg_position_snapshots_no_update'
                  ) THEN
                    CREATE TRIGGER trg_position_snapshots_no_update
                    BEFORE UPDATE ON position_snapshots
                    FOR EACH ROW
                    EXECUTE FUNCTION ai_trading_prevent_append_only_mutation();
                  END IF;
                END $$;
                """,
                """
                DO $$
                BEGIN
                  IF NOT EXISTS (
                    SELECT 1 FROM pg_trigger WHERE tgname = 'trg_position_snapshots_no_delete'
                  ) THEN
                    CREATE TRIGGER trg_position_snapshots_no_delete
                    BEFORE DELETE ON position_snapshots
                    FOR EACH ROW
                    EXECUTE FUNCTION ai_trading_prevent_append_only_mutation();
                  END IF;
                END $$;
                """,
                """
                DO $$
                BEGIN
                  IF NOT EXISTS (
                    SELECT 1 FROM pg_trigger WHERE tgname = 'trg_risk_snapshots_no_update'
                  ) THEN
                    CREATE TRIGGER trg_risk_snapshots_no_update
                    BEFORE UPDATE ON risk_snapshots
                    FOR EACH ROW
                    EXECUTE FUNCTION ai_trading_prevent_append_only_mutation();
                  END IF;
                END $$;
                """,
                """
                DO $$
                BEGIN
                  IF NOT EXISTS (
                    SELECT 1 FROM pg_trigger WHERE tgname = 'trg_risk_snapshots_no_delete'
                  ) THEN
                    CREATE TRIGGER trg_risk_snapshots_no_delete
                    BEFORE DELETE ON risk_snapshots
                    FOR EACH ROW
                    EXECUTE FUNCTION ai_trading_prevent_append_only_mutation();
                  END IF;
                END $$;
                """,
            ]

        for stmt in statements:
            conn.execute(text(stmt))

    def _write_jsonl(self, payload: Mapping[str, Any]) -> None:
        if not self._jsonl_enabled:
            return
        try:
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self._jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(payload), sort_keys=True))
                handle.write("\n")
        except OSError as exc:
            logger.warning(
                "OMS_EVENT_JSONL_WRITE_FAILED",
                extra={"path": str(self._jsonl_path), "error": str(exc)},
            )

    def _next_sequence_no(self, intent_id: str | None) -> int:
        if intent_id in (None, ""):
            return 0
        assert _OMS_EVENTS_TABLE is not None
        stmt = select(func.max(_OMS_EVENTS_TABLE.c.sequence_no)).where(
            _OMS_EVENTS_TABLE.c.intent_id == str(intent_id)
        )
        with self._session_factory() as session:
            current = session.execute(stmt).scalar_one_or_none()
        if current is None:
            return 1
        try:
            return int(current) + 1
        except (TypeError, ValueError):
            return 1

    def append_oms_event(self, event: OmsEvent) -> bool:
        """Append OMS event. Returns False on duplicate idempotency."""

        assert _OMS_EVENTS_TABLE is not None
        normalized = event.normalized()
        sequence_no = (
            int(normalized.sequence_no)
            if normalized.sequence_no is not None
            else self._next_sequence_no(normalized.intent_id)
        )
        created_at = self._utc_now_iso()
        payload_dict = {
            "event_uuid": normalized.event_uuid,
            "intent_id": normalized.intent_id,
            "event_type": normalized.event_type,
            "event_ts": normalized.event_ts,
            "event_source": normalized.event_source,
            "idempotency_key": normalized.idempotency_key,
            "sequence_no": max(0, int(sequence_no)),
            "payload_json": json.dumps(normalized.payload, sort_keys=True),
            "policy_hash": normalized.policy_hash,
            "model_hash": normalized.model_hash,
            "error_code": normalized.error_code,
            "broker_order_id": normalized.broker_order_id,
            "fill_id": normalized.fill_id,
            "created_at": created_at,
        }
        db_persisted = False
        with self._lock:
            try:
                with self._session_factory.begin() as session:
                    session.execute(insert(_OMS_EVENTS_TABLE).values(**payload_dict))
                db_persisted = True
            except IntegrityError:
                db_persisted = False
            except Exception as exc:
                logger.warning(
                    "OMS_EVENT_DB_WRITE_FAILED",
                    extra={
                        "intent_id": normalized.intent_id,
                        "event_type": normalized.event_type,
                        "event_source": normalized.event_source,
                        "idempotency_key": normalized.idempotency_key,
                        "error": str(exc),
                    },
                )
        self._write_jsonl(
            {
                "kind": "oms_event",
                "db_persisted": db_persisted,
                **payload_dict,
                "payload": normalized.payload,
            }
        )
        return db_persisted

    def append_oms_event_payload(
        self,
        *,
        event_type: str,
        event_source: str,
        idempotency_key: str,
        payload: Mapping[str, Any] | None = None,
        intent_id: str | None = None,
        sequence_no: int | None = None,
        event_ts: str | None = None,
        event_uuid: str | None = None,
        policy_hash: str | None = None,
        model_hash: str | None = None,
        error_code: str | None = None,
        broker_order_id: str | None = None,
        fill_id: str | None = None,
    ) -> bool:
        """Append an OMS event from primitive payload fields."""

        normalized_type = str(event_type or "").strip().upper() or "RECONCILE_UPDATE"
        event = OmsEvent(
            event_type=normalized_type,  # type: ignore[arg-type]
            event_source=str(event_source or "").strip() or "unknown",
            idempotency_key=str(idempotency_key or "").strip(),
            payload=dict(payload or {}),
            intent_id=(str(intent_id) if intent_id not in (None, "") else None),
            sequence_no=sequence_no,
            event_ts=event_ts,
            event_uuid=event_uuid,
            policy_hash=policy_hash,
            model_hash=model_hash,
            error_code=error_code,
            broker_order_id=broker_order_id,
            fill_id=fill_id,
        )
        return self.append_oms_event(event)

    def append_batch(self, events: Iterable[OmsEvent]) -> int:
        """Append a batch of OMS events, returning successful insert count."""

        inserted = 0
        for event in events:
            if self.append_oms_event(event):
                inserted += 1
        return inserted

    def append_decision_event(self, event: DecisionEvent) -> bool:
        """Append decision event. Returns False on duplicate idempotency."""

        assert _DECISION_EVENTS_TABLE is not None
        normalized = event.normalized()
        payload_dict = {
            "decision_uuid": normalized.decision_uuid,
            "decision_ts": normalized.decision_ts,
            "symbol": normalized.symbol,
            "strategy_id": normalized.strategy_id,
            "decision_action": normalized.decision_action,
            "confidence": normalized.confidence,
            "expected_edge_bps": normalized.expected_edge_bps,
            "policy_hash": normalized.policy_hash,
            "model_hash": normalized.model_hash,
            "config_hash": normalized.config_hash,
            "idempotency_key": normalized.idempotency_key,
            "features_json": json.dumps(normalized.features or {}, sort_keys=True),
            "context_json": json.dumps(normalized.context or {}, sort_keys=True),
            "created_at": self._utc_now_iso(),
        }
        db_persisted = False
        with self._lock:
            try:
                with self._session_factory.begin() as session:
                    session.execute(insert(_DECISION_EVENTS_TABLE).values(**payload_dict))
                db_persisted = True
            except IntegrityError:
                db_persisted = False
            except Exception as exc:
                logger.warning(
                    "DECISION_EVENT_DB_WRITE_FAILED",
                    extra={
                        "symbol": normalized.symbol,
                        "decision_action": normalized.decision_action,
                        "idempotency_key": normalized.idempotency_key,
                        "error": str(exc),
                    },
                )
        self._write_jsonl({"kind": "decision_event", "db_persisted": db_persisted, **payload_dict})
        return db_persisted

    def append_position_snapshot_payload(
        self,
        *,
        snapshot_source: str,
        idempotency_key: str,
        symbol: str,
        quantity: float,
        payload: Mapping[str, Any] | None = None,
        side: str | None = None,
        avg_entry_price: float | None = None,
        market_price: float | None = None,
        market_value: float | None = None,
        unrealized_pnl: float | None = None,
        snapshot_ts: str | None = None,
        snapshot_uuid: str | None = None,
        policy_hash: str | None = None,
        model_hash: str | None = None,
    ) -> bool:
        """Append a single immutable position snapshot row."""

        assert _POSITION_SNAPSHOTS_TABLE is not None
        resolved_symbol = str(symbol or "").strip().upper()
        if not resolved_symbol:
            raise ValueError("symbol is required for position snapshot")
        payload_dict = {
            "snapshot_uuid": str(snapshot_uuid or new_event_uuid()),
            "snapshot_ts": str(snapshot_ts or self._utc_now_iso()),
            "snapshot_source": str(snapshot_source or "").strip() or "unknown",
            "symbol": resolved_symbol,
            "quantity": float(quantity),
            "side": (str(side).strip().lower() if side not in (None, "") else None),
            "avg_entry_price": (
                float(avg_entry_price)
                if avg_entry_price is not None
                else None
            ),
            "market_price": float(market_price) if market_price is not None else None,
            "market_value": float(market_value) if market_value is not None else None,
            "unrealized_pnl": (
                float(unrealized_pnl) if unrealized_pnl is not None else None
            ),
            "policy_hash": (
                str(policy_hash) if policy_hash not in (None, "") else None
            ),
            "model_hash": str(model_hash) if model_hash not in (None, "") else None,
            "idempotency_key": str(idempotency_key or "").strip(),
            "payload_json": json.dumps(dict(payload or {}), sort_keys=True),
            "created_at": self._utc_now_iso(),
        }
        if not payload_dict["idempotency_key"]:
            raise ValueError("idempotency_key is required for position snapshot")
        db_persisted = False
        with self._lock:
            try:
                with self._session_factory.begin() as session:
                    session.execute(insert(_POSITION_SNAPSHOTS_TABLE).values(**payload_dict))
                db_persisted = True
            except IntegrityError:
                db_persisted = False
            except Exception as exc:
                logger.warning(
                    "POSITION_SNAPSHOT_DB_WRITE_FAILED",
                    extra={
                        "symbol": resolved_symbol,
                        "snapshot_source": payload_dict["snapshot_source"],
                        "idempotency_key": payload_dict["idempotency_key"],
                        "error": str(exc),
                    },
                )
        self._write_jsonl(
            {
                "kind": "position_snapshot",
                "db_persisted": db_persisted,
                **payload_dict,
                "payload": dict(payload or {}),
            }
        )
        return db_persisted

    def append_risk_snapshot_payload(
        self,
        *,
        snapshot_source: str,
        idempotency_key: str,
        payload: Mapping[str, Any] | None = None,
        snapshot_ts: str | None = None,
        snapshot_uuid: str | None = None,
        policy_hash: str | None = None,
        model_hash: str | None = None,
        config_hash: str | None = None,
        exposure_pct: float | None = None,
        drawdown_pct: float | None = None,
        var_95: float | None = None,
        var_99: float | None = None,
        positions_count: int | None = None,
        open_orders_count: int | None = None,
    ) -> bool:
        """Append an immutable risk snapshot row."""

        assert _RISK_SNAPSHOTS_TABLE is not None
        payload_dict = {
            "snapshot_uuid": str(snapshot_uuid or new_event_uuid()),
            "snapshot_ts": str(snapshot_ts or self._utc_now_iso()),
            "snapshot_source": str(snapshot_source or "").strip() or "unknown",
            "idempotency_key": str(idempotency_key or "").strip(),
            "policy_hash": (
                str(policy_hash) if policy_hash not in (None, "") else None
            ),
            "model_hash": str(model_hash) if model_hash not in (None, "") else None,
            "config_hash": str(config_hash) if config_hash not in (None, "") else None,
            "exposure_pct": float(exposure_pct) if exposure_pct is not None else None,
            "drawdown_pct": float(drawdown_pct) if drawdown_pct is not None else None,
            "var_95": float(var_95) if var_95 is not None else None,
            "var_99": float(var_99) if var_99 is not None else None,
            "positions_count": (
                int(positions_count) if positions_count is not None else None
            ),
            "open_orders_count": (
                int(open_orders_count) if open_orders_count is not None else None
            ),
            "payload_json": json.dumps(dict(payload or {}), sort_keys=True),
            "created_at": self._utc_now_iso(),
        }
        if not payload_dict["idempotency_key"]:
            raise ValueError("idempotency_key is required for risk snapshot")
        db_persisted = False
        with self._lock:
            try:
                with self._session_factory.begin() as session:
                    session.execute(insert(_RISK_SNAPSHOTS_TABLE).values(**payload_dict))
                db_persisted = True
            except IntegrityError:
                db_persisted = False
            except Exception as exc:
                logger.warning(
                    "RISK_SNAPSHOT_DB_WRITE_FAILED",
                    extra={
                        "snapshot_source": payload_dict["snapshot_source"],
                        "idempotency_key": payload_dict["idempotency_key"],
                        "error": str(exc),
                    },
                )
        self._write_jsonl(
            {
                "kind": "risk_snapshot",
                "db_persisted": db_persisted,
                **payload_dict,
                "payload": dict(payload or {}),
            }
        )
        return db_persisted

    def list_oms_events(
        self,
        *,
        intent_id: str | None = None,
        event_source: str | None = None,
        event_type: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """List OMS events optionally filtered by intent, source, and type."""

        assert _OMS_EVENTS_TABLE is not None
        stmt = select(_OMS_EVENTS_TABLE).order_by(_OMS_EVENTS_TABLE.c.event_id.asc()).limit(
            max(1, int(limit))
        )
        if intent_id not in (None, ""):
            stmt = stmt.where(_OMS_EVENTS_TABLE.c.intent_id == str(intent_id))
        if event_source not in (None, ""):
            stmt = stmt.where(_OMS_EVENTS_TABLE.c.event_source == str(event_source).strip())
        if event_type not in (None, ""):
            stmt = stmt.where(
                _OMS_EVENTS_TABLE.c.event_type == str(event_type).strip().upper()
            )
        with self._session_factory() as session:
            rows = session.execute(stmt).mappings().all()
        return [dict(row) for row in rows]

    def list_decision_events(self, *, symbol: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        """List decision events optionally filtered by symbol."""

        assert _DECISION_EVENTS_TABLE is not None
        stmt = select(_DECISION_EVENTS_TABLE).order_by(
            _DECISION_EVENTS_TABLE.c.decision_id.asc()
        ).limit(max(1, int(limit)))
        if symbol not in (None, ""):
            stmt = stmt.where(_DECISION_EVENTS_TABLE.c.symbol == str(symbol).upper())
        with self._session_factory() as session:
            rows = session.execute(stmt).mappings().all()
        return [dict(row) for row in rows]

    def list_position_snapshots(
        self,
        *,
        symbol: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """List immutable position snapshot rows ordered by insertion."""

        assert _POSITION_SNAPSHOTS_TABLE is not None
        stmt = select(_POSITION_SNAPSHOTS_TABLE).order_by(
            _POSITION_SNAPSHOTS_TABLE.c.snapshot_id.asc()
        ).limit(max(1, int(limit)))
        if symbol not in (None, ""):
            stmt = stmt.where(_POSITION_SNAPSHOTS_TABLE.c.symbol == str(symbol).upper())
        with self._session_factory() as session:
            rows = session.execute(stmt).mappings().all()
        return [dict(row) for row in rows]

    def list_risk_snapshots(
        self,
        *,
        source: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """List immutable risk snapshot rows ordered by insertion."""

        assert _RISK_SNAPSHOTS_TABLE is not None
        stmt = select(_RISK_SNAPSHOTS_TABLE).order_by(
            _RISK_SNAPSHOTS_TABLE.c.risk_snapshot_id.asc()
        ).limit(max(1, int(limit)))
        if source not in (None, ""):
            stmt = stmt.where(
                _RISK_SNAPSHOTS_TABLE.c.snapshot_source == str(source).strip()
            )
        with self._session_factory() as session:
            rows = session.execute(stmt).mappings().all()
        return [dict(row) for row in rows]

    def migration_status(self, expected_revision: str | None = None) -> dict[str, Any]:
        """Return migration status from alembic_version table when available."""

        expected = str(expected_revision or "").strip() or None
        payload: dict[str, Any] = {
            "expected_revision": expected,
            "current_revision": None,
            "at_head": None,
        }
        try:
            with self._engine.connect() as conn:
                row = conn.execute(text("SELECT version_num FROM alembic_version LIMIT 1")).first()
            if row is not None and len(row) > 0:
                payload["current_revision"] = str(row[0])
        except Exception:
            return payload

        if expected:
            payload["at_head"] = payload["current_revision"] == expected
        return payload

    def is_healthy(self, expected_revision: str | None = None) -> dict[str, Any]:
        """Return readiness payload for health checks."""

        payload: dict[str, Any] = {
            "ok": False,
            "connected": False,
            "database_url_configured": bool(str(self._database_url or "").strip()),
            "backend": "sqlite" if self._database_url.startswith("sqlite:") else "postgres",
        }
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            payload["connected"] = True
            migration = self.migration_status(expected_revision=expected_revision)
            payload["migration"] = migration
            at_head = migration.get("at_head")
            if at_head is None:
                payload["ok"] = True
            else:
                payload["ok"] = bool(at_head)
            return payload
        except Exception as exc:
            payload["error"] = str(exc)
            return payload

    def close(self) -> None:
        """Dispose engine resources when owned by this store."""

        if not self._owns_engine:
            return
        with self._lock:
            try:
                self._engine.dispose()
            except Exception:
                logger.debug("EVENT_STORE_CLOSE_FAILED", exc_info=True)


def decision_event_to_payload(event: DecisionEvent) -> dict[str, Any]:
    """Serialize decision event dataclass to plain mapping."""

    return asdict(event.normalized())
