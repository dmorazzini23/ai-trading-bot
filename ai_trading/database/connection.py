"""Database connection management backed by SQLAlchemy sessions."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
import threading
import uuid
from typing import Any

from sqlalchemy import (
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    insert,
    select,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from ai_trading.config.management import get_env
from ai_trading.logging import logger

_LEGACY_METADATA = MetaData()

_TRADES_TABLE = Table(
    "trades",
    _LEGACY_METADATA,
    Column("id", String(128), primary_key=True),
    Column("symbol", String(32), nullable=True),
    Column("side", String(16), nullable=True),
    Column("order_type", String(32), nullable=True),
    Column("quantity", Float, nullable=True),
    Column("price", Float, nullable=True),
    Column("executed_price", Float, nullable=True),
    Column("status", String(32), nullable=True),
    Column("created_at", String(64), nullable=True),
    Column("executed_at", String(64), nullable=True),
    Column("commission", Float, nullable=True),
    Column("slippage", Float, nullable=True),
    Column("strategy_id", String(128), nullable=True),
    Column("signal_strength", Float, nullable=True),
    Column("stop_loss", Float, nullable=True),
    Column("take_profit", Float, nullable=True),
    Column("notes", Text, nullable=True),
    Column("market_data_snapshot", Text, nullable=True),
)

_PORTFOLIO_TABLE = Table(
    "portfolio",
    _LEGACY_METADATA,
    Column("id", String(128), primary_key=True),
    Column("account_id", String(128), nullable=True),
    Column("symbol", String(32), nullable=True),
    Column("quantity", Float, nullable=True),
    Column("average_cost", Float, nullable=True),
    Column("current_price", Float, nullable=True),
    Column("last_updated", String(64), nullable=True),
    Column("asset_class", String(32), nullable=True),
    Column("sector", String(128), nullable=True),
    Column("market_value", Float, nullable=True),
    Column("unrealized_pnl", Float, nullable=True),
    Column("realized_pnl", Float, nullable=True),
    Column("day_change", Float, nullable=True),
    Column("day_change_percent", Float, nullable=True),
)

_RISK_METRICS_TABLE = Table(
    "risk_metrics",
    _LEGACY_METADATA,
    Column("id", String(128), primary_key=True),
    Column("portfolio_id", String(128), nullable=True),
    Column("calculation_date", String(64), nullable=True),
    Column("var_95", Float, nullable=True),
    Column("var_99", Float, nullable=True),
    Column("expected_shortfall", Float, nullable=True),
    Column("sharpe_ratio", Float, nullable=True),
    Column("sortino_ratio", Float, nullable=True),
    Column("max_drawdown", Float, nullable=True),
    Column("current_drawdown", Float, nullable=True),
    Column("volatility", Float, nullable=True),
    Column("beta", Float, nullable=True),
    Column("correlation_spy", Float, nullable=True),
    Column("concentration_risk", Float, nullable=True),
    Column("liquidity_risk", Float, nullable=True),
)

_PERFORMANCE_METRICS_TABLE = Table(
    "performance_metrics",
    _LEGACY_METADATA,
    Column("id", String(128), primary_key=True),
    Column("strategy_id", String(128), nullable=True),
    Column("portfolio_id", String(128), nullable=True),
    Column("period_start", String(64), nullable=True),
    Column("period_end", String(64), nullable=True),
    Column("total_return", Float, nullable=True),
    Column("annualized_return", Float, nullable=True),
    Column("benchmark_return", Float, nullable=True),
    Column("alpha", Float, nullable=True),
    Column("tracking_error", Float, nullable=True),
    Column("information_ratio", Float, nullable=True),
    Column("win_rate", Float, nullable=True),
    Column("profit_factor", Float, nullable=True),
    Column("average_win", Float, nullable=True),
    Column("average_loss", Float, nullable=True),
    Column("largest_win", Float, nullable=True),
    Column("largest_loss", Float, nullable=True),
    Column("total_trades", Integer, nullable=True),
    Column("winning_trades", Integer, nullable=True),
    Column("losing_trades", Integer, nullable=True),
)

_MODEL_TABLES: dict[str, Table] = {
    "Trade": _TRADES_TABLE,
    "Portfolio": _PORTFOLIO_TABLE,
    "RiskMetric": _RISK_METRICS_TABLE,
    "PerformanceMetric": _PERFORMANCE_METRICS_TABLE,
}


def _normalize_database_url(connection_string: str | None) -> str:
    raw = str(connection_string or "").strip()
    if not raw:
        raw = str(get_env("DATABASE_URL", "", cast=str) or "").strip()
    if not raw:
        return "sqlite:///trading.db"
    if raw.startswith("postgres://"):
        return f"postgresql+psycopg://{raw[len('postgres://') :]}"
    if raw.startswith("postgresql://") and "+" not in raw.split("://", 1)[0]:
        return f"postgresql+psycopg://{raw[len('postgresql://') :]}"
    return raw


class DatabaseManager:
    """Database connection manager with SQLAlchemy-backed sessions."""

    def __init__(self, connection_string: str | None = None, **kwargs: Any):
        self.connection_string = _normalize_database_url(connection_string)
        self.pool_size = int(kwargs.get("pool_size", 20))
        self.max_overflow = int(kwargs.get("max_overflow", 10))
        self.pool_timeout = int(kwargs.get("pool_timeout", 30))
        self.pool_recycle = int(kwargs.get("pool_recycle", 3600))
        self.connect_timeout = int(kwargs.get("connect_timeout", 10))
        self._connections: dict[int | None, "DatabaseSession"] = {}
        self._connection_lock = threading.RLock()
        self._is_connected = False
        self._engine: Engine | None = None
        self._session_factory: sessionmaker[Session] | None = None
        logger.info(
            "DatabaseManager initialized",
            extra={"connection_string": self.connection_string},
        )

    def connect(self) -> bool:
        """Establish database connection and initialize session factory."""
        with self._connection_lock:
            if self._is_connected:
                logger.debug("Database already connected")
                return True
            try:
                connect_args: dict[str, Any] = {}
                engine_kwargs: dict[str, Any] = {
                    "future": True,
                    "pool_pre_ping": True,
                }
                if self.connection_string.startswith("sqlite:"):
                    connect_args["check_same_thread"] = False
                else:
                    connect_args["connect_timeout"] = max(1, int(self.connect_timeout))
                    engine_kwargs.update(
                        {
                            "pool_size": max(1, int(self.pool_size)),
                            "max_overflow": max(0, int(self.max_overflow)),
                            "pool_timeout": max(1, int(self.pool_timeout)),
                            "pool_recycle": max(30, int(self.pool_recycle)),
                        }
                    )

                self._engine = create_engine(
                    self.connection_string,
                    connect_args=connect_args,
                    **engine_kwargs,
                )
                with self._engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                _LEGACY_METADATA.create_all(self._engine, checkfirst=True)
                self._session_factory = sessionmaker(
                    bind=self._engine,
                    autoflush=False,
                    expire_on_commit=False,
                    future=True,
                )
                self._is_connected = True
                logger.info("Database connection established successfully")
                return True
            except (SQLAlchemyError, TimeoutError, OSError, RuntimeError, ValueError) as exc:
                logger.error(f"Failed to connect to database: {exc}")
                self._engine = None
                self._session_factory = None
                self._is_connected = False
                return False

    def disconnect(self) -> None:
        """Close database connections and cleanup resources."""
        with self._connection_lock:
            if not self._is_connected:
                logger.debug("Database not connected")
                return
            for session in list(self._connections.values()):
                try:
                    session.close()
                except (SQLAlchemyError, RuntimeError, ValueError):
                    logger.debug("Database session close failed", exc_info=True)
            self._connections.clear()
            if self._engine is not None:
                try:
                    self._engine.dispose()
                except (SQLAlchemyError, RuntimeError, ValueError):
                    logger.debug("Database engine dispose failed", exc_info=True)
            self._engine = None
            self._session_factory = None
            self._is_connected = False
            logger.info("Database connection closed successfully")

    def is_healthy(self) -> bool:
        """Check database connection health."""
        if not self._is_connected or self._engine is None:
            return False
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except (SQLAlchemyError, TimeoutError, OSError, RuntimeError) as exc:
            logger.error(f"Database health check failed: {exc}")
            return False

    def get_connection_info(self) -> dict[str, Any]:
        """Get current connection pool information."""
        backend = "sqlite" if self.connection_string.startswith("sqlite:") else "postgres"
        return {
            "connection_string": self.connection_string,
            "backend": backend,
            "is_connected": self._is_connected,
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "active_connections": len(self._connections),
            "pool_timeout": self.pool_timeout,
            "pool_recycle": self.pool_recycle,
        }

    @contextmanager
    def get_session(self):
        """Get database session with automatic cleanup."""
        with self._connection_lock:
            if not self._is_connected or self._session_factory is None:
                raise RuntimeError("Database not connected")
            session_id = threading.current_thread().ident
            raw_session = self._session_factory()
            session = DatabaseSession(session_id=session_id, session=raw_session)
            self._connections[session_id] = session
        logger.debug(f"Database session {session_id} created")
        try:
            yield session
        except (SQLAlchemyError, TimeoutError, OSError, RuntimeError, ValueError) as exc:
            logger.error(f"Database session error: {exc}")
            session.rollback()
            raise
        finally:
            try:
                session.close()
            finally:
                with self._connection_lock:
                    self._connections.pop(session_id, None)
            logger.debug(f"Database session {session_id} closed")


class DatabaseSession:
    """Database session wrapper for transaction management."""

    def __init__(self, session_id: int | None, session: Session) -> None:
        self.session_id = session_id
        self._session = session
        self.is_active = True
        self.transaction_active = False

    def _require_active(self) -> None:
        if not self.is_active:
            raise RuntimeError("Session is not active")

    @staticmethod
    def _resolve_table(model_or_instance: Any) -> Table:
        model_name = (
            model_or_instance.__name__
            if isinstance(model_or_instance, type)
            else type(model_or_instance).__name__
        )
        table = _MODEL_TABLES.get(model_name)
        if table is None:
            raise ValueError(f"Unsupported model type: {model_name}")
        return table

    @staticmethod
    def _coerce_payload(value: Any) -> dict[str, Any]:
        payload: dict[str, Any]
        if isinstance(value, Mapping):
            payload = dict(value)
        elif hasattr(value, "to_dict") and callable(value.to_dict):
            payload = dict(value.to_dict())
        else:
            payload = dict(getattr(value, "__dict__", {}))
        for key, item in list(payload.items()):
            if isinstance(item, datetime):
                payload[key] = item.isoformat()
        return payload

    def begin(self) -> None:
        """Begin a database transaction."""
        self._require_active()
        if self.transaction_active:
            logger.warning(f"Transaction already active for session {self.session_id}")
            return
        self._session.begin()
        self.transaction_active = True
        logger.debug(f"Transaction started for session {self.session_id}")

    def commit(self) -> None:
        """Commit the current transaction."""
        self._require_active()
        try:
            self._session.commit()
            self.transaction_active = False
            logger.debug(f"Transaction committed for session {self.session_id}")
        except (SQLAlchemyError, RuntimeError, ValueError):
            self.rollback()
            raise

    def rollback(self) -> None:
        """Rollback the current transaction."""
        if not self.is_active:
            return
        self._session.rollback()
        self.transaction_active = False
        logger.debug(f"Transaction rolled back for session {self.session_id}")

    def close(self) -> None:
        """Close the database session."""
        if not self.is_active:
            return
        if self._session.in_transaction():
            self._session.rollback()
        self._session.close()
        self.is_active = False
        self.transaction_active = False
        logger.debug(f"Session {self.session_id} closed")

    def execute(self, query: str, params: dict | None = None) -> dict[str, Any]:
        """Execute a SQL query and return rows when available."""
        self._require_active()
        result = self._session.execute(text(query), params or {})
        rows: list[dict[str, Any]] = []
        if result.returns_rows:
            rows = [dict(row) for row in result.mappings().all()]
        rowcount = int(result.rowcount or 0) if result.rowcount is not None else 0
        if rowcount < 0:
            rowcount = 0
        self.transaction_active = self._session.in_transaction()
        return {"rows_affected": rowcount, "rows": rows, "result": "success"}

    def query(self, model_class: Any, filters: dict | None = None) -> list[Any]:
        """Query database for model instances."""
        self._require_active()
        table = self._resolve_table(model_class)
        stmt = select(table)
        for key, value in (filters or {}).items():
            if key in table.c:
                stmt = stmt.where(table.c[key] == value)
        rows = self._session.execute(stmt).mappings().all()
        if not isinstance(model_class, type):
            return [dict(row) for row in rows]
        instances: list[Any] = []
        for row in rows:
            payload = dict(row)
            try:
                instances.append(model_class(**payload))
            except (TypeError, ValueError):
                instances.append(payload)
        return instances

    def add(self, instance: Any) -> None:
        """Add instance to session for insertion."""
        self._require_active()
        table = self._resolve_table(instance)
        payload = self._coerce_payload(instance)
        allowed_keys = set(table.c.keys())
        payload = {key: value for key, value in payload.items() if key in allowed_keys}
        if "id" in allowed_keys and payload.get("id") in (None, ""):
            payload["id"] = str(uuid.uuid4())
        if "created_at" in allowed_keys and payload.get("created_at") in (None, ""):
            payload["created_at"] = datetime.now(UTC).isoformat()
        self._session.execute(insert(table).values(**payload))
        self.transaction_active = self._session.in_transaction()
        logger.debug(f"Added {type(instance).__name__} to session {self.session_id}")

    def delete(self, instance: Any) -> None:
        """Delete instance by primary ID."""
        self._require_active()
        table = self._resolve_table(instance)
        instance_id = getattr(instance, "id", None)
        if instance_id in (None, ""):
            raise ValueError("Cannot delete instance without id")
        self._session.execute(delete(table).where(table.c.id == str(instance_id)))
        self.transaction_active = self._session.in_transaction()
        logger.debug(f"Deleted {type(instance).__name__} in session {self.session_id}")


_db_manager: DatabaseManager | None = None


def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def get_session():
    """Get a database session from the global manager."""
    db_manager = get_database_manager()
    return db_manager.get_session()


def initialize_database(connection_string: str | None = None, **kwargs: Any) -> bool:
    """Initialize the global database manager."""
    global _db_manager
    try:
        _db_manager = DatabaseManager(connection_string, **kwargs)
        return _db_manager.connect()
    except (SQLAlchemyError, TimeoutError, ConnectionError, OSError, RuntimeError) as exc:
        logger.error(f"Failed to initialize database: {exc}")
        return False


def shutdown_database() -> None:
    """Shutdown the global database manager."""
    global _db_manager
    if _db_manager is not None:
        _db_manager.disconnect()
        _db_manager = None
