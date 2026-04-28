"""Tests for SQLAlchemy-backed database connection behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_trading.database import connection as db_connection
from ai_trading.database.models import Trade


def test_database_manager_requires_explicit_url(monkeypatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(ValueError, match="DATABASE_URL or an explicit connection string is required"):
        db_connection.DatabaseManager()


def test_connect_handles_timeouterror(monkeypatch) -> None:
    """TimeoutError during engine construction should return False."""
    db = db_connection.DatabaseManager("sqlite:///:memory:")

    def boom(*_args, **_kwargs):
        raise TimeoutError("connect timeout")

    monkeypatch.setattr(db_connection, "create_engine", boom)
    assert db.connect() is False


def test_session_add_query_delete_roundtrip(tmp_path: Path) -> None:
    """Session should perform real insert/query/delete operations."""
    db = db_connection.DatabaseManager(f"sqlite:///{tmp_path / 'roundtrip.db'}")
    assert db.connect() is True

    trade = Trade(symbol="AAPL", side="buy", quantity=2, price=101.5, status="filled")
    with db.get_session() as session:
        session.add(trade)
        session.commit()

    with db.get_session() as session:
        results = session.query(Trade, {"symbol": "AAPL"})
        assert len(results) == 1
        assert results[0].symbol == "AAPL"
        raw = session.execute("SELECT COUNT(*) AS n FROM trades WHERE symbol = :symbol", {"symbol": "AAPL"})
        assert int(raw["rows"][0]["n"]) == 1
        session.delete(results[0])
        session.commit()

    with db.get_session() as session:
        assert session.query(Trade, {"symbol": "AAPL"}) == []
    db.disconnect()


def test_nested_sessions_preserve_same_thread_active_tracking(tmp_path: Path) -> None:
    db = db_connection.DatabaseManager(f"sqlite:///{tmp_path / 'nested.db'}")
    assert db.connect() is True
    try:
        with db.get_session() as outer:
            assert db.get_connection_info()["active_connections"] == 1
            with db.get_session() as inner:
                assert outer.is_active is True
                assert inner.is_active is True
                assert db.get_connection_info()["active_connections"] == 2
            assert outer.is_active is True
            assert db.get_connection_info()["active_connections"] == 1
        assert db.get_connection_info()["active_connections"] == 0
    finally:
        db.disconnect()


def test_initialize_database_global_session_lifecycle(tmp_path: Path) -> None:
    """Global initialize/get_session/shutdown lifecycle should be usable."""
    db_url = f"sqlite:///{tmp_path / 'global.db'}"
    assert db_connection.initialize_database(db_url) is True
    try:
        with db_connection.get_session() as session:
            payload = session.execute("SELECT 1 AS ok")
            assert payload["rows"][0]["ok"] == 1
    finally:
        db_connection.shutdown_database()


def test_get_connection_info_redacts_credentials() -> None:
    db = db_connection.DatabaseManager("postgresql://user:secret@example.com/trading")

    info = db.get_connection_info()

    rendered = str(info["connection_string"])
    assert rendered.startswith("postgresql")
    assert rendered.endswith("@example.com/trading")
    assert "***" in rendered
    assert "secret" not in rendered


def test_connect_fails_closed_when_legacy_schema_missing_without_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(db_connection, "is_test_runtime", lambda: False)
    with pytest.raises(RuntimeError, match="retired outside tests"):
        db_connection.DatabaseManager("sqlite:///:memory:")
