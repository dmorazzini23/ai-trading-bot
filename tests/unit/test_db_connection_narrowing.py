"""Tests for SQLAlchemy-backed database connection behavior."""

from __future__ import annotations

from pathlib import Path

from ai_trading.database import connection as db_connection
from ai_trading.database.models import Trade


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
