from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ai_trading.oms import event_store, intent_store


class _BeginContext:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def __enter__(self) -> Any:
        return self._conn

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False


class _Engine:
    def __init__(self, conn: Any) -> None:
        self.conn = conn
        self.begins = 0

    def begin(self) -> _BeginContext:
        self.begins += 1
        return _BeginContext(self.conn)


def test_intent_store_postgres_bootstrap_creates_tables_before_marking_bootstrapped(
    monkeypatch,
) -> None:
    calls: list[tuple[str, Any]] = []
    conn = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))
    engine = _Engine(conn)

    class _Metadata:
        def create_all(self, bind: Any, checkfirst: bool = True) -> None:
            calls.append(("create_all", bind, checkfirst))

    monkeypatch.setattr(intent_store, "_METADATA", _Metadata())
    monkeypatch.setattr(intent_store, "_BOOTSTRAPPED_DATABASE_URLS", set())
    store = object.__new__(intent_store.IntentStore)
    store._database_url = "postgresql+psycopg://oms"
    store._engine = engine

    store._bootstrap()

    assert calls == [("create_all", conn, True)]
    assert "postgresql+psycopg://oms" in intent_store._BOOTSTRAPPED_DATABASE_URLS
    assert engine.begins == 1


def test_event_store_postgres_bootstrap_creates_tables_before_triggers(monkeypatch) -> None:
    calls: list[str] = []
    conn = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))
    engine = _Engine(conn)

    class _Metadata:
        def create_all(self, bind: Any, checkfirst: bool = True) -> None:
            assert bind is conn
            assert checkfirst is True
            calls.append("create_all")

    def _ensure_append_only_guards(self: Any, bind: Any) -> None:
        assert bind is conn
        calls.append("guards")

    monkeypatch.setattr(event_store, "_EVENT_METADATA", _Metadata())
    monkeypatch.setattr(event_store, "_BOOTSTRAPPED_DATABASE_URLS", set())
    monkeypatch.setattr(
        event_store.EventStore,
        "_ensure_append_only_guards",
        _ensure_append_only_guards,
    )
    store = object.__new__(event_store.EventStore)
    store._database_url = "postgresql+psycopg://oms"
    store._engine = engine

    store._bootstrap()

    assert calls == ["create_all", "guards"]
    assert "postgresql+psycopg://oms" in event_store._BOOTSTRAPPED_DATABASE_URLS
    assert engine.begins == 1
