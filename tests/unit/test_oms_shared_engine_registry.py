from __future__ import annotations

from collections.abc import Generator

import pytest

from ai_trading.oms import event_store as event_store_mod
from ai_trading.oms import intent_store as intent_store_mod
from ai_trading.oms.engine_registry import reset_shared_engines
from ai_trading.oms.event_store import EventStore
from ai_trading.oms.intent_store import IntentStore


pytest.importorskip("sqlalchemy")


@pytest.fixture(autouse=True)
def _reset_shared_engine_registry() -> Generator[None, None, None]:
    reset_shared_engines()
    yield
    reset_shared_engines()


def test_event_store_reuses_shared_postgres_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:pass@db.example.com:5432/postgres")

    created: list[object] = []
    bootstrap_calls = {"create_all": 0, "guards": 0}

    class _BeginCtx:
        def __enter__(self):
            return object()

        def __exit__(self, exc_type, exc, tb):
            return False

    class _Engine:
        def begin(self):
            return _BeginCtx()

        def dispose(self):
            return None

    def _fake_create_engine(url: str, **kwargs):
        engine = _Engine()
        created.append((url, kwargs, engine))
        return engine

    monkeypatch.setattr(event_store_mod, "create_engine", _fake_create_engine)
    monkeypatch.setattr(event_store_mod, "sessionmaker", lambda **_kwargs: object())
    assert event_store_mod._EVENT_METADATA is not None
    monkeypatch.setattr(
        event_store_mod._EVENT_METADATA,
        "create_all",
        lambda _conn, checkfirst=True: bootstrap_calls.__setitem__(
            "create_all",
            int(bootstrap_calls["create_all"]) + 1,
        ),
    )
    monkeypatch.setattr(
        event_store_mod.EventStore,
        "_ensure_append_only_guards",
        lambda self, conn: bootstrap_calls.__setitem__(
            "guards",
            int(bootstrap_calls["guards"]) + 1,
        ),
    )

    first = EventStore()
    second = EventStore()

    assert len(created) == 1
    assert first._engine is second._engine
    assert bootstrap_calls["create_all"] == 1
    assert bootstrap_calls["guards"] == 1
    first.close()
    second.close()
    assert len(created) == 1


def test_intent_store_reuses_shared_postgres_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:pass@db.example.com:5432/postgres")

    created: list[object] = []

    class _BeginCtx:
        def __enter__(self):
            return object()

        def __exit__(self, exc_type, exc, tb):
            return False

    class _Engine:
        def begin(self):
            return _BeginCtx()

        def dispose(self):
            return None

    def _fake_create_engine(url: str, **kwargs):
        engine = _Engine()
        created.append((url, kwargs, engine))
        return engine

    monkeypatch.setattr(intent_store_mod, "create_engine", _fake_create_engine)
    monkeypatch.setattr(intent_store_mod, "sessionmaker", lambda **_kwargs: object())
    assert intent_store_mod._METADATA is not None
    monkeypatch.setattr(
        intent_store_mod._METADATA,
        "create_all",
        lambda _conn, **_kwargs: None,
    )

    first = IntentStore()
    second = IntentStore()

    assert len(created) == 1
    assert first._engine is second._engine
    first.close()
    second.close()
    assert len(created) == 1
