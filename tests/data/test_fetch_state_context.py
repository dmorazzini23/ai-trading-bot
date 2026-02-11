from __future__ import annotations

import inspect
import threading
from datetime import UTC, datetime
from types import SimpleNamespace

from ai_trading.data import fetch


def test_gap_ratio_state_is_isolated_per_thread() -> None:
    """Per-call fetch state should not bleed across concurrent threads."""

    original_state = dict(getattr(fetch, "_state", {}) or {})
    barrier = threading.Barrier(2)
    results: dict[str, float | None] = {}
    errors: list[Exception] = []

    def _worker(name: str, ratio: float) -> None:
        try:
            fetch._set_fetch_state({})
            fetch._record_gap_ratio_state(ratio, metadata={"gap_ratio": ratio})
            barrier.wait(timeout=2)
            results[name] = fetch._current_gap_ratio()
        except Exception as exc:  # pragma: no cover - defensive thread capture
            errors.append(exc)

    t1 = threading.Thread(target=_worker, args=("one", 0.1))
    t2 = threading.Thread(target=_worker, args=("two", 0.2))
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    fetch._set_fetch_state(original_state)

    assert not errors
    assert results["one"] == 0.1
    assert results["two"] == 0.2


def test_get_minute_df_uses_call_local_state(monkeypatch) -> None:
    """Minute fetch should keep call-local ``_state`` even under concurrency."""

    class _StopMinuteFetch(Exception):
        pass

    barrier = threading.Barrier(2)
    seen_local_state_ids: dict[str, int] = {}
    errors: list[Exception] = []

    monkeypatch.setattr(fetch, "_ensure_pandas", lambda: __import__("pandas"))
    monkeypatch.setattr(fetch, "_detect_pytest_env", lambda: False)
    monkeypatch.setattr(fetch, "_env_source_override", lambda *_a, **_k: None)
    monkeypatch.setattr(
        fetch,
        "_last_complete_minute",
        lambda _pd: datetime(2024, 1, 2, 16, 0, tzinfo=UTC),
    )
    monkeypatch.setattr(fetch, "_used_fallback", lambda *_a, **_k: False)
    monkeypatch.setattr(fetch, "_window_has_trading_session", lambda *_a, **_k: True)
    monkeypatch.setattr(fetch, "_http_fallback_permitted", lambda *_a, **_k: True)
    monkeypatch.setattr(fetch, "_ensure_override_state_current", lambda: None)
    monkeypatch.setattr(fetch, "_normalize_feed_value", lambda value: value)

    def _resolve_backup_provider_stub():
        frame = inspect.currentframe()
        caller = frame.f_back if frame is not None else None
        local_state = caller.f_locals.get("_state") if caller is not None else None
        if not isinstance(local_state, dict):
            raise AssertionError("expected call-local _state dict")
        seen_local_state_ids[threading.current_thread().name] = id(local_state)
        try:
            barrier.wait(timeout=2)
        except threading.BrokenBarrierError:
            pass
        raise _StopMinuteFetch()

    monkeypatch.setattr(fetch, "_resolve_backup_provider", _resolve_backup_provider_stub)

    start = datetime(2024, 1, 2, 15, 0, tzinfo=UTC)
    end = datetime(2024, 1, 2, 15, 30, tzinfo=UTC)

    def _worker(name: str) -> None:
        try:
            fetch.get_minute_df(name, start, end, feed="iex")
        except _StopMinuteFetch:
            return
        except Exception as exc:  # pragma: no cover - defensive capture
            errors.append(exc)

    t1 = threading.Thread(target=_worker, args=("AAPL",), name="fetch-thread-1")
    t2 = threading.Thread(target=_worker, args=("MSFT",), name="fetch-thread-2")
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert not errors
    assert set(seen_local_state_ids) == {"fetch-thread-1", "fetch-thread-2"}
    assert len(set(seen_local_state_ids.values())) == 2


def test_record_session_last_request_is_isolated_per_thread() -> None:
    """Request bookkeeping should write into each thread-local fetch state."""

    original_state = dict(getattr(fetch, "_state", {}) or {})
    barrier = threading.Barrier(2)
    results: dict[str, list[str]] = {}
    errors: list[Exception] = []

    def _worker(name: str, feed: str) -> None:
        try:
            local_state: dict[str, object] = {}
            fetch._set_fetch_state(local_state)
            barrier.wait(timeout=2)
            fetch._record_session_last_request(
                SimpleNamespace(),
                "GET",
                "https://example.com/v2/stocks/bars",
                {"feed": feed},
                {},
            )
            calls = local_state.get("calls") if isinstance(local_state, dict) else None
            if isinstance(calls, dict):
                feeds = calls.get("feeds")
                if isinstance(feeds, list):
                    results[name] = [str(value) for value in feeds]
                else:
                    results[name] = []
            else:
                results[name] = []
        except Exception as exc:  # pragma: no cover - defensive thread capture
            errors.append(exc)

    t1 = threading.Thread(target=_worker, args=("one", "iex"))
    t2 = threading.Thread(target=_worker, args=("two", "sip"))
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    fetch._set_fetch_state(original_state)

    assert not errors
    assert results["one"] == ["iex"]
    assert results["two"] == ["sip"]
