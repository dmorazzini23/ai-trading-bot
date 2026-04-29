import threading

from ai_trading.data.fetch import fallback_concurrency


def test_fallback_slot_releases_acquired_semaphore_after_rebuild(monkeypatch):
    monkeypatch.setattr(fallback_concurrency, "_resolve_limit_from_env", lambda: 1)
    fallback_concurrency.reset_fallback_counters(reset_limit=True)

    with fallback_concurrency.fallback_slot():
        acquired = fallback_concurrency._semaphore
        assert isinstance(acquired, threading.BoundedSemaphore)
        fallback_concurrency.reload_fallback_limit()
        assert fallback_concurrency._semaphore is not acquired

    assert fallback_concurrency.get_active_slots() == 0
    assert acquired.acquire(blocking=False) is True
    acquired.release()
    with fallback_concurrency.fallback_slot():
        assert fallback_concurrency.get_active_slots() == 1
