import asyncio

from tests.conftest import reload_module


def _reload_pooling(mod_or_name="ai_trading.http.pooling"):
    pooling = reload_module(mod_or_name)
    pooling.reset_host_semaphores()
    return pooling


async def _max_concurrency(pooling, worker_count: int) -> int:
    semaphore = pooling.get_host_semaphore()
    current = 0
    max_seen = 0

    async def worker() -> None:
        nonlocal current, max_seen
        async with semaphore:
            current += 1
            max_seen = max(max_seen, current)
            await asyncio.sleep(0.01)
            current -= 1

    await asyncio.gather(*(worker() for _ in range(worker_count)))
    return max_seen


async def _semaphore_id(pooling) -> int:
    return id(pooling.get_host_semaphore())


async def _refresh_and_get_id(pooling) -> int:
    return id(pooling.refresh_host_semaphore())


def test_host_limit_enforced(monkeypatch):
    monkeypatch.setenv("AI_TRADING_HOST_LIMIT", "2")
    pooling = _reload_pooling()

    max_seen = asyncio.run(_max_concurrency(pooling, worker_count=5))
    assert max_seen == 2

    monkeypatch.delenv("AI_TRADING_HOST_LIMIT", raising=False)
    pooling = _reload_pooling(pooling)


def test_host_limit_updates_when_env_changes(monkeypatch):
    monkeypatch.setenv("AI_TRADING_HOST_LIMIT", "2")
    pooling = _reload_pooling()

    loop = asyncio.new_event_loop()
    try:
        first_id = loop.run_until_complete(_semaphore_id(pooling))
        assert loop.run_until_complete(_max_concurrency(pooling, worker_count=5)) == 2

        monkeypatch.setenv("AI_TRADING_HOST_LIMIT", "3")
        second_id = loop.run_until_complete(_semaphore_id(pooling))
        assert second_id != first_id
        assert loop.run_until_complete(_max_concurrency(pooling, worker_count=6)) == 3

        monkeypatch.setenv("AI_TRADING_HOST_LIMIT", "4")
        refreshed_id = loop.run_until_complete(_refresh_and_get_id(pooling))
        assert refreshed_id != second_id
        assert loop.run_until_complete(_max_concurrency(pooling, worker_count=7)) == 4
    finally:
        loop.close()

    monkeypatch.delenv("AI_TRADING_HOST_LIMIT", raising=False)
    _reload_pooling(pooling)


def test_host_semaphore_is_scoped_per_event_loop(monkeypatch):
    monkeypatch.setenv("AI_TRADING_HOST_LIMIT", "3")
    pooling = _reload_pooling()

    loop_a = asyncio.new_event_loop()
    loop_b = asyncio.new_event_loop()
    try:
        first_a = loop_a.run_until_complete(_semaphore_id(pooling))
        again_a = loop_a.run_until_complete(_semaphore_id(pooling))
        assert first_a == again_a

        first_b = loop_b.run_until_complete(_semaphore_id(pooling))
        assert first_b != first_a

        assert loop_a.run_until_complete(_max_concurrency(pooling, worker_count=6)) == 3
        assert loop_b.run_until_complete(_max_concurrency(pooling, worker_count=6)) == 3

        monkeypatch.setenv("AI_TRADING_HOST_LIMIT", "5")
        refreshed_a = loop_a.run_until_complete(_semaphore_id(pooling))
        refreshed_b = loop_b.run_until_complete(_semaphore_id(pooling))
        assert refreshed_a != first_a
        assert refreshed_b != first_b
        assert refreshed_a != refreshed_b

        assert loop_a.run_until_complete(_max_concurrency(pooling, worker_count=7)) == 5
        assert loop_b.run_until_complete(_max_concurrency(pooling, worker_count=7)) == 5
    finally:
        loop_a.close()
        loop_b.close()

    monkeypatch.delenv("AI_TRADING_HOST_LIMIT", raising=False)
    _reload_pooling(pooling)


def test_invalidating_limit_cache_refreshes_semaphore(monkeypatch):
    monkeypatch.setenv("AI_TRADING_HOST_LIMIT", "2")
    pooling = _reload_pooling()

    loop = asyncio.new_event_loop()
    try:
        first_id = loop.run_until_complete(_semaphore_id(pooling))
        assert loop.run_until_complete(_max_concurrency(pooling, worker_count=4)) == 2

        monkeypatch.setenv("AI_TRADING_HOST_LIMIT", "5")
        pooling.invalidate_host_limit_cache()

        second_id = loop.run_until_complete(_semaphore_id(pooling))
        assert second_id != first_id
        assert loop.run_until_complete(_max_concurrency(pooling, worker_count=8)) == 5
    finally:
        loop.close()

    monkeypatch.delenv("AI_TRADING_HOST_LIMIT", raising=False)
    _reload_pooling(pooling)
