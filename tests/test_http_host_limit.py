import asyncio
import os
import types

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


async def _semaphore_state(pooling) -> tuple[int, int | None, int | None]:
    semaphore = pooling.get_host_semaphore()
    recorded_limit = getattr(semaphore, "_ai_trading_host_limit", None)
    recorded_version = getattr(semaphore, "_ai_trading_host_limit_version", None)
    return id(semaphore), recorded_limit, recorded_version


async def _refresh_and_get_id(pooling) -> int:
    return id(pooling.refresh_host_semaphore())


async def _refresh_state(pooling) -> tuple[int, int | None, int | None]:
    semaphore = pooling.refresh_host_semaphore()
    recorded_limit = getattr(semaphore, "_ai_trading_host_limit", None)
    recorded_version = getattr(semaphore, "_ai_trading_host_limit_version", None)
    return id(semaphore), recorded_limit, recorded_version


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


def test_get_host_semaphore_rebuilds_on_env_update(monkeypatch):
    monkeypatch.setenv("AI_TRADING_HOST_LIMIT", "2")
    pooling = _reload_pooling()

    loop = asyncio.new_event_loop()
    try:
        first_id, first_limit, first_version = loop.run_until_complete(
            _semaphore_state(pooling)
        )
        assert first_limit == 2
        assert isinstance(first_version, int)
        assert loop.run_until_complete(_max_concurrency(pooling, worker_count=5)) == 2

        monkeypatch.setenv("AI_TRADING_HOST_LIMIT", "5")
        second_id, second_limit, second_version = loop.run_until_complete(
            _semaphore_state(pooling)
        )
        assert second_id != first_id
        assert second_limit == 5
        assert isinstance(second_version, int)
        assert second_version != first_version
        assert loop.run_until_complete(_max_concurrency(pooling, worker_count=7)) == 5
    finally:
        loop.close()

    monkeypatch.delenv("AI_TRADING_HOST_LIMIT", raising=False)
    _reload_pooling(pooling)


def test_reload_host_limit_if_env_changed_triggers_refresh(monkeypatch):
    monkeypatch.setenv("AI_TRADING_HOST_LIMIT", "2")
    pooling = _reload_pooling()

    loop = asyncio.new_event_loop()
    try:
        initial_snapshot = pooling.get_host_limit_snapshot()
        first_id = loop.run_until_complete(_semaphore_id(pooling))

        monkeypatch.setenv("AI_TRADING_HOST_LIMIT", "5")
        snapshot = pooling.reload_host_limit_if_env_changed()
        assert snapshot.limit == 5
        assert snapshot.version != initial_snapshot.version

        second_id = loop.run_until_complete(_semaphore_id(pooling))
        assert second_id != first_id
        assert loop.run_until_complete(_max_concurrency(pooling, worker_count=8)) == 5
    finally:
        loop.close()

    monkeypatch.delenv("AI_TRADING_HOST_LIMIT", raising=False)
    _reload_pooling(pooling)


def test_reload_host_limit_refreshes_cache_and_semaphores(monkeypatch):
    monkeypatch.setenv("AI_TRADING_HOST_LIMIT", "2")
    pooling = _reload_pooling()

    loop = asyncio.new_event_loop()
    try:
        first_id, first_limit, first_version = loop.run_until_complete(
            _semaphore_state(pooling)
        )
        assert first_limit == 2
        assert isinstance(first_version, int)

        monkeypatch.setenv("HTTP_MAX_PER_HOST", "6")
        monkeypatch.delenv("AI_TRADING_HOST_LIMIT", raising=False)

        snapshot = pooling.reload_host_limit_if_env_changed()
        assert snapshot.limit == 6
        assert snapshot.version != first_version

        cache = pooling._LIMIT_CACHE
        assert cache is not None
        expected_snapshot = (
            os.getenv("HTTP_MAX_PER_HOST"),
            os.getenv("AI_TRADING_HTTP_HOST_LIMIT"),
            os.getenv("AI_TRADING_HOST_LIMIT"),
        )
        assert cache.env_snapshot == expected_snapshot

        second_id, second_limit, second_version = loop.run_until_complete(
            _semaphore_state(pooling)
        )
        assert second_id != first_id
        assert second_limit == 6
        assert second_version == snapshot.version
    finally:
        loop.close()

    monkeypatch.delenv("HTTP_MAX_PER_HOST", raising=False)
    _reload_pooling(pooling)


def test_refresh_host_semaphore_uses_latest_limit(monkeypatch):
    monkeypatch.setenv("AI_TRADING_HOST_LIMIT", "3")
    pooling = _reload_pooling()

    loop = asyncio.new_event_loop()
    try:
        first_id, first_limit, first_version = loop.run_until_complete(
            _semaphore_state(pooling)
        )
        assert first_limit == 3
        assert isinstance(first_version, int)

        monkeypatch.setenv("AI_TRADING_HOST_LIMIT", "6")
        refreshed_id, refreshed_limit, refreshed_version = loop.run_until_complete(
            _refresh_state(pooling)
        )
        assert refreshed_id != first_id
        assert refreshed_limit == 6
        assert isinstance(refreshed_version, int)
        assert refreshed_version != first_version
        assert loop.run_until_complete(_max_concurrency(pooling, worker_count=8)) == 6
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


def test_http_host_limit_controller_legacy_env(monkeypatch):
    monkeypatch.delenv("AI_TRADING_HTTP_HOST_LIMIT", raising=False)
    monkeypatch.setenv("AI_TRADING_HOST_LIMIT", "5")

    http = reload_module("ai_trading.net.http")

    class _StubAdapter:
        def __init__(self, *, max_retries, pool_connections, pool_maxsize):
            self.max_retries = max_retries
            self.pool_connections = pool_connections
            self.pool_maxsize = pool_maxsize

    monkeypatch.setattr(http, "HTTPAdapter", _StubAdapter, raising=False)

    import types

    class _StubSession:
        def __init__(self):
            self.adapters = {"https://": types.SimpleNamespace(max_retries="sentinel")}

        def mount(self, prefix, adapter):
            self.adapters[prefix] = adapter

    session = _StubSession()

    http._HOST_LIMIT_CONTROLLER.apply(session)

    adapter = session.adapters["https://"]
    assert adapter.pool_connections == 5
    assert adapter.pool_maxsize == 5

    monkeypatch.setenv("AI_TRADING_HTTP_HOST_LIMIT", "3")
    monkeypatch.setenv("AI_TRADING_HOST_LIMIT", "6")

    http._HOST_LIMIT_CONTROLLER.reload_if_changed(session)

    adapter = session.adapters["https://"]
    assert adapter.pool_connections == 3
    assert adapter.pool_maxsize == 3


def test_switching_between_env_keys_refreshes_semaphore(monkeypatch):
    monkeypatch.setenv("AI_TRADING_HOST_LIMIT", "4")
    monkeypatch.delenv("AI_TRADING_HTTP_HOST_LIMIT", raising=False)
    monkeypatch.delenv("HTTP_MAX_PER_HOST", raising=False)
    pooling = _reload_pooling()

    loop = asyncio.new_event_loop()
    try:
        first_id = loop.run_until_complete(_semaphore_id(pooling))
        assert loop.run_until_complete(_max_concurrency(pooling, worker_count=6)) == 4

        monkeypatch.delenv("AI_TRADING_HOST_LIMIT", raising=False)
        monkeypatch.setenv("AI_TRADING_HTTP_HOST_LIMIT", "4")

        second_id = loop.run_until_complete(_semaphore_id(pooling))
        assert second_id != first_id
        assert loop.run_until_complete(_max_concurrency(pooling, worker_count=6)) == 4
    finally:
        loop.close()

    monkeypatch.delenv("AI_TRADING_HTTP_HOST_LIMIT", raising=False)
    _reload_pooling(pooling)


def test_config_changes_refresh_semaphore(monkeypatch):
    monkeypatch.delenv("AI_TRADING_HOST_LIMIT", raising=False)
    monkeypatch.delenv("AI_TRADING_HTTP_HOST_LIMIT", raising=False)
    monkeypatch.delenv("HTTP_MAX_PER_HOST", raising=False)

    pooling = _reload_pooling()

    cfg = types.SimpleNamespace(host_concurrency_limit=2)
    monkeypatch.setattr(pooling.config, "get_trading_config", lambda: cfg)

    loop = asyncio.new_event_loop()
    try:
        first_id = loop.run_until_complete(_semaphore_id(pooling))
        assert loop.run_until_complete(_max_concurrency(pooling, worker_count=4)) == 2

        cfg.host_concurrency_limit = 6
        second_id = loop.run_until_complete(_semaphore_id(pooling))
        assert second_id != first_id
        assert loop.run_until_complete(_max_concurrency(pooling, worker_count=8)) == 6

        cfg.host_concurrency_limit = "invalid"
        third_id = loop.run_until_complete(_semaphore_id(pooling))
        assert third_id != second_id
        assert loop.run_until_complete(_max_concurrency(pooling, worker_count=9)) == pooling._DEFAULT_LIMIT
    finally:
        loop.close()

    _reload_pooling(pooling)


def test_fallback_concurrency_shares_pooling_host_limit(monkeypatch):
    from ai_trading.data import fallback_concurrency as legacy_concurrency
    from ai_trading.data.fallback import concurrency as modern_concurrency

    monkeypatch.setenv("AI_TRADING_HOST_LIMIT", "3")
    pooling = _reload_pooling()

    assert legacy_concurrency is modern_concurrency

    legacy_concurrency.reset_tracking_state()
    legacy_concurrency.reset_peak_simultaneous_workers()

    original_state = getattr(legacy_concurrency, "_POOLING_LIMIT_STATE", None)
    legacy_concurrency._POOLING_LIMIT_STATE = None

    try:
        snapshot = pooling.reload_host_limit_if_env_changed()
        assert snapshot.limit == 3

        async def _exercise() -> tuple[int | None, int | None]:
            limit = legacy_concurrency._get_effective_host_limit()
            semaphore = legacy_concurrency._get_host_limit_semaphore()
            assert isinstance(semaphore, asyncio.Semaphore)
            recorded_limit = getattr(semaphore, "_ai_trading_host_limit", None)
            return limit, recorded_limit

        loop = asyncio.new_event_loop()
        try:
            limit, recorded_limit = loop.run_until_complete(_exercise())
        finally:
            loop.close()

        assert limit == 3
        assert recorded_limit == 3
    finally:
        legacy_concurrency._POOLING_LIMIT_STATE = original_state
        monkeypatch.delenv("AI_TRADING_HOST_LIMIT", raising=False)
        _reload_pooling(pooling)


def test_pooling_reload_invalidates_fallback_snapshot(monkeypatch):
    from ai_trading.data.fallback import concurrency as fallback

    original_state = getattr(fallback, "_POOLING_LIMIT_STATE", None)

    monkeypatch.setenv("AI_TRADING_HOST_LIMIT", "4")
    pooling = _reload_pooling()

    async def _prime() -> None:
        async def worker(symbol: str) -> str:
            await asyncio.sleep(0)
            return symbol

        await fallback.run_with_concurrency(["AA", "BB"], worker, max_concurrency=6)

    asyncio.run(_prime())

    assert getattr(fallback, "_POOLING_LIMIT_STATE", None) is not None
    assert fallback._POOLING_LIMIT_STATE[0] == 4  # type: ignore[index]

    monkeypatch.setenv("AI_TRADING_HOST_LIMIT", "2")
    pooling = _reload_pooling(pooling)

    assert getattr(fallback, "_POOLING_LIMIT_STATE", None) is None

    limit_after_reload = fallback._get_effective_host_limit()
    assert limit_after_reload == 2

    asyncio.run(_prime())

    refreshed_state = getattr(fallback, "_POOLING_LIMIT_STATE", None)
    assert refreshed_state is not None
    assert refreshed_state[0] == 2

    fallback.reset_tracking_state()
    fallback.reset_peak_simultaneous_workers()
    monkeypatch.delenv("AI_TRADING_HOST_LIMIT", raising=False)
    try:
        if original_state is None:
            fallback._POOLING_LIMIT_STATE = None
        else:
            fallback._POOLING_LIMIT_STATE = original_state
    finally:
        _reload_pooling(pooling)
