import asyncio
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
