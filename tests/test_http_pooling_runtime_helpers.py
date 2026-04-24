from __future__ import annotations

import asyncio
import threading
import types

from ai_trading.http import pooling


def _reset_pooling_state() -> None:
    pooling.testing_reset_host_limits()
    pooling._LIMIT_CACHE = None
    pooling._LIMIT_VERSION = 0
    pooling._LAST_LIMIT_ENV_SNAPSHOT = None
    pooling._HOST_SEMAPHORES.clear()
    pooling._HOST_LIMITERS.clear()
    pooling._RETIRED_SEMAPHORES.clear()


def test_pooling_state_normalization_accepts_tuples_and_objects():
    assert pooling._normalise_pooling_state((0, "7")) == (1, 7)
    assert pooling._normalise_pooling_state(types.SimpleNamespace(limit="4", version=9)) == (4, 9)
    assert pooling._normalise_pooling_state(types.SimpleNamespace(limit="bad", version=9)) is None
    assert pooling._normalise_pooling_state(object()) is None
    assert pooling._normalise_pooling_state(None) is None


def test_host_key_normalization_and_override_resolution(monkeypatch):
    values = {
        "HTTP_RPS_LIMIT_API_ALPACA_MARKETS": "0",
        "HTTP_RPS_LIMIT_DEFAULT": "5",
    }
    monkeypatch.setattr(pooling, "_env_raw", lambda key, default=None: values.get(key, default))

    assert pooling._normalize_host(" API.Alpaca.Markets ") == "api.alpaca.markets"
    assert pooling._normalize_host("") == pooling._DEFAULT_HOST_KEY
    assert pooling._build_host_override_key("api.alpaca-markets") == (
        "HTTP_RPS_LIMIT_API_ALPACA_MARKETS"
    )
    assert pooling._build_host_override_key("...") == "HTTP_RPS_LIMIT_DEFAULT"
    assert pooling._resolve_host_override_limit("api.alpaca-markets") == 1
    assert pooling._resolve_host_override_limit("") == 5

    values["HTTP_RPS_LIMIT_API_ALPACA_MARKETS"] = "bad"
    assert pooling._resolve_host_override_limit("api.alpaca-markets") is None


def test_compute_limit_env_priority_and_config_fallback(monkeypatch):
    monkeypatch.setattr(pooling, "_env_raw", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pooling.config, "get_env", lambda *_args, **_kwargs: "6")

    assert pooling._compute_limit("0") == 1
    assert pooling._compute_limit("bad") == 6

    def raise_config(*_args, **_kwargs):
        raise RuntimeError("config unavailable")

    monkeypatch.setattr(pooling.config, "get_env", raise_config)
    monkeypatch.setattr(pooling, "AI_TRADING_FALLBACK_EXCEPTIONS", (RuntimeError,))
    assert pooling._compute_limit("bad") == pooling._DEFAULT_LIMIT


def test_read_limit_source_prefers_env_then_config(monkeypatch):
    env_values = {
        "AI_TRADING_HOST_LIMIT": None,
        "AI_TRADING_HTTP_HOST_LIMIT": "8",
        "HTTP_MAX_WORKERS": "9",
        "HTTP_MAX_PER_HOST": "10",
    }
    monkeypatch.setattr(pooling, "_env_raw", lambda key, default=None: env_values.get(key, default))
    monkeypatch.setattr(pooling.config, "get_trading_config", lambda: None)

    assert pooling._read_limit_source((None, "8", "10")) == (
        8,
        "AI_TRADING_HTTP_HOST_LIMIT",
        "8",
        None,
    )

    env_values.update(dict.fromkeys(env_values, None))
    cfg = types.SimpleNamespace(host_concurrency_limit="11")
    monkeypatch.setattr(pooling.config, "get_trading_config", lambda: cfg)
    limit, env_key, raw_env, config_id = pooling._read_limit_source((None, None, None))
    assert (limit, env_key, raw_env, config_id) == (11, None, None, id(cfg))

    cfg.host_concurrency_limit = "bad"
    limit, *_ = pooling._read_limit_source((None, None, None))
    assert limit == pooling._DEFAULT_LIMIT


def test_sync_limit_cache_from_pooling_advances_versions(monkeypatch):
    _reset_pooling_state()
    monkeypatch.setattr(pooling, "_current_env_snapshot", lambda: (None, None, None))

    first = pooling._sync_limit_cache_from_pooling(0, 1)
    second = pooling._sync_limit_cache_from_pooling(4, 1)

    assert first == pooling.HostLimitSnapshot(1, 1)
    assert second == pooling.HostLimitSnapshot(4, 2)
    assert pooling._LIMIT_CACHE is not None
    assert pooling._LIMIT_CACHE.limit == 4


def test_reset_host_semaphores_preserves_or_rebuilds_pooling_state(monkeypatch):
    _reset_pooling_state()
    captured: list[tuple[int, int]] = []
    monkeypatch.setattr(
        pooling,
        "_set_pooling_limit_state",
        lambda limit, version: captured.append((limit, version)),
    )
    pooling._LIMIT_CACHE = pooling._ResolvedLimitCache(
        env_key="AI_TRADING_HOST_LIMIT",
        raw_env="3",
        limit=3,
        version=4,
        config_id=None,
        env_snapshot=(None, None, None),
    )
    pooling._LIMIT_VERSION = 4

    pooling.reset_host_semaphores(clear_limit_cache=False)

    assert pooling._LIMIT_CACHE is not None
    assert pooling._LIMIT_CACHE.version == 5
    assert captured[-1] == (3, 5)

    pooling._LIMIT_CACHE = None
    pooling._LIMIT_VERSION = 6
    monkeypatch.setattr(pooling, "_get_pooling_limit_state", lambda: (2, 5))
    pooling.reset_host_semaphores(clear_limit_cache=False, bump_version=False)
    assert captured[-1] == (2, 6)


def test_get_host_map_and_purge_closed_loops():
    _reset_pooling_state()
    loop = asyncio.new_event_loop()
    closed_loop = asyncio.new_event_loop()
    try:
        host_map = pooling._get_host_map(loop)
        assert pooling._get_host_map(loop) is host_map
        pooling._HOST_SEMAPHORES[closed_loop] = {}
        closed_loop.close()
        pooling._purge_closed_loops()
        assert closed_loop not in pooling._HOST_SEMAPHORES
    finally:
        loop.close()


def test_loop_semaphore_rebuilds_for_stale_versions_and_limits(monkeypatch):
    _reset_pooling_state()
    loop = asyncio.new_event_loop()
    monkeypatch.setattr(pooling, "_resolve_host_override_limit", lambda host: 2 if host == "api" else None)
    try:
        first = pooling._get_or_create_loop_semaphore(
            loop,
            "api",
            pooling.HostLimitSnapshot(4, 1),
        )
        second = pooling._get_or_create_loop_semaphore(
            loop,
            "other",
            pooling.HostLimitSnapshot(4, 2),
        )
        third = pooling._get_or_create_loop_semaphore(
            loop,
            "api",
            pooling.HostLimitSnapshot(5, 2),
        )

        assert getattr(first, "_ai_trading_host_limit") == 2
        assert getattr(second, "_ai_trading_host_limit") == 4
        assert third is not first
        assert len(pooling._RETIRED_SEMAPHORES) >= 1
    finally:
        loop.close()


def test_get_host_semaphore_uses_pooling_state_when_newer(monkeypatch):
    _reset_pooling_state()
    monkeypatch.setattr(pooling, "reload_host_limit_if_env_changed", lambda: pooling.HostLimitSnapshot(3, 2))
    monkeypatch.setattr(pooling, "_get_pooling_limit_state", lambda: (5, 4))
    captured: list[tuple[int, int]] = []
    monkeypatch.setattr(
        pooling,
        "_set_pooling_limit_state",
        lambda limit, version: captured.append((limit, version)),
    )

    async def run():
        sem = pooling.get_host_semaphore("api.alpaca.markets")
        return getattr(sem, "_ai_trading_host_limit"), getattr(
            sem,
            "_ai_trading_host_limit_version",
        )

    assert asyncio.run(run()) == (5, 4)
    assert pooling._LIMIT_CACHE is not None
    assert pooling._LIMIT_CACHE.limit == 5
    assert captured == []


def test_refresh_host_semaphore_updates_retired_and_pooling_state(monkeypatch):
    _reset_pooling_state()
    loop = asyncio.new_event_loop()
    captured: list[tuple[int, int]] = []
    monkeypatch.setattr(
        pooling,
        "_set_pooling_limit_state",
        lambda limit, version: captured.append((limit, version)),
    )
    try:
        first = pooling.refresh_host_semaphore(
            "api",
            loop=loop,
            snapshot=pooling.HostLimitSnapshot(3, 1),
        )
        second = pooling.refresh_host_semaphore(
            "api",
            loop=loop,
            snapshot=pooling.HostLimitSnapshot(4, 2),
        )

        assert second is not first
        assert pooling._RETIRED_SEMAPHORES[-1] is first
        assert getattr(second, "_ai_trading_host_limit") == 4
        assert captured[-1] == (4, 2)
    finally:
        loop.close()


def test_async_host_limiter_from_url_acquires_and_releases(monkeypatch):
    _reset_pooling_state()
    sem = asyncio.Semaphore(1)
    calls: list[str] = []

    def fake_get_host_semaphore(hostname=None):
        calls.append(hostname)
        return sem

    monkeypatch.setattr(pooling, "get_host_semaphore", fake_get_host_semaphore)

    async def run():
        limiter = pooling.limit_url("https://API.Alpaca.Markets/v2/orders")
        async with limiter as acquired:
            assert acquired is limiter
            assert limiter._acquired is True
            assert sem.locked() is True
        assert limiter._acquired is False
        assert sem.locked() is False

    asyncio.run(run())
    assert calls == ["api.alpaca.markets"]


def test_async_host_limiter_cleans_up_failed_acquire(monkeypatch):
    class BadSemaphore:
        async def acquire(self):
            raise RuntimeError("blocked")

    monkeypatch.setattr(pooling, "get_host_semaphore", lambda _host: BadSemaphore())
    limiter = pooling.AsyncHostLimiter("api")

    async def run():
        try:
            await limiter.__aenter__()
        except RuntimeError:
            return
        raise AssertionError("expected acquire failure")

    asyncio.run(run())
    assert limiter._semaphore is None
    assert limiter._acquired is False


def test_sync_host_limiter_releases_even_if_release_raises(monkeypatch):
    events: list[str] = []

    class Limiter:
        def acquire(self):
            events.append("acquire")

        def release(self):
            events.append("release")
            raise ValueError("already released")

    monkeypatch.setattr(pooling, "get_host_limiter", lambda host: Limiter())

    with pooling.host_limiter("API.Alpaca.Markets"):
        events.append("inside")

    assert events == ["acquire", "inside", "release"]


def test_threading_host_limiter_refreshes_when_limit_changes(monkeypatch):
    _reset_pooling_state()
    default_limit = {"value": "2"}
    override: dict[str, int | None] = {"value": None}
    monkeypatch.setattr(
        pooling,
        "_env_raw",
        lambda key, default=None: default_limit["value"]
        if key == "AI_TRADING_HTTP_HOST_LIMIT"
        else default,
    )
    monkeypatch.setattr(pooling, "_resolve_host_override_limit", lambda _host: override["value"])

    first = pooling.get_host_limiter("API.Alpaca.Markets")
    assert isinstance(first, threading.Semaphore)
    assert getattr(first, "_ai_trading_host_limit") == 2

    default_limit["value"] = "4"
    second = pooling.get_host_limiter("API.Alpaca.Markets")
    assert second is not first
    assert getattr(second, "_ai_trading_host_limit") == 4

    override["value"] = 1
    third = pooling.get_host_limiter("API.Alpaca.Markets")
    assert third is not second
    assert getattr(third, "_ai_trading_host_limit") == 1
