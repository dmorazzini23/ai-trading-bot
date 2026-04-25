from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import ai_trading.net.http_host_limit as http_host_limit


@pytest.fixture(autouse=True)
def _reset_counters(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AI_TRADING_FALLBACK_PEAK_PATH", str(tmp_path / "peak.json"))
    monkeypatch.setattr(http_host_limit, "_IN_FLIGHT", 0)
    monkeypatch.setattr(http_host_limit, "_PEAK", 0)
    http_host_limit._FALLBACK_SYNC_LIMITERS.clear()
    http_host_limit._FALLBACK_ASYNC_LIMITERS.clear()


def test_peak_path_load_persist_and_record_peak(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    peak_path = tmp_path / "nested" / "peak.json"
    monkeypatch.setenv("AI_TRADING_FALLBACK_PEAK_PATH", str(peak_path))

    assert http_host_limit._load_peak_from_disk() == 0
    http_host_limit.record_peak(0)
    assert not peak_path.exists()

    http_host_limit.record_peak(3)
    assert http_host_limit.current_peak() == 3
    assert http_host_limit._load_peak_from_disk() == 3

    peak_path.write_text('{"peak": "bad"}')
    assert http_host_limit._load_peak_from_disk() == 0
    peak_path.write_text("not json")
    assert http_host_limit._load_peak_from_disk() == 0


def test_peak_path_and_persist_failure_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        http_host_limit,
        "get_env",
        lambda *_args, **_kwargs: object(),
    )
    assert http_host_limit._peak_path() == Path(http_host_limit._DEFAULT_PEAK_PATH)

    monkeypatch.setattr(
        http_host_limit,
        "get_env",
        lambda *_args, **_kwargs: str(tmp_path / "peak.json"),
    )
    monkeypatch.setattr(Path, "write_text", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk")))

    http_host_limit._persist_peak(5)
    assert not (tmp_path / "peak.json").exists()


def test_fallback_limit_host_normalization_and_sync_limiter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        http_host_limit,
        "get_env",
        lambda key, default=None, **_kwargs: "bad" if key == "AI_TRADING_HOST_LIMIT" else default,
    )
    monkeypatch.setattr(http_host_limit.pooling, "_DEFAULT_LIMIT", 4, raising=False)
    assert http_host_limit._resolve_fallback_limit() == 4

    monkeypatch.setattr(
        http_host_limit,
        "get_env",
        lambda key, default=None, **_kwargs: "2" if key == "HTTP_MAX_PER_HOST" else default,
    )
    monkeypatch.setattr(http_host_limit.pooling, "_normalize_host", lambda _host: (_ for _ in ()).throw(RuntimeError("bad")), raising=False)
    monkeypatch.setattr(http_host_limit.pooling, "get_host_limiter", lambda _host: (_ for _ in ()).throw(RuntimeError("bad")), raising=False)

    assert http_host_limit._normalize_host(" Example.COM ") == "example.com"
    semaphore = http_host_limit._get_sync_semaphore("example.com")
    assert semaphore is http_host_limit._get_sync_semaphore("example.com")

    with http_host_limit.host_limiter("example.com"):
        assert http_host_limit.current_inflight() == 1
    assert http_host_limit.current_inflight() == 0


def test_async_limiter_fallback_and_custom_limiter_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenReleaseSemaphore:
        async def acquire(self) -> bool:
            return True

        def release(self) -> None:
            raise ValueError("over release")

    monkeypatch.setattr(
        http_host_limit,
        "_get_async_semaphore",
        lambda _host: BrokenReleaseSemaphore(),
    )

    async def _fallback() -> None:
        async with http_host_limit._FallbackAsyncLimiter("example.com"):
            assert http_host_limit.current_inflight() == 1
        assert http_host_limit.current_inflight() == 0

    asyncio.run(_fallback())

    class InnerLimiter:
        entered = False
        exited = False

        async def __aenter__(self) -> "InnerLimiter":
            self.entered = True
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            self.exited = True

    monkeypatch.setattr(http_host_limit.pooling, "AsyncHostLimiter", lambda _host: InnerLimiter(), raising=False)

    async def _tracked() -> None:
        async with http_host_limit.host_limiter_async("example.com"):
            assert http_host_limit.current_inflight() == 1
        assert http_host_limit.current_inflight() == 0

    asyncio.run(_tracked())


def test_reload_with_peak_file_uses_disk_value(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    peak_path = tmp_path / "peak.json"
    peak_path.write_text('{"peak": 7}')
    monkeypatch.setenv("AI_TRADING_FALLBACK_PEAK_PATH", str(peak_path))

    reloaded = importlib.reload(http_host_limit)
    try:
        assert reloaded.current_peak() == 7
    finally:
        importlib.reload(http_host_limit)
