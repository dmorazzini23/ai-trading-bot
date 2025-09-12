"""HTTP pooling tests with system date set to 2025-01-01 or later.

These tests freeze the clock to ensure the runtime date is compatible with
logic that assumes a post-2024 environment.
"""

import pytest

from ai_trading.utils import http as H
from tests.helpers.dummy_http import DummyResp
from tests.conftest import reload_module

try:  # pragma: no cover - optional dependency
    from freezegun import freeze_time  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    from contextlib import contextmanager

    @contextmanager
    def freeze_time(*_a, **_k):
        yield


@pytest.fixture(autouse=True)
def _freeze_2025(_freeze_clock):
    """Ensure tests run with the system date frozen to 2025-01-01."""
    with freeze_time("2025-01-01", tz_offset=0):
        yield


def test_pool_config_defaults(monkeypatch):
    monkeypatch.delenv("HTTP_MAX_WORKERS", raising=False)
    monkeypatch.delenv("HTTP_MAX_PER_HOST", raising=False)
    st = H.pool_stats()
    assert st["workers"] == 8
    assert st["per_host"] == 6
    assert st["pool_maxsize"] >= st["workers"]


def test_host_semaphore_respects_env(monkeypatch):
    def fake_get(url, timeout=None, headers=None):
        return DummyResp(text="ok")

    monkeypatch.setattr(H, "get", fake_get)
    monkeypatch.setenv("HTTP_MAX_PER_HOST", "3")
    reload_module(H)
    _ = H.map_get(["https://example.com"])
    assert H.pool_stats()["per_host"] == 3

