from __future__ import annotations

import builtins
import importlib.util
import logging
import sys
import time
from pathlib import Path


def _load_module_without_cachetools(monkeypatch):
    """Load the idempotency module while forcing cachetools ImportError."""

    monkeypatch.delitem(sys.modules, "cachetools", raising=False)
    original_import = builtins.__import__

    def _raise_for_cachetools(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: D401
        if name == "cachetools" or name.startswith("cachetools."):
            raise ImportError("cachetools missing for fallback test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _raise_for_cachetools)

    module_path = Path(__file__).resolve().parents[1] / "ai_trading" / "execution" / "idempotency.py"
    spec = importlib.util.spec_from_file_location("ai_trading.execution.idempotency_fallback", module_path)
    assert spec and spec.loader is not None, "module specification must be available"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_order_idempotency_cache_fallback(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    module = _load_module_without_cachetools(monkeypatch)

    assert any("cachetools not available" in record.message for record in caplog.records), (
        "Fallback import should emit a diagnostic message"
    )

    cache = module.OrderIdempotencyCache(ttl_seconds=0.1, max_size=10)
    key = cache.generate_key("AAPL", "buy", 1)

    assert cache.is_duplicate(key) is False
    cache.mark_submitted(key, "OID-1")
    assert cache.is_duplicate(key) is True
    assert cache.get_existing_order(key)["order_id"] == "OID-1"

    time.sleep(0.15)
    assert cache.get_existing_order(key) is None
    assert cache.is_duplicate(key) is False

    cache.mark_submitted(key, "OID-2")
    time.sleep(0.15)
    cleared = cache.clear_expired()
    assert cleared >= 1
    assert cache.stats()["size"] == 0

    size_cache = module.OrderIdempotencyCache(ttl_seconds=5, max_size=2)
    first = size_cache.generate_key("AAA", "buy", 1)
    second = size_cache.generate_key("BBB", "sell", 2)
    third = size_cache.generate_key("CCC", "buy", 3)

    size_cache.mark_submitted(first, "FIRST")
    size_cache.mark_submitted(second, "SECOND")
    size_cache.mark_submitted(third, "THIRD")

    assert size_cache.is_duplicate(second) is True
    assert size_cache.is_duplicate(third) is True
    assert size_cache.is_duplicate(first) is False, "LRU eviction should remove the oldest entry"
    assert size_cache.stats()["size"] <= 2


def test_fallback_cache_expires_entries_on_get(monkeypatch):
    module = _load_module_without_cachetools(monkeypatch)

    time_state = {"now": 0.0}

    def _fake_monotonic() -> float:
        return time_state["now"]

    monkeypatch.setattr(module, "monotonic_time", _fake_monotonic)

    cache = module.OrderIdempotencyCache(ttl_seconds=1, max_size=10)
    key = cache.generate_key("MSFT", "buy", 5)

    cache.mark_submitted(key, "ORDER-1")
    assert cache.get_existing_order(key)["order_id"] == "ORDER-1"

    time_state["now"] += 2

    assert cache.get_existing_order(key) is None
    assert cache.is_duplicate(key) is False


def test_fallback_ttl_cache_contains_prunes_expired_entries(monkeypatch):
    module = _load_module_without_cachetools(monkeypatch)

    time_state = {"now": 0.0}

    def _fake_monotonic() -> float:
        return time_state["now"]

    monkeypatch.setattr(module, "monotonic_time", _fake_monotonic)

    cache = module.TTLCache(maxsize=10, ttl=1)
    cache["foo"] = "bar"

    assert "foo" in cache

    time_state["now"] = 2.0

    assert ("foo" in cache) is False
    assert "foo" not in cache
    assert len(cache) == 0


def test_fallback_ttl_cache_get_prunes_expired_entries(monkeypatch):
    module = _load_module_without_cachetools(monkeypatch)

    time_state = {"now": 0.0}

    def _fake_monotonic() -> float:
        return time_state["now"]

    monkeypatch.setattr(module, "monotonic_time", _fake_monotonic)

    cache = module.TTLCache(maxsize=10, ttl=1)
    cache["foo"] = "bar"

    assert cache.get("foo") == "bar"

    time_state["now"] = 2.0

    assert cache.get("foo") is None
    assert cache.get("foo") is None  # ensure the entry stays absent on repeated lookup
    assert len(cache) == 0
