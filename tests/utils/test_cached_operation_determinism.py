from __future__ import annotations

import ai_trading.utils.performance as performance_mod
from ai_trading.utils.performance import cached_operation


def test_cached_operation_does_not_use_builtin_hash(monkeypatch) -> None:
    calls = {"count": 0}

    @cached_operation(cache_ttl=60)
    def add(a: int, b: int = 0) -> int:
        calls["count"] += 1
        return a + b

    def _boom(*_args, **_kwargs) -> int:
        raise AssertionError("module-level hash() should not be used for cache keys")

    monkeypatch.setattr(performance_mod, "hash", _boom, raising=False)

    assert add(1, b=2) == 3
    assert add(1, b=2) == 3
    assert calls["count"] == 1


def test_cached_operation_cache_key_is_kwarg_order_invariant() -> None:
    calls = {"count": 0}

    @cached_operation(cache_ttl=60)
    def combine(**kwargs: int) -> int:
        calls["count"] += 1
        return int(kwargs["a"]) + int(kwargs["b"])

    assert combine(a=1, b=2) == 3
    assert combine(b=2, a=1) == 3
    assert calls["count"] == 1
