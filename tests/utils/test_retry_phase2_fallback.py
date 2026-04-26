from __future__ import annotations

import builtins
import importlib
import sys
from types import ModuleType

import pytest


def _import_retry_without_tenacity(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tenacity":
            raise ImportError("tenacity not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "tenacity", raising=False)
    monkeypatch.delitem(sys.modules, "ai_trading.utils.retry", raising=False)
    return importlib.import_module("ai_trading.utils.retry")


def test_fallback_retry_fixed_mode_succeeds_after_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    retry_mod = _import_retry_without_tenacity(monkeypatch)
    sleeps: list[float] = []
    calls = {"count": 0}
    monkeypatch.setattr(retry_mod.time, "sleep", lambda seconds: sleeps.append(seconds))

    @retry_mod.retry(retries=3, delay=0.5, mode="fixed", exceptions=(ValueError,))
    def flaky() -> str:
        calls["count"] += 1
        if calls["count"] < 3:
            raise ValueError("temporary")
        return "ok"

    assert flaky() == "ok"
    assert calls["count"] == 3
    assert sleeps == [0.5, 0.5]


def test_fallback_retry_linear_mode_wraps_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    retry_mod = _import_retry_without_tenacity(monkeypatch)
    sleeps: list[float] = []
    monkeypatch.setattr(retry_mod.time, "sleep", lambda seconds: sleeps.append(seconds))

    @retry_mod.retry(retries=3, delay=0.25, backoff=0.75, mode="linear", exceptions=(RuntimeError,))
    def always_fails() -> None:
        raise RuntimeError("still down")

    with pytest.raises(retry_mod.RetryError):
        always_fails()

    assert sleeps == [0.25, 1.0]


def test_fallback_retry_predicate_controls_retry_and_reraise(monkeypatch: pytest.MonkeyPatch) -> None:
    retry_mod = _import_retry_without_tenacity(monkeypatch)
    sleeps: list[float] = []
    calls = {"count": 0}
    monkeypatch.setattr(retry_mod.time, "sleep", lambda seconds: sleeps.append(seconds))

    @retry_mod.retry(
        stop=retry_mod.stop_after_attempt(2),
        retry=retry_mod.retry_if_exception_type(KeyError),
        reraise=True,
    )
    def key_error() -> None:
        calls["count"] += 1
        raise KeyError("retryable")

    with pytest.raises(KeyError):
        key_error()

    assert calls["count"] == 2
    assert sleeps == [0.1]


def test_fallback_retry_policy_error_fails_safe_to_no_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    retry_mod = _import_retry_without_tenacity(monkeypatch)
    calls = {"count": 0}

    def broken_policy(_exc: BaseException) -> bool:
        raise RuntimeError("policy failed")

    @retry_mod.retry(retries=5, retry=broken_policy, exceptions=(ValueError,))
    def raises_once() -> None:
        calls["count"] += 1
        raise ValueError("do not retry when policy breaks")

    with pytest.raises(ValueError):
        raises_once()

    assert calls["count"] == 1
