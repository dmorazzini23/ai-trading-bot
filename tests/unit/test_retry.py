"""Unit tests for retry_call utility."""

from __future__ import annotations

import time

import pytest
from ai_trading.utils.retry import retry_call


class _Flaky:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        if self.calls < 2:
            raise RuntimeError("fail")
        return "ok"


class _AlwaysFail:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1
        raise ValueError("nope")


@pytest.mark.unit
def test_retry_eventually_succeeds() -> None:
    flaky = _Flaky()
    result = retry_call(flaky, exceptions=(RuntimeError,), retries=2)
    assert result == "ok"
    assert flaky.calls == 2


@pytest.mark.unit
def test_retry_raises_after_exhaustion() -> None:
    failing = _AlwaysFail()
    with pytest.raises(ValueError):
        retry_call(failing, exceptions=(ValueError,), retries=2)
    assert failing.calls == 3


@pytest.mark.unit
def test_fast_retry_skips_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAST_RETRY_IN_TESTS", "1")
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    calls = {"n": 0}

    def func() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("boom")
        return "done"

    start = time.perf_counter()
    result = retry_call(func, exceptions=(RuntimeError,), retries=1, backoff=0.5)
    elapsed = time.perf_counter() - start
    assert result == "done"
    assert calls["n"] == 2
    assert elapsed < 0.01
