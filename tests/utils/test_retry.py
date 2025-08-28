import time

import pytest
from ai_trading.utils.retry import retry_call


class Flaky:
    def __init__(self, fail_times: int) -> None:
        self.calls = 0
        self.fail_times = fail_times

    def __call__(self) -> str:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise TimeoutError("boom")
        return "ok"


class Bad:
    def __call__(self) -> None:
        raise ValueError("bad")


def test_retry_succeeds_and_sleeps(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    fn = Flaky(2)
    assert retry_call(fn, exceptions=(TimeoutError,), retries=3) == "ok"
    assert fn.calls == 3
    assert len(sleeps) == 2


def test_non_listed_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    fn = Bad()
    with pytest.raises(ValueError):
        retry_call(fn, exceptions=(TimeoutError,), retries=5)


def test_backoff_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    fn = Flaky(5)
    with pytest.raises(TimeoutError):
        retry_call(fn, exceptions=(TimeoutError,), retries=4, backoff=0.1, max_backoff=0.3, jitter=0)
    assert len(sleeps) == 4
    assert sleeps[-1] <= 0.3
    assert sleeps == sorted(sleeps)


def test_backoff_sequence_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    def always_fail() -> None:
        raise TimeoutError("boom")

    with pytest.raises(TimeoutError):
        retry_call(
            always_fail,
            exceptions=(TimeoutError,),
            retries=5,
            backoff=0.1,
            max_backoff=0.4,
            jitter=0,
        )
    assert sleeps == [0.1, 0.2, 0.4, 0.4, 0.4]
