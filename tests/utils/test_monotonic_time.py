from collections import deque

import pytest

from ai_trading.utils import time as time_utils


def test_monotonic_time_iterator_fallback(monkeypatch):
    time_utils._LAST_MONOTONIC_VALUE = None
    monotonic_values = deque([1.23, 4.56])
    fallback_values = deque([10.0, 20.0])
    real_time = time_utils._time_module.time

    def fake_monotonic():
        if not monotonic_values:
            raise StopIteration
        return monotonic_values.popleft()

    def fake_time():
        if fallback_values:
            return fallback_values.popleft()
        return real_time()

    monkeypatch.setattr(time_utils._time_module, "monotonic", fake_monotonic)
    monkeypatch.setattr(time_utils._time_module, "time", fake_time)

    assert time_utils.monotonic_time() == pytest.approx(1.23)
    assert time_utils.monotonic_time() == pytest.approx(4.56)
    assert time_utils.monotonic_time() == pytest.approx(10.0)
    assert time_utils.monotonic_time() == pytest.approx(20.0)
    assert isinstance(time_utils.monotonic_time(), float)


def test_monotonic_time_iterator_exhaustion_resilient(monkeypatch):
    time_utils._LAST_MONOTONIC_VALUE = None
    monotonic_values = iter([1.0, 2.0])
    fallback_values = deque([0.5, 0.6, 10.0])
    real_time = time_utils._time_module.time

    def fake_monotonic():
        return next(monotonic_values)

    def fake_time():
        if fallback_values:
            return fallback_values.popleft()
        return real_time()

    monkeypatch.setattr(time_utils._time_module, "monotonic", fake_monotonic)
    monkeypatch.setattr(time_utils._time_module, "time", fake_time)

    results = [time_utils.monotonic_time() for _ in range(5)]

    assert results[0] == pytest.approx(1.0)
    assert results[1] == pytest.approx(2.0)
    assert all(isinstance(value, float) for value in results)
    assert results[2] >= results[1]
    assert results[3] >= results[2]
    assert results[4] >= results[3]


def test_monotonic_time_runtime_error_from_generator(monkeypatch):
    time_utils._LAST_MONOTONIC_VALUE = None
    fallback_values = deque([3.0, 4.0])
    real_time = time_utils._time_module.time

    def fake_monotonic():
        def _generator():
            if False:  # pragma: no cover - generator marker
                yield None
            raise StopIteration

        return next(_generator())

    def fake_time():
        if fallback_values:
            return fallback_values.popleft()
        return real_time()

    monkeypatch.setattr(time_utils._time_module, "monotonic", fake_monotonic)
    monkeypatch.setattr(time_utils._time_module, "time", fake_time)

    assert time_utils.monotonic_time() == pytest.approx(3.0)
    assert time_utils.monotonic_time() == pytest.approx(4.0)
    assert isinstance(time_utils.monotonic_time(), float)
