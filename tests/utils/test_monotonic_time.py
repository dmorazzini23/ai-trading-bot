from collections import deque

import pytest

from ai_trading.utils import time as time_utils


def test_monotonic_time_iterator_fallback(monkeypatch):
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
