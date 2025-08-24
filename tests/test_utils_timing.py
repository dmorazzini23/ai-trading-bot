"""Sanity checks for centralized timing helpers exported by ai_trading.utils."""
from __future__ import annotations

from time import perf_counter

import pytest

from ai_trading.utils import HTTP_TIMEOUT, clamp_timeout, sleep


def test_timing_exports_exist_and_behave():
    assert isinstance(HTTP_TIMEOUT, int | float)
    assert HTTP_TIMEOUT > 0

    assert clamp_timeout(None) == pytest.approx(float(HTTP_TIMEOUT))
    assert clamp_timeout(0.0) >= 0.0
    assert clamp_timeout(-1.0) >= 0.0

    t0 = perf_counter()
    sleep(0.001)
    dt = perf_counter() - t0
    assert dt >= 0.0
    assert dt < max(0.25, HTTP_TIMEOUT)

