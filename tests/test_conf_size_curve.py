from tests.optdeps import require
require("numpy")
import numpy as np
import pytest
from ai_trading.strategies.performance_allocator import _compute_conf_multiplier


@pytest.mark.parametrize(
    "th,max_boost,gamma",
    [
        (0.7, 1.15, 1.0),
        (0.7, 1.25, 0.5),
        (0.7, 1.10, 2.0),
    ],
)
def test_monotonic_and_bounds(th, max_boost, gamma):
    xs = np.linspace(th, 1.0, 50)
    ys = [_compute_conf_multiplier(float(x), th, max_boost, gamma) for x in xs]
    assert min(ys) >= 1.0 - 1e-9
    assert max(ys) <= max_boost + 1e-9
    assert all(b + 1e-12 >= a for a, b in zip(ys, ys[1:]))
    assert abs(ys[0] - 1.0) < 1e-6
    assert abs(ys[-1] - max_boost) < 1e-6
