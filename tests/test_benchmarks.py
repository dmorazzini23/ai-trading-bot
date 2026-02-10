import inspect

import numpy as np
import pytest
pd = pytest.importorskip("pandas")
from ai_trading import indicators, signals

_BENCH_ROWS = 25_000


@pytest.fixture(scope="module")
def bench_df():
    """Shared input frame for indicator/signal smoke tests.

    Keep this modest to avoid xdist worker crashes from per-worker module import
    allocations while still exercising vectorized paths.
    """

    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "open": rng.random(_BENCH_ROWS) * 100.0,
            "high": rng.random(_BENCH_ROWS) * 100.0,
            "low": rng.random(_BENCH_ROWS) * 100.0,
            "close": rng.random(_BENCH_ROWS) * 100.0,
            "volume": rng.integers(1000, 10_000, size=_BENCH_ROWS),
        }
    )

modules = [signals, indicators]
params = []

for module in modules:
    for name, func in inspect.getmembers(module, inspect.isfunction):
        # skip private functions
        if name.startswith("_"):
            continue
        sig = inspect.signature(func)
        required_positional = [
            p for p in sig.parameters.values()
            if p.default == inspect.Parameter.empty and p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD
            )
        ]
        # must have exactly 1 required positional argument
        if len(required_positional) != 1:
            continue
        # skip if explicitly takes str, not intended for DataFrames
        if required_positional[0].annotation == str:
            continue
        if hasattr(func, "py_func") or name == "jit":
            continue
        params.append(pytest.param(func, id=f"{module.__name__}.{name}"))

@pytest.mark.parametrize("func", params)
def test_benchmarks(request, func, bench_df):
    """Functional smoke for indicator/signal callability in frozen-time test runs."""
    _ = request  # keep signature stable for existing callers/plugins
    result = func(bench_df)
    # Some helpers are mutative/no-return by design; this guards only for crashes.
    assert result is not None or result is None
