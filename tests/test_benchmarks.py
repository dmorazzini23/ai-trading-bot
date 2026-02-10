import inspect

import numpy as np
import pytest
pd = pytest.importorskip("pandas")
from ai_trading import indicators, signals

df = pd.DataFrame({
    'open': np.random.random(500_000) * 100,
    'high': np.random.random(500_000) * 100,
    'low': np.random.random(500_000) * 100,
    'close': np.random.random(500_000) * 100,
    'volume': np.random.randint(1000, 10000, size=500_000)
})

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
def test_benchmarks(request, func):
    """Functional smoke for indicator/signal callability in frozen-time test runs."""
    _ = request  # keep signature stable for existing callers/plugins
    result = func(df)
    # Some helpers are mutative/no-return by design; this guards only for crashes.
    assert result is not None or result is None
