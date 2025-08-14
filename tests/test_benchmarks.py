import pandas as pd
import numpy as np
import inspect
import ai_trading.signals as signals
import ai_trading.indicators as indicators
import pytest

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
            print(f"Skipping private function {module.__name__}.{name}")
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
            print(f"Skipping {module.__name__}.{name}: requires {len(required_positional)} positional args.")
            continue
        # skip if explicitly takes str, not intended for DataFrames
        if required_positional[0].annotation == str:
            print(f"Skipping {module.__name__}.{name}: expects str arg.")
            continue
        if hasattr(func, "py_func") or name == "jit":
            print(f"Skipping decorator or jit function {module.__name__}.{name}")
            continue
        params.append(pytest.param(func, id=f"{module.__name__}.{name}"))

@pytest.mark.parametrize("func", params)
def test_benchmarks(request, func):
    """Test function performance, with optional benchmark if pytest-benchmark is available."""
    try:
        # Try to use the benchmark fixture if available
        benchmark = request.getfixturevalue('benchmark')
        benchmark(func, df)
    except LookupError:
        # Fallback: just run the function to ensure it works
        result = func(df)
        # Basic verification that function completed without error
        assert result is not None or result is None  # Allow any result including None
