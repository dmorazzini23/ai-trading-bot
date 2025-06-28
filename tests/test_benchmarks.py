import pandas as pd
import numpy as np
import inspect
import signals
import indicators
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
        sig = inspect.signature(func)
        required_positional = [
            p for p in sig.parameters.values()
            if p.default == inspect.Parameter.empty and p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD
            )
        ]
        # filter: must have exactly 1 positional argument
        if len(required_positional) != 1:
            print(f"Skipping {module.__name__}.{name} — requires {len(required_positional)} positional args: {[p.name for p in required_positional]}")
            continue
        # avoid jit or decorator proxies
        if hasattr(func, "py_func") or name == "jit":
            print(f"Skipping decorator or jit-wrapped function {module.__name__}.{name}")
            continue
        params.append(pytest.param(func, id=f"{module.__name__}.{name}"))

@pytest.mark.parametrize("func", params)
def test_benchmarks(benchmark, func):
    benchmark(func, df)
