import pandas as pd
import numpy as np
import inspect
import signals
import indicators
import pytest

# Generate a large random DataFrame for benchmarking
# using half a million rows to keep runtime reasonable.
df = pd.DataFrame({
    'Open': np.random.random(500_000) * 100,
    'High': np.random.random(500_000) * 100,
    'Low': np.random.random(500_000) * 100,
    'Close': np.random.random(500_000) * 100,
    'Volume': np.random.randint(1000, 10000, size=500_000)
})

params = []
modules = [signals, indicators]
for module in modules:
    for name, func in inspect.getmembers(module, inspect.isfunction):
        params.append(pytest.param(func, id=f"{module.__name__}.{name}"))


@pytest.mark.parametrize("func", params)
def test_benchmarks(benchmark, func):
    benchmark(func, df)
