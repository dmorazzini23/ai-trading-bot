import pandas as pd
import numpy as np
import inspect
import signals
import indicators

df = pd.DataFrame({
    'Open': np.random.random(500_000) * 100,
    'High': np.random.random(500_000) * 100,
    'Low': np.random.random(500_000) * 100,
    'Close': np.random.random(500_000) * 100,
    'Volume': np.random.randint(1000, 10000, size=500_000)
})

modules = [signals, indicators]
funcs = []
for module in modules:
    funcs.extend(inspect.getmembers(module, inspect.isfunction))

import pytest

@pytest.mark.parametrize("func", [pytest.param(func, id=mod.__name__ + '.' + name)
    for mod in modules
    for name, func in inspect.getmembers(mod, inspect.isfunction)])
def test_benchmarks(benchmark, func):
    benchmark(func, df)

