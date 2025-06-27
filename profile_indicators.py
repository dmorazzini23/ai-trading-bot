import time
import pandas as pd
import numpy as np
import inspect
import signals
import indicators


def profile(func, *args, **kwargs):
    start = time.perf_counter()
    try:
        result = func(*args, **kwargs)
    except Exception as e:
        print(f"{func.__name__} failed: {e}")
        return None, -1
    elapsed = time.perf_counter() - start
    print(f"{func.__name__} took {elapsed:.6f} sec")
    return result, elapsed


def run_profiles():
    timings = []
    df = pd.DataFrame({
        'Open': np.random.random(500_000) * 100,
        'High': np.random.random(500_000) * 100,
        'Low': np.random.random(500_000) * 100,
        'Close': np.random.random(500_000) * 100,
        'Volume': np.random.randint(1000, 10000, size=500_000)
    })

    modules = [signals, indicators]
    for module in modules:
        funcs = inspect.getmembers(module, inspect.isfunction)
        for name, func in funcs:
            _, elapsed = profile(func, df)
            timings.append((module.__name__ + "." + name, elapsed))

    pd.DataFrame(timings, columns=["Function", "Time(sec)"]).to_csv("indicator_timings.csv", index=False)


if __name__ == "__main__":
    run_profiles()

