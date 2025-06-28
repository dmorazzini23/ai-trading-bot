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
        'open': np.random.random(500_000) * 100,
        'high': np.random.random(500_000) * 100,
        'low': np.random.random(500_000) * 100,
        'close': np.random.random(500_000) * 100,
        'volume': np.random.randint(1000, 10000, size=500_000)
    })

    modules = [signals, indicators]
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
            if len(required_positional) != 1:
                print(f"Skipping {module.__name__}.{name} — requires {len(required_positional)} positional args: {[p.name for p in required_positional]}")
                continue
            if hasattr(func, "py_func") or name == "jit":
                print(f"Skipping decorator or jit-wrapped function {module.__name__}.{name}")
                continue
            _, elapsed = profile(func, df)
            timings.append((module.__name__ + "." + name, elapsed))

    pd.DataFrame(timings, columns=["Function", "Time(sec)"]).to_csv("indicator_timings.csv", index=False)

if __name__ == "__main__":
    run_profiles()
