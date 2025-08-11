import logging

logger = logging.getLogger(__name__)
import inspect
import time

import numpy as np
import pandas as pd

from ai_trading import signals
from ai_trading import indicators

def profile(func, *args, **kwargs):
    start = time.perf_counter()
    try:
        result = func(*args, **kwargs)
    except Exception as e:
        logger.error("%s failed: %s", func.__name__, e)
        return None, -1
    elapsed = time.perf_counter() - start
    logger.info("%s took %.6f sec", func.__name__, elapsed)
    return result, elapsed

def run_profiles():
    timings = []
    df = pd.DataFrame({
        'open': np.random.random(100_000) * 100,
        'high': np.random.random(100_000) * 100,
        'low': np.random.random(100_000) * 100,
        'close': np.random.random(100_000) * 100,
        'volume': np.random.randint(1000, 10000, size=100_000)
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
                logger.info("Skipping %s.%s - requires %s positional args", module.__name__, name, len(required_positional))
                continue
            if hasattr(func, "py_func") or name == "jit":
                logger.info("Skipping decorator or jit-wrapped function %s.%s", module.__name__, name)
                continue
            _, elapsed = profile(func, df)
            timings.append((module.__name__ + "." + name, elapsed))

    pd.DataFrame(timings, columns=["Function", "Time(sec)"]).to_csv("indicator_timings.csv", index=False)

if __name__ == "__main__":
    run_profiles()
