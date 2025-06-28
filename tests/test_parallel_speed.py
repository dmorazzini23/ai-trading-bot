import time
import pandas as pd
import signals

def test_parallel_vs_serial_prep_speed():
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
    n = 60
    data = pd.DataFrame(
        {
            "open": range(1, n + 1),
            "high": range(2, n + 2),
            "low": range(n),
            "close": range(1, n + 1),
            "volume": [100] * n,
        }
    )

    start_serial = time.perf_counter()
    for _ in symbols:
        signals.prepare_indicators(data)
    duration_serial = time.perf_counter() - start_serial

    start_parallel = time.perf_counter()
    signals.prepare_indicators_parallel(symbols, {s: data for s in symbols})
    duration_parallel = time.perf_counter() - start_parallel

    assert duration_parallel < duration_serial * 0.9, \
        f"Parallel too slow vs serial: {duration_parallel:.4f} vs {duration_serial:.4f}"
