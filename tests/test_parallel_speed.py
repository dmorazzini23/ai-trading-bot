import time
import pandas as pd
import signals

def test_parallel_vs_serial_prep_speed():
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
    data = pd.DataFrame({
        "open": [1, 2, 3, 4, 5],
        "high": [2, 3, 4, 5, 6],
        "low": [0, 1, 2, 3, 4],
        "close": [1, 2, 3, 4, 5],
        "volume": [100, 200, 300, 400, 500],
    })

    start_serial = time.perf_counter()
    for _ in symbols:
        signals.prepare_indicators(data)
    duration_serial = time.perf_counter() - start_serial

    start_parallel = time.perf_counter()
    signals.prepare_indicators_parallel(symbols, {s: data for s in symbols})
    duration_parallel = time.perf_counter() - start_parallel

    assert duration_parallel < duration_serial * 0.9, \
        f"Parallel too slow vs serial: {duration_parallel:.4f} vs {duration_serial:.4f}"
