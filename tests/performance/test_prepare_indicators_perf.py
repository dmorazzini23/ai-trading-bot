import os
import statistics
import time

import numpy as np
import pandas as pd

from ai_trading.core.bot_engine import prepare_indicators


def test_prepare_indicators_within_budget():
    """prepare_indicators should run quickly on typical workloads."""

    n = 2000
    df = pd.DataFrame(
        {
            "open": np.random.uniform(100, 200, n),
            "high": np.random.uniform(100, 200, n),
            "low": np.random.uniform(100, 200, n),
            "close": np.random.uniform(100, 200, n),
            "volume": np.random.randint(1_000_000, 5_000_000, n),
        }
    )

    # Warm cache/one-time setup effects before measuring.
    out = prepare_indicators(df.copy())
    durations: list[float] = []
    for _ in range(3):
        start = time.perf_counter()
        out = prepare_indicators(df.copy())
        durations.append(time.perf_counter() - start)

    assert not out.empty
    # Keep this strict enough to catch regressions but resilient to shared-host jitter.
    budget_s = float(os.getenv("AI_TRADING_TEST_PREPARE_INDICATORS_BUDGET_S", "0.14"))
    best = min(durations)
    median = statistics.median(durations)
    assert best < budget_s, (
        f"prepare_indicators best={best:.6f}s median={median:.6f}s budget={budget_s:.6f}s"
    )
