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

    start = time.perf_counter()
    out = prepare_indicators(df.copy())
    duration = time.perf_counter() - start

    assert not out.empty
    # Expect completion well under 100ms on commodity hardware
    assert duration < 0.1

