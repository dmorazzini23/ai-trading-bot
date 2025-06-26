#!/usr/bin/env python3.12
"""Profile the prepare_indicators function."""

from __future__ import annotations

import cProfile
import pstats
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bot_engine import prepare_indicators


def generate_dummy_prices(days: int = 365) -> pd.DataFrame:
    """Generate a dummy OHLCV DataFrame with minute frequency."""
    minutes = days * 390
    index = pd.date_range("2023-01-01", periods=minutes, freq="1min")
    data = {
        "open": np.random.uniform(100, 200, size=minutes),
        "high": np.random.uniform(100, 200, size=minutes),
        "low": np.random.uniform(100, 200, size=minutes),
        "close": np.random.uniform(100, 200, size=minutes),
        "volume": np.random.randint(1_000, 10_000, size=minutes),
    }
    return pd.DataFrame(data, index=index)


def main() -> None:
    df = generate_dummy_prices()
    profiler = cProfile.Profile()
    profiler.enable()
    prepare_indicators(df.copy())
    profiler.disable()

    stats = pstats.Stats(profiler).sort_stats("cumtime")
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    out_file = logs_dir / "prepare_indicators_profile.txt"
    with out_file.open("w") as f:
        stats.stream = f
        stats.print_stats()
    stats.stream = sys.stdout
    stats.print_stats(20)


if __name__ == "__main__":  # pragma: no cover - manual profiling utility
    main()
