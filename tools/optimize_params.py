#!/usr/bin/env python
"""Lightweight offline parameter optimizer.

Runs a quick grid search over simple strategies (momentum/mean_reversion)
using local CSV data in ./data (if present) or a generated synthetic series.

Outputs the best parameter set by crude Sharpe-like scoring. Intended for
developer use; not imported by runtime.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import math

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover - optional
    raise SystemExit("pandas is required: pip install 'ai-trading-bot[pandas]'") from exc


def _load_prices_from_data_dir(symbol: str | None = None) -> pd.Series:
    data_dir = Path("data")
    if data_dir.exists():
        # Look for a CSV with a close column
        cands = list(data_dir.glob("*.csv"))
        for p in cands:
            try:
                df = pd.read_csv(p)
                for col in ("close", "Close", "adj_close", "Adj Close"):
                    if col in df.columns and len(df) > 50:
                        return pd.to_numeric(df[col], errors="coerce").dropna().astype(float)
            except Exception:
                continue
    # Fallback: synthetic drift + noise series
    n = 1000
    import numpy as np

    rng = np.random.default_rng(42)
    returns = rng.normal(loc=0.0002, scale=0.01, size=n)
    prices = 100.0 * (1.0 + pd.Series(returns)).cumprod()
    return prices


def _score_returns(rets: pd.Series) -> float:
    rets = pd.to_numeric(rets, errors="coerce").dropna()
    if rets.empty:
        return -math.inf
    mu = rets.mean()
    sd = rets.std(ddof=0)
    if sd == 0:
        return -math.inf
    return float(252**0.5 * mu / sd)  # crude Sharpe


def _evaluate_momentum(prices: pd.Series, lookback: int, threshold: float) -> float:
    if len(prices) <= lookback + 2:
        return -math.inf
    prev = prices.shift(lookback)
    mom = prices / prev - 1.0
    signal = (mom > threshold).astype(int)
    fwd = prices.pct_change().shift(-1)
    strat_rets = (signal * fwd).dropna()
    return _score_returns(strat_rets)


def _evaluate_mean_reversion(prices: pd.Series, lookback: int, z_entry: float) -> float:
    roll = prices.rolling(window=lookback, min_periods=lookback)
    mu = roll.mean()
    sd = roll.std(ddof=0)
    z = (prices - mu) / (sd + 1e-12)
    # long when deeply negative, flat otherwise (simple rule)
    signal = (z <= -abs(z_entry)).astype(int)
    fwd = prices.pct_change().shift(-1)
    strat_rets = (signal * fwd).dropna()
    return _score_returns(strat_rets)


def _grid(values: Iterable) -> list:
    return list(values)


def optimize(strategy: str) -> Tuple[dict, float]:
    prices = _load_prices_from_data_dir()
    best, best_score = {}, -math.inf
    if strategy == "momentum":
        for lookback in _grid(range(10, 61, 5)):
            for thr in _grid([x / 100.0 for x in range(0, 21, 2)]):
                sc = _evaluate_momentum(prices, lookback, thr)
                if sc > best_score:
                    best, best_score = ({"lookback": lookback, "threshold": thr}, sc)
    elif strategy == "mean_reversion":
        for lookback in _grid(range(5, 41, 5)):
            for z in _grid([x / 10.0 for x in range(5, 26, 1)]):
                sc = _evaluate_mean_reversion(prices, lookback, z)
                if sc > best_score:
                    best, best_score = ({"lookback": lookback, "z_entry": z}, sc)
    else:
        raise SystemExit(f"Unknown strategy: {strategy}")
    return best, best_score


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Offline parameter optimizer")
    p.add_argument("strategy", choices=["momentum", "mean_reversion"], help="Strategy to optimize")
    args = p.parse_args(argv)
    params, score = optimize(args.strategy)
    print({"strategy": args.strategy, "best_params": params, "score": round(score, 4)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

