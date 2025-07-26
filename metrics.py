"""Performance metrics for trading results."""

from __future__ import annotations

import pandas as pd
import numpy as np


def compute_basic_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Return Sharpe ratio and max drawdown from ``df`` with a ``return`` column."""
    if "return" not in df:
        return {"sharpe": 0.0, "max_drawdown": 0.0}
    ret = df["return"].astype(float)
    if ret.empty:
        return {"sharpe": 0.0, "max_drawdown": 0.0}
    sharpe = ret.mean() / (ret.std() or 1e-9) * (252 ** 0.5)
    cumulative = (1 + ret).cumprod()
    drawdown = cumulative.cummax() - cumulative
    max_dd = drawdown.max()
    return {"sharpe": float(sharpe), "max_drawdown": float(max_dd)}
