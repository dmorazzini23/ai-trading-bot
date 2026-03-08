"""Institutional validation helpers for promotion and governance gates."""
from __future__ import annotations

import math
import random
from statistics import mean
from typing import TYPE_CHECKING, Any

import numpy as np
from ai_trading.utils.lazy_imports import load_pandas

from ai_trading.data.splits import PurgedGroupTimeSeriesSplit, validate_no_leakage
from ai_trading.risk.metrics import RiskMetricsCalculator

if TYPE_CHECKING:
    import pandas as pd

_pd = load_pandas()


def _clean_returns(values: list[float] | tuple[float, ...] | np.ndarray | None) -> list[float]:
    if values is None:
        return []
    cleaned: list[float] = []
    for raw in values:
        try:
            parsed = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            cleaned.append(parsed)
    return cleaned


def _max_drawdown_from_returns(returns: list[float]) -> float:
    if not returns:
        return 0.0
    cumulative = np.cumprod(np.array([1.0 + value for value in returns], dtype=float))
    peaks = np.maximum.accumulate(cumulative)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdowns = np.where(peaks > 0.0, cumulative / peaks - 1.0, 0.0)
    return float(abs(np.nanmin(drawdowns))) if drawdowns.size else 0.0


def run_purged_walk_forward_validation(
    frame: pd.DataFrame,
    *,
    return_col: str = "ret",
    timestamp_col: str = "timestamp",
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    purge_pct: float = 0.02,
    min_fold_samples: int = 20,
) -> dict[str, Any]:
    """Run purged walk-forward validation with embargo and leakage checks."""

    if frame is None or frame.empty or return_col not in frame.columns:
        return {
            "fold_count": 0,
            "passed_folds": 0,
            "pass_ratio": 0.0,
            "folds": [],
        }

    data = frame.copy()
    if timestamp_col not in data.columns:
        data[timestamp_col] = np.arange(len(data))
    timeline = _pd.to_datetime(data[timestamp_col], utc=True, errors="coerce")
    data = data.assign(**{timestamp_col: timeline}).dropna(subset=[timestamp_col])
    if data.empty:
        return {
            "fold_count": 0,
            "passed_folds": 0,
            "pass_ratio": 0.0,
            "folds": [],
        }

    data = data.sort_values(timestamp_col).reset_index(drop=True)
    timeline = _pd.to_datetime(data[timestamp_col], utc=True)
    splitter = PurgedGroupTimeSeriesSplit(
        n_splits=max(1, int(n_splits)),
        embargo_pct=max(0.0, float(embargo_pct)),
        purge_pct=max(0.0, float(purge_pct)),
    )

    folds: list[dict[str, Any]] = []
    passed = 0
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(data)):
        train_df = data.iloc[train_idx]
        test_df = data.iloc[test_idx]
        leakage_ok = bool(validate_no_leakage(train_idx, test_idx, timeline.to_numpy(), t1=None))
        enough_samples = len(train_df) >= int(min_fold_samples) and len(test_df) >= int(min_fold_samples)
        fold_pass = bool(leakage_ok and enough_samples)
        if fold_pass:
            passed += 1

        train_returns = _clean_returns(train_df[return_col].tolist())
        test_returns = _clean_returns(test_df[return_col].tolist())
        folds.append(
            {
                "fold": fold_idx,
                "train_samples": len(train_df),
                "test_samples": len(test_df),
                "leakage_ok": leakage_ok,
                "train_expectancy_bps": float(mean(train_returns) * 10000.0) if train_returns else 0.0,
                "test_expectancy_bps": float(mean(test_returns) * 10000.0) if test_returns else 0.0,
                "test_hit_rate": float(sum(1 for value in test_returns if value > 0.0) / len(test_returns))
                if test_returns
                else 0.0,
                "passed": fold_pass,
            }
        )

    fold_count = len(folds)
    pass_ratio = float(passed / fold_count) if fold_count else 0.0
    return {
        "fold_count": fold_count,
        "passed_folds": int(passed),
        "pass_ratio": pass_ratio,
        "folds": folds,
    }


def run_monte_carlo_trade_sequence_stress(
    trade_returns: list[float] | tuple[float, ...] | np.ndarray,
    *,
    trials: int = 1000,
    horizon: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap trade-sequence outcomes to estimate stress quantiles."""

    returns = _clean_returns(trade_returns)
    if not returns:
        return {
            "trials": 0,
            "horizon": 0,
            "p05_return_bps": 0.0,
            "p50_return_bps": 0.0,
            "p95_return_bps": 0.0,
            "p95_max_drawdown_bps": 0.0,
        }

    horizon_len = max(1, int(horizon or len(returns)))
    trial_count = max(1, int(trials))
    rng = random.Random(int(seed))

    terminal_returns_bps: list[float] = []
    max_drawdowns_bps: list[float] = []
    for _ in range(trial_count):
        path = [rng.choice(returns) for _ in range(horizon_len)]
        cumulative = 1.0
        equity_curve = [cumulative]
        for trade_ret in path:
            cumulative *= 1.0 + float(trade_ret)
            equity_curve.append(cumulative)
        terminal_returns_bps.append((cumulative - 1.0) * 10000.0)
        max_drawdowns_bps.append(_max_drawdown_from_returns(path) * 10000.0)

    return {
        "trials": trial_count,
        "horizon": horizon_len,
        "p05_return_bps": float(np.percentile(terminal_returns_bps, 5)),
        "p50_return_bps": float(np.percentile(terminal_returns_bps, 50)),
        "p95_return_bps": float(np.percentile(terminal_returns_bps, 95)),
        "p95_max_drawdown_bps": float(np.percentile(max_drawdowns_bps, 95)),
    }


def run_regime_split_validation(
    frame: pd.DataFrame,
    *,
    regime_col: str = "regime",
    return_col: str = "ret",
    min_samples: int = 30,
    min_expectancy_bps: float = 0.0,
    min_hit_rate: float = 0.45,
) -> dict[str, Any]:
    """Evaluate performance consistency across market regimes."""

    if (
        frame is None
        or frame.empty
        or regime_col not in frame.columns
        or return_col not in frame.columns
    ):
        return {"regimes": {}, "eligible_regimes": 0, "pass_ratio": 0.0}

    regimes: dict[str, dict[str, Any]] = {}
    eligible = 0
    passed = 0
    for regime_name, subset in frame.groupby(regime_col):
        returns = _clean_returns(subset[return_col].tolist())
        sample_count = len(returns)
        if sample_count < max(1, int(min_samples)):
            regimes[str(regime_name)] = {
                "samples": sample_count,
                "expectancy_bps": 0.0,
                "hit_rate": 0.0,
                "eligible": False,
                "passed": False,
            }
            continue

        eligible += 1
        expectancy_bps = float(mean(returns) * 10000.0) if returns else 0.0
        hit_rate = float(sum(1 for value in returns if value > 0.0) / len(returns)) if returns else 0.0
        passed_regime = expectancy_bps >= float(min_expectancy_bps) and hit_rate >= float(min_hit_rate)
        if passed_regime:
            passed += 1
        regimes[str(regime_name)] = {
            "samples": sample_count,
            "expectancy_bps": expectancy_bps,
            "hit_rate": hit_rate,
            "eligible": True,
            "passed": passed_regime,
        }

    pass_ratio = float(passed / eligible) if eligible else 0.0
    return {"regimes": regimes, "eligible_regimes": eligible, "pass_ratio": pass_ratio}


def compute_risk_adjusted_scorecard(
    returns: list[float] | tuple[float, ...] | np.ndarray,
) -> dict[str, float]:
    """Compute promotion-ready risk-adjusted metrics from returns."""

    cleaned = _clean_returns(returns)
    calculator = RiskMetricsCalculator()
    sharpe = float(calculator.calculate_sharpe_ratio(cleaned))
    sortino = float(calculator.calculate_sortino_ratio(cleaned))
    max_drawdown = float(calculator.calculate_max_drawdown(cleaned))
    calmar = float(calculator.calculate_calmar_ratio(cleaned))
    tail_loss_95 = float(calculator.calculate_tail_loss(cleaned, confidence_level=0.95))
    risk_of_ruin = float(calculator.calculate_risk_of_ruin(cleaned))
    return {
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "tail_loss_95": tail_loss_95,
        "risk_of_ruin": risk_of_ruin,
    }


__all__ = [
    "compute_risk_adjusted_scorecard",
    "run_monte_carlo_trade_sequence_stress",
    "run_purged_walk_forward_validation",
    "run_regime_split_validation",
]
