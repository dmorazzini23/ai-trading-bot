"""
Stacking + Meta-Labeling Walk-Forward Demo.

Builds simple features and targets from daily bars, runs walk-forward
analysis in both rolling and anchored modes using the 'stacking' model,
and records aggregate evaluation metrics to the model registry.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from typing import Any

try:  # runtime-safe imports
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from ai_trading.evaluation.walkforward import WalkForwardEvaluator
from ai_trading.model_registry import record_evaluation
from ai_trading.logging import logger


def _fetch_symbol_df(symbol: str, start: datetime, end: datetime) -> "pd.DataFrame | None":
    try:
        from ai_trading.data.fetch import get_daily_df

        return get_daily_df(symbol, start, end)
    except Exception as exc:
        logger.warning("FETCH_FAILED", extra={"symbol": symbol, "detail": str(exc)})
        return None


def _build_dataset(df: "pd.DataFrame") -> "pd.DataFrame":
    import numpy as np

    data = df.copy()
    data = data.sort_index()
    data["ret1"] = data["close"].pct_change()
    data["mom5"] = data["close"].pct_change(5)
    data["vol20"] = data["ret1"].rolling(20).std()
    data["target"] = data["close"].pct_change().shift(-1)  # next period return
    data = data.dropna()
    return data[["ret1", "mom5", "vol20", "target"]]


def run_for_symbols(symbols: list[str]) -> None:
    if pd is None:
        raise RuntimeError("pandas is required for this demo")
    end = datetime.now(UTC)
    start = end - timedelta(days=3 * 365)
    for mode in ("rolling", "anchored"):
        try:
            all_frames = []
            for sym in symbols:
                df = _fetch_symbol_df(sym, start, end)
                if df is None or df.empty:
                    continue
                built = _build_dataset(df)
                built.columns = [f"{c}" if c == "target" else c for c in built.columns]
                all_frames.append(built)
            if not all_frames:
                logger.warning("No data for symbols: %s", symbols)
                continue
            # For demo simplicity evaluate one symbol at a time; record per-symbol
            for sym in symbols:
                df = _fetch_symbol_df(sym, start, end)
                if df is None or df.empty:
                    continue
                data = _build_dataset(df)
                evaluator = WalkForwardEvaluator(mode=mode, train_span=252, test_span=63, embargo_pct=0.01)
                result = evaluator.run_walkforward(
                    data=data,
                    target_col="target",
                    feature_cols=[c for c in data.columns if c != "target"],
                    model_type="stacking",
                    feature_pipeline_params=None,
                    save_results=False,
                )
                metrics = result.get("aggregate_metrics", {})
                logger.info("WFA %s %s metrics: %s", mode, sym, metrics)
                record_evaluation(sym, {"mode": mode, **metrics})
        except Exception as exc:
            logger.error("STACKING_WFA_FAILED", extra={"cause": exc.__class__.__name__, "detail": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Stacking WFA Demo")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols", default="SPY,AAPL,MSFT")
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    run_for_symbols(symbols)


if __name__ == "__main__":
    main()

