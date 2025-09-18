"""Retraining CLI helpers exposed for public use.

This module provides a thin, documented surface for retraining related
utilities.  It intentionally delegates to the implementations that live in
``ai_trading.core`` and ``ai_trading.meta_learning`` so that the behaviour is
consistent with the running bot while keeping import time minimal.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - import guard for type checkers only
    import pandas as pd

__all__ = ["atomic_joblib_dump", "detect_regime", "main", "build_parser"]


def _load_bot_engine():
    from ai_trading.core import bot_engine as _bot_engine

    return _bot_engine


def detect_regime(df: "pd.DataFrame") -> str:
    """Delegate market regime detection to the bot engine implementation."""

    return _load_bot_engine().detect_regime(df)


def atomic_joblib_dump(obj: object, path: Path | str) -> None:
    """Persist ``obj`` to ``path`` using an atomic replace strategy."""

    _load_bot_engine().atomic_joblib_dump(obj, str(Path(path)))


def build_parser() -> argparse.ArgumentParser:
    """Return an argument parser for the retraining CLI."""

    parser = argparse.ArgumentParser(
        prog="python -m retrain",
        description=(
            "Retrain the meta-learning ensemble from historical trade logs. "
            "All arguments are optional; defaults match in-package settings."
        ),
    )
    parser.add_argument(
        "--trade-log",
        type=Path,
        default=Path("data/trades.csv"),
        help="CSV file containing historical trade records.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("meta_model.pkl"),
        help="Destination for the updated meta-learner model.",
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        default=Path("meta_retrain_history.pkl"),
        help="Path to persist retrain metrics history.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum number of valid trade rows required before retraining.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point used by ``python -m retrain``."""

    parser = build_parser()
    args = parser.parse_args(argv)

    from ai_trading.meta_learning.core import retrain_meta_learner

    logger.info(
        "Retraining meta learner",
        extra={
            "trade_log": str(args.trade_log),
            "model_path": str(args.model_path),
            "history_path": str(args.history_path),
            "min_samples": int(args.min_samples),
        },
    )
    success = retrain_meta_learner(
        trade_log_path=str(args.trade_log),
        model_path=str(args.model_path),
        history_path=str(args.history_path),
        min_samples=int(args.min_samples),
    )
    if success:
        logger.info("Retraining complete", extra={"model_path": str(args.model_path)})
        return 0

    logger.error("Retraining failed", extra={"trade_log": str(args.trade_log)})
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI execution helper
    raise SystemExit(main())
