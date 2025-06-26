"""Metric logging utilities for backtests."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd


class MetricsLogger:
    """Collects metrics from backtest runs and writes them to disk."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self.records: list[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)

    def log_run(self, params: Dict[str, Any], result) -> None:
        """Record a run's parameters and resulting metrics."""
        cfg_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
        rec = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "params": params,
            "config_hash": cfg_hash,
            "cumulative_return": result.cumulative_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
        }
        self.records.append(rec)
        self.logger.info(
            "Run %s CR=%.4f Sharpe=%.4f DD=%.4f",
            cfg_hash,
            rec["cumulative_return"],
            rec["sharpe_ratio"],
            rec["max_drawdown"],
        )

    def flush(self) -> None:
        """Write accumulated records to ``self.path`` if provided."""
        if not self.path:
            return
        if self.path.endswith(".json"):
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.records, f, indent=2)
        else:
            pd.DataFrame(self.records).to_csv(self.path, index=False)
