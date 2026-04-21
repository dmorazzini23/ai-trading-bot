from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast


class PortfolioService:
    """Canonical portfolio service facade over the runtime weighting module."""

    boundary_type = "facade"
    canonical_runtime_owner = "ai_trading.portfolio.compute_portfolio_weights"

    def compute_portfolio_weights(self, ctx: Any, symbols: list[str]) -> dict[str, float]:
        try:
            from ai_trading.portfolio import compute_portfolio_weights as _compute_weights
        except ImportError as exc:
            raise RuntimeError("compute_portfolio_weights is unavailable.") from exc

        return cast(dict[str, float], _compute_weights(ctx, symbols))

    def ensure_portfolio_weights(
        self,
        ctx: Any,
        symbols: list[str],
        *,
        logger: Any,
    ) -> dict[str, float]:
        """Compute validated portfolio weights without hidden equal-weight fallback."""

        if not symbols:
            return {}
        weights_raw = self.compute_portfolio_weights(ctx, symbols)
        if not isinstance(weights_raw, Mapping):
            logger.error("PORTFOLIO_WEIGHT_INVALID", extra={"reason": "non_mapping"})
            raise RuntimeError("compute_portfolio_weights must return a mapping.")

        normalized: dict[str, float] = {}
        for symbol in symbols:
            raw_weight = weights_raw.get(symbol)
            try:
                weight = float(raw_weight)
            except (TypeError, ValueError):
                logger.error(
                    "PORTFOLIO_WEIGHT_INVALID",
                    extra={"symbol": symbol, "reason": "non_numeric", "value": raw_weight},
                )
                raise RuntimeError(f"Invalid portfolio weight for {symbol}.") from None
            if not math.isfinite(weight):
                logger.error(
                    "PORTFOLIO_WEIGHT_INVALID",
                    extra={"symbol": symbol, "reason": "non_finite", "value": raw_weight},
                )
                raise RuntimeError(f"Invalid portfolio weight for {symbol}.")
            normalized[str(symbol)] = float(weight)

        total = float(sum(normalized.values()))
        if not math.isfinite(total) or total <= 0.0:
            logger.error(
                "PORTFOLIO_WEIGHT_INVALID",
                extra={"reason": "non_positive_total", "total": total},
            )
            raise RuntimeError("Portfolio weights must sum to a positive finite value.")
        return {
            symbol: float(weight) / total
            for symbol, weight in normalized.items()
        }


def compute_portfolio_weights(ctx: Any, symbols: list[str]) -> dict[str, float]:
    return PortfolioService().compute_portfolio_weights(ctx, symbols)


def ensure_portfolio_weights(
    ctx: Any,
    symbols: list[str],
    *,
    logger: Any,
) -> dict[str, float]:
    return PortfolioService().ensure_portfolio_weights(ctx, symbols, logger=logger)


__all__ = ["PortfolioService", "compute_portfolio_weights", "ensure_portfolio_weights"]
