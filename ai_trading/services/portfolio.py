from __future__ import annotations

from typing import Any, cast


def compute_portfolio_weights(ctx: Any, symbols: list[str]) -> dict[str, float]:
    """Delegate portfolio sizing to the canonical portfolio module."""

    from ai_trading.portfolio import compute_portfolio_weights as _compute_weights

    return cast(dict[str, float], _compute_weights(ctx, symbols))


def ensure_portfolio_weights(
    ctx: Any,
    symbols: list[str],
    *,
    logger: Any,
) -> dict[str, float]:
    """Compute portfolio weights with the legacy even-weight fallback."""

    try:
        from ai_trading import portfolio

        if hasattr(portfolio, "compute_portfolio_weights"):
            return cast(
                dict[str, float],
                portfolio.compute_portfolio_weights(ctx, symbols),
            )
        logger.warning("compute_portfolio_weights not found, using fallback method.")
    except (ZeroDivisionError, ValueError, KeyError) as exc:
        logger.error(
            "PORTFOLIO_WEIGHT_FAILED",
            extra={"cause": exc.__class__.__name__, "detail": str(exc)},
        )
    count = len(symbols)
    return {symbol: (1.0 / count) for symbol in symbols if count > 0}


__all__ = ["compute_portfolio_weights", "ensure_portfolio_weights"]
