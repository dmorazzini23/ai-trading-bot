from __future__ import annotations

from typing import Any


def evaluate_signal_and_confirm(
    ctx: Any,
    state: Any,
    symbol: str,
    df: Any,
    model: Any,
    *,
    conf_threshold: float,
    logger: Any,
) -> tuple[int, float, str]:
    """Evaluate a signal and enforce the configured confirmation threshold."""

    sig, conf, strat = ctx.signal_manager.evaluate(ctx, state, df, symbol, model)
    if sig == -1 or float(conf) < float(conf_threshold):
        logger.debug(
            "SKIP_LOW_SIGNAL",
            extra={"symbol": symbol, "sig": sig, "conf": conf},
        )
        return -1, 0.0, ""
    return int(sig), float(conf), str(strat)


def generate_directional_signals(df: Any) -> Any:
    """Return directional +1/-1/0 signals from a ``price`` series."""

    price = df["price"]
    diff = price.diff().fillna(0)
    return diff.apply(lambda value: 1 if value > 0 else (-1 if value < 0 else 0))


__all__ = ["evaluate_signal_and_confirm", "generate_directional_signals"]
