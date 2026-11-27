from __future__ import annotations

from typing import Any


def generate_directional_signals(df: Any) -> Any:
    """Return directional +1/-1/0 signals from a ``price`` series."""

    price = df["price"]
    diff = price.diff().fillna(0)
    return diff.apply(lambda value: 1 if value > 0 else (-1 if value < 0 else 0))


__all__ = ["generate_directional_signals"]
