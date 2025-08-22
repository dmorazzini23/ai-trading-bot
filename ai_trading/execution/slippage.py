from __future__ import annotations

# AI-AGENT-REF: deterministic slippage estimator promoted from test helper

def estimate(price: float, side: str, bps: float = 0.0) -> float:
    """
    Deterministic slippage estimator.
    - price: last/mark price
    - side : 'buy' or 'sell'
    - bps  : basis points to add/subtract (default 0 for CI determinism)
    """
    if not isinstance(price, int | float) or price <= 0:
        raise ValueError("price must be a positive number")
    side_l = (side or "").lower()
    if side_l not in {"buy", "sell"}:
        raise ValueError("side must be 'buy' or 'sell'")
    adj = price * (bps / 10_000.0)
    return price + adj if side_l == "buy" else price - adj
