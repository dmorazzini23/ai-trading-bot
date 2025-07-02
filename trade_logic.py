# AI-AGENT-REF: basic trade utilities

import random

def compute_order_price(symbol_data):
    raw_price = extract_price(symbol_data)
    price = max(raw_price, 1e-3)
    _, slipped = simulate_execution(price, 1)
    return slipped


def simulate_execution(price: float, qty: int) -> tuple[int, float]:
    """Return filled quantity and price after slippage and partial fill."""

    if qty <= 0 or price <= 0:
        return 0, price
    slippage = random.uniform(-0.0002, 0.0002)
    fill_price = price * (1 + slippage)
    fill_ratio = random.uniform(0.9, 1.0)
    filled_qty = max(1, int(qty * fill_ratio))
    return filled_qty, fill_price
