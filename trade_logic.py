# AI-AGENT-REF: basic trade utilities

import random
import metrics_logger
from logger import get_logger
from capital_scaling import drawdown_adjusted_kelly

log = get_logger(__name__)


def should_enter_trade(price_data, signals, risk_params):
    """Determine whether a trade entry conditions are met."""
    # AI-AGENT-REF: improved evaluation for unit tests
    recent_gain = (price_data[-1] - price_data[-2]) / price_data[-2]
    signal_strength = signals.get("signal_strength", 0)
    max_risk = risk_params.get("max_risk", 0.02)
    return signal_strength > 0.7 and recent_gain > 0 and max_risk < 0.05

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


def pyramiding_logic(current_position: float, profit_in_atr: float, base_size: float) -> float:
    """Return new position size applying pyramiding rules."""
    if profit_in_atr > 1.0 and current_position < 2 * base_size:
        new_pos = current_position + 0.25 * base_size
        metrics_logger.log_pyramid_add("generic", new_pos)
        return new_pos
    return current_position


# AI-AGENT-REF: execute trade with drawdown-aware Kelly sizing
def execute_trade(signal: int, position_size: float, price: float, equity_peak: float, account_value: float, raw_kelly: float) -> None:
    adj_kelly = drawdown_adjusted_kelly(account_value, equity_peak, raw_kelly)
    final_size = position_size * adj_kelly
    if signal == 1:
        log.info("BUY %s at %s", final_size, price)
    elif signal == -1:
        log.info("SELL %s at %s", final_size, price)
    else:
        log.info("HOLD")
