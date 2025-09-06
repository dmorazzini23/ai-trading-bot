"""Order-related type definitions."""
from enum import Enum


class OrderSide(Enum):
    """Order side enumeration including short selling."""
    BUY = "buy"
    SELL = "sell"
    SELL_SHORT = "sell_short"

    def __str__(self) -> str:
        return self.value
