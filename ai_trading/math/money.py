"""
Exact money math using Decimal for profit-critical P&L calculations.

Provides Money class for precise financial calculations, avoiding float
arithmetic errors that can cause silent P&L drag.
"""
from ai_trading.logging import get_logger
from decimal import ROUND_HALF_EVEN, Decimal, getcontext
from typing import TYPE_CHECKING, Union
if TYPE_CHECKING:
    pass
getcontext().prec = 28
logger = get_logger(__name__)
Number = Union[int, float, str, Decimal, 'Money']

class Money:
    """
    Exact money representation using Decimal arithmetic.

    Ensures precise calculations for cash/price/P&L computations and
    provides quantization to tick/lot sizes for order execution.
    """

    def __init__(self, amount: Number, tick: Decimal | None=None):
        """
        Initialize Money with precise decimal amount.

        Args:
            amount: Numeric amount to store as Decimal
            tick: Optional tick size for automatic quantization
        """
        self._amount = to_decimal(amount)
        self._tick = tick
        if self._tick is not None:
            self._amount = self._amount.quantize(self._tick, rounding=ROUND_HALF_EVEN)

    @property
    def amount(self) -> Decimal:
        """Get the underlying Decimal amount."""
        return self._amount

    def quantize(self, tick: Decimal) -> 'Money':
        """
        Quantize to given tick size using banker's rounding.

        Args:
            tick: Tick size for quantization (e.g., Decimal('0.01') for cents)

        Returns:
            New Money object with quantized amount
        """
        quantized = self._amount.quantize(tick, rounding=ROUND_HALF_EVEN)
        return Money(quantized)

    def __add__(self, other: Union['Money', Number]) -> 'Money':
        """Add Money or number, returning Money."""
        if isinstance(other, Money):
            return Money(self._amount + other._amount)
        return Money(self._amount + to_decimal(other))

    def __radd__(self, other: Union['Money', Number]) -> 'Money':
        """Right addition."""
        return self.__add__(other)

    def __sub__(self, other: Union['Money', Number]) -> 'Money':
        """Subtract Money or number, returning Money."""
        if isinstance(other, Money):
            return Money(self._amount - other._amount)
        return Money(self._amount - to_decimal(other))

    def __rsub__(self, other: Union['Money', Number]) -> 'Money':
        """Right subtraction."""
        return Money(to_decimal(other) - self._amount)

    def __mul__(self, other: Union['Money', Number]) -> Union['Money', Decimal]:
        """Multiply Money by number, returning Money or Decimal."""
        if isinstance(other, Money):
            return self._amount * other._amount
        return Money(self._amount * to_decimal(other))

    def __rmul__(self, other: Union['Money', Number]) -> 'Money':
        """Right multiplication."""
        return Money(to_decimal(other) * self._amount)

    def __truediv__(self, other: Union['Money', Number]) -> Union['Money', Decimal]:
        """Divide Money by number, returning Money or Decimal."""
        if isinstance(other, Money):
            return self._amount / other._amount
        return Money(self._amount / to_decimal(other))

    def __rtruediv__(self, other: Union['Money', Number]) -> Decimal:
        """Right division."""
        return to_decimal(other) / self._amount

    def __neg__(self) -> 'Money':
        """Negate Money."""
        return Money(-self._amount)

    def __abs__(self) -> 'Money':
        """Absolute value of Money."""
        return Money(abs(self._amount))

    def __eq__(self, other: Union['Money', Number]) -> bool:
        """Check equality."""
        if isinstance(other, Money):
            return self._amount == other._amount
        return self._amount == to_decimal(other)

    def __lt__(self, other: Union['Money', Number]) -> bool:
        """Less than comparison."""
        if isinstance(other, Money):
            return self._amount < other._amount
        return self._amount < to_decimal(other)

    def __le__(self, other: Union['Money', Number]) -> bool:
        """Less than or equal comparison."""
        if isinstance(other, Money):
            return self._amount <= other._amount
        return self._amount <= to_decimal(other)

    def __gt__(self, other: Union['Money', Number]) -> bool:
        """Greater than comparison."""
        if isinstance(other, Money):
            return self._amount > other._amount
        return self._amount > to_decimal(other)

    def __ge__(self, other: Union['Money', Number]) -> bool:
        """Greater than or equal comparison."""
        if isinstance(other, Money):
            return self._amount >= other._amount
        return self._amount >= to_decimal(other)

    def __str__(self) -> str:
        """String representation."""
        return str(self._amount)

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"Money('{self._amount}')"

    def __float__(self) -> float:
        """Convert to float (use with caution)."""
        return float(self._amount)

    def __int__(self) -> int:
        """Convert to int (truncated)."""
        return int(self._amount)

def to_decimal(value: Number) -> Decimal:
    """
    Convert number to Decimal with proper handling.

    Args:
        value: Number to convert

    Returns:
        Decimal representation
    """
    if isinstance(value, Decimal):
        return value
    elif isinstance(value, Money):
        return value._amount
    elif isinstance(value, int | str):
        return Decimal(value)
    elif isinstance(value, float):
        return Decimal(str(value))
    else:
        raise TypeError(f'Cannot convert {type(value)} to Decimal')

def round_to_tick(price: Number, tick_size: Decimal) -> Money:
    """
    Round price to nearest tick size.

    Args:
        price: Price to round
        tick_size: Minimum price increment

    Returns:
        Money object rounded to tick
    """
    return Money(price).quantize(tick_size)

def round_to_lot(quantity: Number, lot_size: int) -> int:
    """
    Round quantity to nearest lot size.

    Args:
        quantity: Quantity to round
        lot_size: Minimum quantity increment

    Returns:
        Integer quantity rounded to lot
    """
    decimal_qty = to_decimal(quantity)
    lots = (decimal_qty / lot_size).quantize(Decimal('1'), rounding=ROUND_HALF_EVEN)
    return int(lots * lot_size)