"""Symbol specifications for tick, lot, and trading hours.

Provides centralized mapping of symbol trading specifications for
precise order sizing and execution.
"""

from decimal import Decimal
from typing import NamedTuple

from ai_trading.logging import get_logger


logger = get_logger(__name__)


class SymbolSpec(NamedTuple):
    """Trading specifications for a symbol."""

    tick: Decimal
    lot: int
    multiplier: int = 1
    trading_hours: str = "09:30-16:00"


DEFAULT_SYMBOL_SPECS: dict[str, SymbolSpec] = {
    "AAPL": SymbolSpec(tick=Decimal("0.01"), lot=1),
    "MSFT": SymbolSpec(tick=Decimal("0.01"), lot=1),
    "GOOGL": SymbolSpec(tick=Decimal("0.01"), lot=1),
    "AMZN": SymbolSpec(tick=Decimal("0.01"), lot=1),
    "TSLA": SymbolSpec(tick=Decimal("0.01"), lot=1),
    "NVDA": SymbolSpec(tick=Decimal("0.01"), lot=1),
    "META": SymbolSpec(tick=Decimal("0.01"), lot=1),
    "UNH": SymbolSpec(tick=Decimal("0.01"), lot=1),
    "JNJ": SymbolSpec(tick=Decimal("0.01"), lot=1),
    "ADBE": SymbolSpec(tick=Decimal("0.01"), lot=1, trading_hours="09:30-16:00"),
    "HD": SymbolSpec(tick=Decimal("0.01"), lot=1, trading_hours="09:30-16:00"),
    "AMGN": SymbolSpec(tick=Decimal("0.01"), lot=1, trading_hours="09:30-16:00"),
    "XOM": SymbolSpec(tick=Decimal("0.01"), lot=1, trading_hours="09:30-16:00"),
    "BAC": SymbolSpec(tick=Decimal("0.01"), lot=1, trading_hours="09:30-16:00"),
    "V": SymbolSpec(tick=Decimal("0.01"), lot=1, trading_hours="09:30-16:00"),
    "COST": SymbolSpec(tick=Decimal("0.01"), lot=1),
    "NFLX": SymbolSpec(tick=Decimal("0.01"), lot=1),
    "SPY": SymbolSpec(tick=Decimal("0.01"), lot=1),
    "QQQ": SymbolSpec(tick=Decimal("0.01"), lot=1),
    "IWM": SymbolSpec(tick=Decimal("0.01"), lot=1),
    "EFA": SymbolSpec(tick=Decimal("0.01"), lot=1),
    "VTI": SymbolSpec(tick=Decimal("0.01"), lot=1),
    "GLD": SymbolSpec(tick=Decimal("0.01"), lot=1),
    "SLV": SymbolSpec(tick=Decimal("0.01"), lot=1),
    "TLT": SymbolSpec(tick=Decimal("0.01"), lot=1),
    "BTCUSD": SymbolSpec(tick=Decimal("0.01"), lot=1),
    "ETHUSD": SymbolSpec(tick=Decimal("0.01"), lot=1),
    "AGG": SymbolSpec(tick=Decimal("0.01"), lot=1),
    "LQD": SymbolSpec(tick=Decimal("0.01"), lot=1),
    "HYG": SymbolSpec(tick=Decimal("0.01"), lot=1),
    "XLF": SymbolSpec(tick=Decimal("0.01"), lot=1),
    "XLK": SymbolSpec(tick=Decimal("0.01"), lot=1),
    "XLE": SymbolSpec(tick=Decimal("0.01"), lot=1),
    "XLI": SymbolSpec(tick=Decimal("0.01"), lot=1),
    "XLV": SymbolSpec(tick=Decimal("0.01"), lot=1),
}


DEFAULT_SPEC = SymbolSpec(tick=Decimal("0.01"), lot=1)


def get_symbol_spec(symbol: str) -> SymbolSpec:
    """Get trading specifications for a symbol."""

    symbol = symbol.upper().strip()
    spec = DEFAULT_SYMBOL_SPECS.get(symbol, DEFAULT_SPEC)
    if symbol not in DEFAULT_SYMBOL_SPECS:
        logger.debug(f"Using default spec for unknown symbol: {symbol}")
    return spec


def get_tick_size(symbol: str) -> Decimal:
    """Get minimum price increment for symbol."""

    return get_symbol_spec(symbol).tick


def get_lot_size(symbol: str) -> int:
    """Get minimum quantity increment for symbol."""

    return get_symbol_spec(symbol).lot


def add_symbol_spec(
    symbol: str,
    tick: Decimal,
    lot: int,
    multiplier: int = 1,
    trading_hours: str = "09:30-16:00",
) -> None:
    """Add or update symbol specification."""

    symbol = symbol.upper().strip()
    DEFAULT_SYMBOL_SPECS[symbol] = SymbolSpec(
        tick=tick, lot=lot, multiplier=multiplier, trading_hours=trading_hours
    )
    logger.info(
        "Added/updated spec for %s: tick=%s, lot=%s, multiplier=%s, trading_hours=%s",
        symbol,
        tick,
        lot,
        multiplier,
        trading_hours,
    )


def update_specs_from_config(specs_config: dict[str, dict]) -> None:
    """Update symbol specifications from configuration."""

    for symbol, spec_dict in specs_config.items():
        tick = Decimal(str(spec_dict["tick"]))
        lot = int(spec_dict["lot"])
        multiplier = int(spec_dict.get("multiplier", 1))
        trading_hours = str(spec_dict.get("trading_hours", "09:30-16:00"))
        add_symbol_spec(symbol, tick, lot, multiplier, trading_hours)


def get_tick_by_symbol() -> dict[str, Decimal]:
    """Get dictionary mapping symbol to tick size."""

    return {symbol: spec.tick for symbol, spec in DEFAULT_SYMBOL_SPECS.items()}


def get_lot_by_symbol() -> dict[str, int]:
    """Get dictionary mapping symbol to lot size."""

    return {symbol: spec.lot for symbol, spec in DEFAULT_SYMBOL_SPECS.items()}


TICK_BY_SYMBOL = get_tick_by_symbol()
LOT_BY_SYMBOL = get_lot_by_symbol()

