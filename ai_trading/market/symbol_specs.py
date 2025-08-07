"""
Symbol specifications for tick sizes and lot sizes.

Provides centralized mapping of symbol trading specifications for
precise order sizing and execution.
"""

from decimal import Decimal
from typing import Dict, NamedTuple, Optional
import logging

logger = logging.getLogger(__name__)


class SymbolSpec(NamedTuple):
    """Trading specifications for a symbol."""
    tick: Decimal  # Minimum price increment
    lot: int  # Minimum quantity increment
    multiplier: int = 1  # Contract multiplier (futures/options)


# AI-AGENT-REF: Default symbol specifications - extend as needed
DEFAULT_SYMBOL_SPECS: Dict[str, SymbolSpec] = {
    # Major Equities - penny tick
    'AAPL': SymbolSpec(tick=Decimal('0.01'), lot=1),
    'MSFT': SymbolSpec(tick=Decimal('0.01'), lot=1),
    'GOOGL': SymbolSpec(tick=Decimal('0.01'), lot=1),
    'AMZN': SymbolSpec(tick=Decimal('0.01'), lot=1),
    'TSLA': SymbolSpec(tick=Decimal('0.01'), lot=1),
    'NVDA': SymbolSpec(tick=Decimal('0.01'), lot=1),
    'META': SymbolSpec(tick=Decimal('0.01'), lot=1),
    'BRK.B': SymbolSpec(tick=Decimal('0.01'), lot=1),
    'UNH': SymbolSpec(tick=Decimal('0.01'), lot=1),
    'JNJ': SymbolSpec(tick=Decimal('0.01'), lot=1),
    
    # Major ETFs
    'SPY': SymbolSpec(tick=Decimal('0.01'), lot=1),
    'QQQ': SymbolSpec(tick=Decimal('0.01'), lot=1),
    'IWM': SymbolSpec(tick=Decimal('0.01'), lot=1),
    'EFA': SymbolSpec(tick=Decimal('0.01'), lot=1),
    'VTI': SymbolSpec(tick=Decimal('0.01'), lot=1),
    'GLD': SymbolSpec(tick=Decimal('0.01'), lot=1),
    'SLV': SymbolSpec(tick=Decimal('0.01'), lot=1),
    'TLT': SymbolSpec(tick=Decimal('0.01'), lot=1),
    
    # Crypto ETFs
    'BTCUSD': SymbolSpec(tick=Decimal('0.01'), lot=1),
    'ETHUSD': SymbolSpec(tick=Decimal('0.01'), lot=1),
    
    # Bond ETFs
    'AGG': SymbolSpec(tick=Decimal('0.01'), lot=1),
    'LQD': SymbolSpec(tick=Decimal('0.01'), lot=1),
    'HYG': SymbolSpec(tick=Decimal('0.01'), lot=1),
    
    # Sector ETFs
    'XLF': SymbolSpec(tick=Decimal('0.01'), lot=1),
    'XLK': SymbolSpec(tick=Decimal('0.01'), lot=1),
    'XLE': SymbolSpec(tick=Decimal('0.01'), lot=1),
    'XLI': SymbolSpec(tick=Decimal('0.01'), lot=1),
    'XLV': SymbolSpec(tick=Decimal('0.01'), lot=1),
}

# Default fallback for unknown symbols
DEFAULT_SPEC = SymbolSpec(tick=Decimal('0.01'), lot=1)


def get_symbol_spec(symbol: str) -> SymbolSpec:
    """
    Get trading specifications for a symbol.
    
    Args:
        symbol: Symbol to look up
        
    Returns:
        SymbolSpec with tick and lot size
    """
    symbol = symbol.upper().strip()
    spec = DEFAULT_SYMBOL_SPECS.get(symbol, DEFAULT_SPEC)
    
    if symbol not in DEFAULT_SYMBOL_SPECS:
        logger.debug(f"Using default spec for unknown symbol: {symbol}")
    
    return spec


def get_tick_size(symbol: str) -> Decimal:
    """
    Get minimum price increment for symbol.
    
    Args:
        symbol: Symbol to look up
        
    Returns:
        Minimum price increment as Decimal
    """
    return get_symbol_spec(symbol).tick


def get_lot_size(symbol: str) -> int:
    """
    Get minimum quantity increment for symbol.
    
    Args:
        symbol: Symbol to look up
        
    Returns:
        Minimum quantity increment
    """
    return get_symbol_spec(symbol).lot


def add_symbol_spec(symbol: str, tick: Decimal, lot: int, multiplier: int = 1) -> None:
    """
    Add or update symbol specification.
    
    Args:
        symbol: Symbol to add/update
        tick: Minimum price increment
        lot: Minimum quantity increment  
        multiplier: Contract multiplier
    """
    symbol = symbol.upper().strip()
    DEFAULT_SYMBOL_SPECS[symbol] = SymbolSpec(
        tick=tick, 
        lot=lot, 
        multiplier=multiplier
    )
    logger.info(f"Added/updated spec for {symbol}: tick={tick}, lot={lot}, multiplier={multiplier}")


def update_specs_from_config(specs_config: Dict[str, Dict]) -> None:
    """
    Update symbol specifications from configuration.
    
    Args:
        specs_config: Dictionary mapping symbol to spec dict
                     e.g., {'AAPL': {'tick': '0.01', 'lot': 1}}
    """
    for symbol, spec_dict in specs_config.items():
        tick = Decimal(str(spec_dict['tick']))
        lot = int(spec_dict['lot'])
        multiplier = int(spec_dict.get('multiplier', 1))
        add_symbol_spec(symbol, tick, lot, multiplier)