
# SYMBOL UNIVERSE PATCH
# Add this to your bot configuration to override the symbol universe

import os
import json

def get_symbol_universe():
    """Get symbol universe with SQ fix applied"""
    
    # Check for override file
    if os.path.exists('symbol_override.json'):
        with open('symbol_override.json', 'r') as f:
            override = json.load(f)
        return override['universe']
    
    # Check environment variable
    env_symbols = os.getenv('TRADING_SYMBOLS')
    if env_symbols:
        return env_symbols.split(',')
    
    # Default universe (with SQ replaced)
    return ['AMD', 'AMZN', 'BABA', 'CRM', 'CVX', 'GOOGL', 'IWM', 'JNJ', 'JPM', 'KO', 'META', 'MSFT', 'NFLX', 'NVDA', 'PG', 'PLTR', 'QQQ', 'SHOP', 'SPY', 'PYPL', 'TSLA', 'UBER', 'XOM']

# Use this function wherever symbols are loaded
SYMBOL_UNIVERSE = get_symbol_universe()
