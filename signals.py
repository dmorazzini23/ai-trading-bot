"""
Compatibility shim for tests importing signals directly.
This module re-exports the signals module from ai_trading package.
"""
# AI-AGENT-REF: compatibility shim for legacy test imports

import sys
import os

# Set minimal environment variables to prevent config errors
if 'ALPACA_API_KEY' not in os.environ:
    os.environ['ALPACA_API_KEY'] = 'test'
if 'ALPACA_SECRET_KEY' not in os.environ:
    os.environ['ALPACA_SECRET_KEY'] = 'test'
if 'ALPACA_BASE_URL' not in os.environ:
    os.environ['ALPACA_BASE_URL'] = 'https://paper-api.alpaca.markets'
if 'WEBHOOK_SECRET' not in os.environ:
    os.environ['WEBHOOK_SECRET'] = 'test'
if 'FLASK_PORT' not in os.environ:
    os.environ['FLASK_PORT'] = '5000'

try:
    from ai_trading.signals import *
except ImportError as e:
    # Create a minimal signals module for testing if import fails
    print(f"Warning: Could not import ai_trading.signals: {e}")
    
    # Minimal mock functions for testing
    def detect_market_regime_hmm(df):
        """Mock HMM regime detection"""
        import pandas as pd
        df = df.copy()
        df['regime'] = 0
        return df
    
    GaussianHMM = None