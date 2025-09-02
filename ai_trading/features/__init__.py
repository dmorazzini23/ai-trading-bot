"""
Feature engineering public API.
"""
from .indicators import compute_macd, compute_macds, compute_vwap, compute_atr, ensure_columns

def build_features_pipeline(df, symbol: str):
    """
    Minimal pipeline expected by tests:
    - ensure expected columns
    - compute MACD/MACD signal, ATR (if possible), VWAP
    """
    df = compute_macd(df)
    df = compute_macds(df)
    df = compute_atr(df)
    df = compute_vwap(df)
    df = ensure_columns(df, symbol=symbol)
    return df
__all__ = ['build_features_pipeline']
