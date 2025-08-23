"""
Lightweight package initializer for ai_trading.data.

Intentionally avoids importing heavy submodules at import time to prevent
circular imports (e.g., data_fetcher <-> data.bars via package __init__).

Import explicitly from submodules instead, for example:
    from ai_trading.data.bars import safe_get_stock_bars, empty_bars_dataframe
    from ai_trading.data.timeutils import ensure_utc_datetime, previous_business_day
    from ai_trading.data.universe import load_tickers
"""
__all__ = []