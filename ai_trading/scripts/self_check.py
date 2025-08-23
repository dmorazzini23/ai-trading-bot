import os
from ai_trading.alpaca_api import _bars_time_window, get_bars_df
from ai_trading.utils.optional_import import optional_import
TimeFrame = optional_import('alpaca_trade_api.rest', 'TimeFrame')

def main() -> None:
    feed = os.getenv('ALPACA_DATA_FEED', 'iex')
    try:
        df_day = get_bars_df('SPY', TimeFrame.Day)
        df_min = get_bars_df('SPY', TimeFrame.Minute)
        start, end = _bars_time_window(TimeFrame.Day)
        {'msg': 'SELF_CHECK', 'feed': feed, 'spy_day_rows': len(df_day), 'spy_min_rows': len(df_min), 'start': start, 'end': end}
    except (KeyError, ValueError, TypeError):
        raise SystemExit(1)
if __name__ == '__main__':
    main()