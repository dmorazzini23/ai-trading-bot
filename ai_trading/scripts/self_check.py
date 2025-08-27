from ai_trading.alpaca_api import _bars_time_window, get_bars_df
from ai_trading.config.management import get_env, validate_required_env
from ai_trading.logging import logger

from alpaca.data.timeframe import TimeFrame


def main() -> None:
    feed = get_env('ALPACA_DATA_FEED', 'iex')
    try:
        df_day = get_bars_df('SPY', TimeFrame.Day)
        df_min = get_bars_df('SPY', TimeFrame.Minute)
        start, end = _bars_time_window(TimeFrame.Day)
        {
            'msg': 'SELF_CHECK',
            'feed': feed,
            'spy_day_rows': len(df_day),
            'spy_min_rows': len(df_min),
            'start': start,
            'end': end,
        }
    except (KeyError, ValueError, TypeError):
        raise SystemExit(1)


if __name__ == '__main__':
    if not get_env('PYTEST_RUNNING', '0', cast=bool):
        snapshot = validate_required_env()
        logger.debug('ENV_VARS_MASKED', extra=snapshot)
    main()

