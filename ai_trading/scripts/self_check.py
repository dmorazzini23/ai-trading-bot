import datetime as dt
import os
from zoneinfo import ZoneInfo

from alpaca.common.exceptions import APIError
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from ai_trading.config.management import get_env, validate_required_env
from ai_trading.logging import logger


def _bars_time_window(timeframe: TimeFrame) -> tuple[dt.datetime, dt.datetime]:
    now = dt.datetime.now(tz=ZoneInfo("UTC"))
    end = now - dt.timedelta(minutes=1)
    if timeframe == TimeFrame.Day:
        days = int(os.getenv("DATA_LOOKBACK_DAYS_DAILY", 200))
    else:
        days = int(os.getenv("DATA_LOOKBACK_DAYS_MINUTE", 5))
    start = end - dt.timedelta(days=days)
    return start, end


def main() -> None:
    feed = get_env("ALPACA_DATA_FEED", "iex")
    if feed.lower() == "sip" and not get_env("ALPACA_ALLOW_SIP", "0", cast=bool):
        logger.warning("SIP_FEED_DISABLED", extra={"requested": "sip", "using": "iex"})
        feed = "iex"
    client = StockHistoricalDataClient(
        get_env("ALPACA_API_KEY"),
        get_env("ALPACA_SECRET_KEY"),
    )
    try:
        req_day = StockBarsRequest(
            symbol_or_symbols="SPY",
            timeframe=TimeFrame.Day,
            feed=feed,
        )
        req_min = StockBarsRequest(
            symbol_or_symbols="SPY",
            timeframe=TimeFrame.Minute,
            feed=feed,
        )
        df_day = client.get_stock_bars(req_day).df
        df_min = client.get_stock_bars(req_min).df
        start, end = _bars_time_window(TimeFrame.Day)
        {
            "msg": "SELF_CHECK",
            "feed": feed,
            "spy_day_rows": len(df_day),
            "spy_min_rows": len(df_min),
            "start": start.isoformat(),
            "end": end.isoformat(),
        }
    except (APIError, KeyError, ValueError, TypeError):
        raise SystemExit(1)


if __name__ == "__main__":
    if not get_env("PYTEST_RUNNING", "0", cast=bool):
        snapshot = validate_required_env()
        logger.debug("ENV_VARS_MASKED", extra=snapshot)
    main()
