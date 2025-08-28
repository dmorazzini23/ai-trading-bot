import datetime as dt
import os
from zoneinfo import ZoneInfo
from typing import TYPE_CHECKING

from ai_trading.alpaca_api import (
    get_api_error_cls,
    get_data_client_cls,
    get_stock_bars_request_cls,
    get_timeframe_cls,
)
from ai_trading.config.management import get_env, validate_required_env
from ai_trading.logging import logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from alpaca.data.timeframe import TimeFrame  # type: ignore


def _bars_time_window(timeframe: "TimeFrame") -> tuple[dt.datetime, dt.datetime]:
    TimeFrame = get_timeframe_cls()
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
    APIError = get_api_error_cls()
    TimeFrame = get_timeframe_cls()
    StockBarsRequest = get_stock_bars_request_cls()
    DataClient = get_data_client_cls()
    client = DataClient(
        api_key=get_env("ALPACA_API_KEY"),
        secret_key=get_env("ALPACA_SECRET_KEY"),
        base_url=get_env("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
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
        env_count = len(validate_required_env())
        logger.debug("Validated %d environment variables", env_count)
    main()
