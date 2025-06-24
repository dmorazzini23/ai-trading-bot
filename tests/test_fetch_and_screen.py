import types
import pandas as pd
import datetime as dt

import bot_engine as bot
import data_fetcher
from utils import health_check


def _stub_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "volume": [100],
        },
        index=[pd.Timestamp("2024-01-01T09:30:00Z")],
    )


if __name__ == "__main__":
    df = _stub_df()
    data_fetcher.get_minute_df = lambda symbol, start_date, end_date: df
    data_fetcher.get_daily_df = lambda symbol, start, end: df
    ctx = types.SimpleNamespace(data_fetcher=data_fetcher)
    result = data_fetcher.get_minute_df("AAPL", dt.date.today(), dt.date.today())
    if result is None:
        result = data_fetcher.get_daily_df("AAPL", dt.date.today(), dt.date.today())
    health_check(result, "minute")
    bot.screen_universe(["AAPL"], ctx)


