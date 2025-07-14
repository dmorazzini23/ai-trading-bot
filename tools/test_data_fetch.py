import argparse
from datetime import datetime, timedelta, timezone

from logger import init_logger
from data_fetcher import get_minute_df


def main(symbol: str) -> None:
    logger = init_logger("logs/test_data_fetch.log")
    start = datetime.now(timezone.utc) - timedelta(days=1)
    end = datetime.now(timezone.utc)
    try:
        df = get_minute_df(symbol, start, end)
        rows = len(df) if df is not None else 0
        logger.info(
            "TEST_DATA_FETCH",
            extra={"symbol": symbol, "rows": rows, "status": "success"},
        )
        if df is not None:
            logger.debug("Head for %s:\n%s", symbol, df.head())
    except Exception as exc:
        logger.error("TEST_DATA_FETCH_FAILED %s: %s", symbol, exc, exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test data fetch for symbol")
    parser.add_argument("symbol")
    args = parser.parse_args()
    main(args.symbol)
