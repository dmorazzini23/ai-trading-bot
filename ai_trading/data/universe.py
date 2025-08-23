import os
from importlib.resources import files as pkg_files

import pandas as pd

from ai_trading.logging import logger  # AI-AGENT-REF: structured logging


def locate_tickers_csv() -> str | None:
    env = os.getenv("AI_TRADER_TICKERS_CSV")
    if env and os.path.isfile(env):
        return os.path.abspath(env)
    try:
        p = pkg_files("ai_trading.data").joinpath("tickers.csv")
        if p.is_file():
            return str(p)
    # AI-AGENT-REF: narrow broad catch to missing package only
    except ModuleNotFoundError:  # pragma: no cover - defensive
        pass
    return None


def load_universe() -> list[str]:
    path = locate_tickers_csv()
    if not path:
        logger.error("TICKERS_FILE_MISSING", extra={"path": "ai_trading/data/tickers.csv", "fallback": "none"})
        return []
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, ValueError) as e:  # AI-AGENT-REF: guard CSV read
        logger.error(
            "TICKERS_FILE_READ_FAILED",
            extra={"path": path, "error": str(e)},
        )
        return []
    symbols = [str(s).strip().upper() for s in df.iloc[:, 0].tolist() if str(s).strip()]
    logger.info("TICKERS_SOURCE", extra={"path": path, "count": len(symbols)})
    return symbols
