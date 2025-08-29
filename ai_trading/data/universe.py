import os
from importlib.resources import files as pkg_files
from ai_trading.utils.lazy_imports import load_pandas
from ai_trading.logging import logger
from ai_trading.utils.universe import normalize_symbol
from ai_trading.paths import TICKERS_FILE_PATH

# Lazy pandas proxy
pd = load_pandas()

def locate_tickers_csv() -> str | None:
    env = os.getenv("AI_TRADING_TICKERS_CSV")
    if env and os.path.isfile(env):
        return os.path.abspath(env)
    # Check ai_trading.paths.TICKERS_FILE_PATH
    path = os.path.abspath(os.path.expanduser(os.path.normpath(str(TICKERS_FILE_PATH))))
    if os.path.isfile(path):
        return path
    try:
        p = pkg_files("ai_trading.data").joinpath("tickers.csv")
        if p.is_file():
            return str(p)
    except ModuleNotFoundError:
        pass
    cwd = os.path.join(os.getcwd(), "tickers.csv")
    if os.path.isfile(cwd):
        return cwd
    return None

def load_universe() -> list[str]:
    """Load and return normalized tickers from a required CSV file.

    The lookup path is resolved by :func:`locate_tickers_csv`. If no file can be
    located a :class:`RuntimeError` is raised. If the CSV cannot be read a log
    entry is emitted and an empty list is returned.
    """

    path = locate_tickers_csv()
    if path is None:
        logger.error("TICKERS_FILE_MISSING", extra={"path": "tickers.csv"})
        raise RuntimeError("tickers.csv not found")
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, ValueError) as e:
        logger.error(
            "TICKERS_FILE_READ_FAILED", extra={"path": path, "error": str(e)}
        )
        return []
    # Normalize symbols so downstream modules see provider-ready tickers
    symbols = [
        normalize_symbol(str(s)) for s in df.iloc[:, 0].tolist() if str(s).strip()
    ]
    logger.info("TICKERS_SOURCE", extra={"path": path, "count": len(symbols)})
    return symbols
