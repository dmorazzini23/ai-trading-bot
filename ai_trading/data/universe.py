import os
from importlib.resources import files as pkg_files
from ai_trading.utils.lazy_imports import load_pandas
from ai_trading.logging import logger
from ai_trading.utils.universe import normalize_symbol
from ai_trading.paths import TICKERS_FILE_PATH

# Lazy pandas proxy
pd = load_pandas()

# Last-resort tickers if no CSV can be located
FALLBACK_TICKERS = [
    normalize_symbol(s) for s in ["SPY", "AAPL", "MSFT", "AMZN", "GOOGL"]
]

def locate_tickers_csv() -> str | None:
    env = os.getenv('AI_TRADING_TICKERS_CSV')
    if env and os.path.isfile(env):
        return os.path.abspath(env)
    # Check ai_trading.paths.TICKERS_FILE_PATH
    path = os.path.abspath(os.path.expanduser(os.path.normpath(str(TICKERS_FILE_PATH))))
    if os.path.isfile(path):
        return path
    try:
        p = pkg_files('ai_trading.data').joinpath('tickers.csv')
        if p.is_file():
            return str(p)
    except ModuleNotFoundError:
        pass
    cwd = os.path.join(os.getcwd(), "tickers.csv")
    if os.path.isfile(cwd):
        return cwd
    return None

def load_universe() -> list[str]:
    path = locate_tickers_csv()
    if not path:
        logger.error(
            'TICKERS_FILE_MISSING',
            extra={'path': 'tickers.csv', 'fallback': 'default_list'},
        )
        return FALLBACK_TICKERS.copy()
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, ValueError) as e:
        logger.error('TICKERS_FILE_READ_FAILED', extra={'path': path, 'error': str(e)})
        return []
    # Normalize symbols so downstream modules see provider-ready tickers
    symbols = [normalize_symbol(str(s)) for s in df.iloc[:, 0].tolist() if str(s).strip()]
    logger.info('TICKERS_SOURCE', extra={'path': path, 'count': len(symbols)})
    return symbols
