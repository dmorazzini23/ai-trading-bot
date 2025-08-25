from __future__ import annotations


class Provider:
    def __init__(self, session=None):
        self._session = session

    @staticmethod
    def _yf():
        try:
            import yfinance as yf  # local import
            return yf
        except Exception as e:
            raise ImportError(
                "Provider 'yfinance' requested but the 'yfinance' package is not installed"
            ) from e

    def fetch_ohlcv(self, symbol: str, interval: str = "1d", **kwargs):
        yf = self._yf()
        t = yf.Ticker(symbol)
        return t.history(period=kwargs.get("period", "1y"), interval=interval)

