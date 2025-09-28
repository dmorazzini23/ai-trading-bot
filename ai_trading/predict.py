from __future__ import annotations
from functools import lru_cache
from ai_trading.features import prepare as feature_prepare
from ai_trading.net.http import HTTPSession, get_http_session
from ai_trading.exc import RequestException
from ai_trading.utils.http import clamp_request_timeout

try:
    from cachetools import TTLCache as _BaseTTLCache

    class _SentimentTTLCache(_BaseTTLCache):
        def __init__(self, maxsize=1000, ttl=3600):
            super().__init__(maxsize=maxsize, ttl=ttl)
            if not hasattr(self, "maxsize"):
                self.maxsize = maxsize

        def __setitem__(self, key, value):  # noqa: D401
            super().__setitem__(key, value)
            while len(self) > self.maxsize:
                try:
                    first_key = next(iter(self))
                except StopIteration:
                    break
                if first_key == key:
                    continue
                try:
                    super().__delitem__(first_key)
                except KeyError:
                    break

    _CACHETOOLS_AVAILABLE = True
    _sentiment_cache = _SentimentTTLCache(maxsize=1000, ttl=3600)
except ImportError:
    _CACHETOOLS_AVAILABLE = False
    _sentiment_cache: dict[str, float] = {}

_HTTP: HTTPSession = get_http_session()

@lru_cache(maxsize=1024)
def predict(path: str):
    """Minimal prediction interface used in tests."""
    import pandas as pd

    df = pd.read_csv(path)
    features = feature_prepare.prepare_indicators(df)
    model = load_model('default')
    pred = model.predict(features)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0][0]
    return (pred, proba)

def load_model(regime: str):
    raise NotImplementedError

def fetch_sentiment(symbol: str) -> float:
    """Fetch sentiment score with simple caching."""
    if symbol in _sentiment_cache:
        return _sentiment_cache[symbol]
    score = 0.0
    try:
        resp = _HTTP.get(
            f"https://example.com/{symbol}", timeout=clamp_request_timeout(10)
        )
        resp.raise_for_status()
        data = resp.json()
        score = float(data.get("score", 0.0))
    except (RequestException, TimeoutError):
        score = 0.0
    target_cache = _sentiment_cache
    target_cache[symbol] = score
    if len(target_cache) > 1000:
        try:
            excess = len(target_cache) - 1000
            for _ in range(excess):
                first_key = next(iter(target_cache))
                try:
                    del target_cache[first_key]
                except KeyError:
                    break
        except StopIteration:
            pass
    return score
__all__ = ['predict', 'load_model', 'fetch_sentiment', '_sentiment_cache', '_CACHETOOLS_AVAILABLE']
