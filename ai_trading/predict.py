from __future__ import annotations
from functools import lru_cache
from ai_trading.features import prepare as feature_prepare
from ai_trading.net.http import HTTPSession, get_http_session
from ai_trading.exc import RequestException
from ai_trading.utils.http import clamp_request_timeout

try:
    from cachetools import TTLCache

    _CACHETOOLS_AVAILABLE = True
    _sentiment_cache = TTLCache(maxsize=1000, ttl=3600)
except Exception:
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
    if _CACHETOOLS_AVAILABLE:
        _sentiment_cache[symbol] = score
    else:
        if len(_sentiment_cache) >= 1000:
            first_key = next(iter(_sentiment_cache), None)
            if first_key is not None:
                del _sentiment_cache[first_key]
        _sentiment_cache[symbol] = score
    return score
__all__ = ['predict', 'load_model', 'fetch_sentiment', '_sentiment_cache', '_CACHETOOLS_AVAILABLE']
