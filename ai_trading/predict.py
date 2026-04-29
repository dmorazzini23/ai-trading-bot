from __future__ import annotations
import importlib
from functools import lru_cache
from typing import MutableMapping, cast
from ai_trading.net.http import HTTPSession, get_http_session
from ai_trading.exc import RequestException
from ai_trading.utils.http import clamp_request_timeout

_sentiment_cache: MutableMapping[str, float]

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
    _sentiment_cache = {}

_HTTP: HTTPSession | None = None


def _get_predict_http_session() -> HTTPSession:
    """Return the prediction HTTP session, creating it only when needed."""
    global _HTTP
    if _HTTP is None:
        _HTTP = get_http_session()
    return _HTTP


def reset_predict_runtime_cache() -> None:
    """Clear lazily initialized prediction runtime resources."""
    global _HTTP
    _HTTP = None

@lru_cache(maxsize=1024)
def predict(path: str):
    """Minimal prediction interface used in tests."""
    import pandas as pd

    df = pd.read_csv(path)
    feature_prepare = importlib.import_module("ai_trading.features.prepare")
    features = feature_prepare.prepare_indicators(df)
    latest_features = features.iloc[[-1]] if hasattr(features, "iloc") else features[-1:]
    model = load_model('default')
    pred = model.predict(latest_features)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(latest_features)
        row = probabilities[0]
        classes = getattr(model, "classes_", None)
        if classes is not None and len(classes) == len(row):
            positive_idx = next((idx for idx, value in enumerate(classes) if value == 1), None)
            proba = row[positive_idx if positive_idx is not None else len(row) - 1]
        else:
            proba = row[1] if len(row) > 1 else row[0]
    return (pred, proba)

def load_model(regime: str):
    raise NotImplementedError

def fetch_sentiment(symbol: str) -> float:
    """Fetch sentiment score with simple caching."""
    if symbol in _sentiment_cache:
        return float(_sentiment_cache[symbol])
    score = 0.0
    try:
        resp = _get_predict_http_session().get(
            f"https://example.com/{symbol}", timeout=clamp_request_timeout(10)
        )
        resp.raise_for_status()
        data = resp.json()
        score = float(data.get("score", 0.0))
    except (RequestException, TimeoutError):
        score = 0.0
    target_cache = cast(MutableMapping[str, float], _sentiment_cache)
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
