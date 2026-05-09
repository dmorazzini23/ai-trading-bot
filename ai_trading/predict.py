from __future__ import annotations
import importlib
from functools import lru_cache
from typing import MutableMapping, cast

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

def reset_predict_runtime_cache() -> None:
    """Clear lazily initialized prediction runtime resources."""
    _sentiment_cache.clear()

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
    """Fetch sentiment through the canonical sentiment implementation."""
    if symbol in _sentiment_cache:
        return float(_sentiment_cache[symbol])
    sentiment = importlib.import_module("ai_trading.analysis.sentiment")
    score = float(sentiment.fetch_sentiment(None, symbol))
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
