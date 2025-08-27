"""Prediction utilities using trained models."""
from __future__ import annotations
import argparse
import json
import logging
import os
from threading import Lock
import time
import joblib
import pandas as pd
from ai_trading.config.management import TradingConfig, reload_env
from ai_trading.utils.http import http, HTTP_TIMEOUT
from scripts.retrain_model import prepare_indicators

config = TradingConfig.from_env()
_sentiment_lock = Lock()
_last_request_time: dict[str, float] = {}
logger = logging.getLogger(__name__)
reload_env()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INACTIVE_FEATURES_FILE = os.path.join(BASE_DIR, 'inactive_features.json')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
try:
    from cachetools import TTLCache
    _CACHETOOLS_AVAILABLE = True
    _sentiment_cache = TTLCache(maxsize=1000, ttl=300)
except Exception:
    _CACHETOOLS_AVAILABLE = False
    _sentiment_cache: dict[str, tuple[float, float]] | dict[str, float] = {}
_min_request_interval = 1.0
_cache_ttl = 300

def fetch_sentiment(symbol: str) -> float:
    """Return a sentiment score for ``symbol`` using NewsAPI with rate limiting and TTL cache."""
    api_key = getattr(config, 'SENTIMENT_API_KEY', None) or config.NEWS_API_KEY
    if not api_key:
        logger.debug('No sentiment API key configured (checked SENTIMENT_API_KEY and NEWS_API_KEY)')
        return 0.0
    current_time = time.time()
    with _sentiment_lock:
        if _CACHETOOLS_AVAILABLE:
            if symbol in _sentiment_cache:
                cached_score = _sentiment_cache[symbol]
                logger.debug('Using TTL cached sentiment for %s: %.2f', symbol, cached_score)
                return cached_score
        elif symbol in _sentiment_cache:
            cached_score, cached_time = _sentiment_cache[symbol]
            if current_time - cached_time < _cache_ttl:
                logger.debug('Using manual cached sentiment for %s: %.2f', symbol, cached_score)
                return cached_score
            else:
                del _sentiment_cache[symbol]
        if symbol in _last_request_time:
            time_since_last = current_time - _last_request_time[symbol]
            if time_since_last < _min_request_interval:
                logger.warning('fetch_sentiment(%s) rate-limited → returning cached/neutral 0.0', symbol)
                if _CACHETOOLS_AVAILABLE and symbol in _sentiment_cache:
                    return _sentiment_cache[symbol]
                elif not _CACHETOOLS_AVAILABLE and symbol in _sentiment_cache:
                    return _sentiment_cache[symbol][0]
                return 0.0
        _last_request_time[symbol] = current_time
    try:
        base_url = getattr(config, 'SENTIMENT_API_URL', 'https://newsapi.org/v2/everything')
        url = f'{base_url}?q={symbol}&pageSize=5&sortBy=publishedAt&apiKey={api_key}'
        resp = http.get(url, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        arts = resp.json().get('articles', [])
        if not arts:
            score = 0.0
        else:
            score = sum((1 for a in arts if 'positive' in (a.get('title') or '').lower())) / len(arts)
        score = float(score)
        with _sentiment_lock:
            if _CACHETOOLS_AVAILABLE:
                _sentiment_cache[symbol] = score
            else:
                if len(_sentiment_cache) >= 1000:
                    oldest_key = min(_sentiment_cache.keys(), key=lambda k: _sentiment_cache[k][1])
                    del _sentiment_cache[oldest_key]
                _sentiment_cache[symbol] = (score, current_time)
        logger.debug('Fetched fresh sentiment for %s: %.2f', symbol, score)
        return score
    except (Exception, ValueError) as exc:
        logger.error('fetch_sentiment failed for %s: %s', symbol, exc)
        with _sentiment_lock:
            if _CACHETOOLS_AVAILABLE and symbol in _sentiment_cache:
                return _sentiment_cache[symbol]
            elif not _CACHETOOLS_AVAILABLE and symbol in _sentiment_cache:
                return _sentiment_cache[symbol][0]
        return 0.0

def detect_regime(df: pd.DataFrame) -> str:
    """Classify a market regime based on moving average crossovers."""
    if df is None or df.empty or 'close' not in df:
        return 'chop'
    close = df['close'].astype(float)
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    if sma50.iloc[-1] > sma200.iloc[-1]:
        return 'bull'
    if sma50.iloc[-1] < sma200.iloc[-1]:
        return 'bear'
    return 'chop'
MODEL_FILES = {'bull': os.path.join(MODELS_DIR, 'model_bull.pkl'), 'bear': os.path.join(MODELS_DIR, 'model_bear.pkl'), 'chop': os.path.join(MODELS_DIR, 'model_chop.pkl')}

def load_model(regime: str):
    """Load and return the model for the specified ``regime``."""
    path = MODEL_FILES.get(regime)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Model for regime '{regime}' not found: {path}")
    return joblib.load(path)

def predict(csv_path: str, freq: str='intraday') -> tuple[int | None, float | None]:
    """Return the predicted class and probability for the data in ``csv_path``."""
    df = pd.read_csv(csv_path)
    symbol = os.path.splitext(os.path.basename(csv_path))[0]
    try:
        import inspect
        if 'symbol' in inspect.signature(prepare_indicators).parameters:
            feat = prepare_indicators(df, freq=freq, symbol=symbol)
        else:
            feat = prepare_indicators(df, freq=freq)
    except (TypeError, ValueError, AttributeError):
        feat = prepare_indicators(df, freq=freq)
    if os.path.exists(INACTIVE_FEATURES_FILE):
        try:
            with open(INACTIVE_FEATURES_FILE, encoding='utf-8') as f:
                inactive = set(json.load(f))
            feat = feat.drop(columns=[c for c in inactive if c in feat.columns], errors='ignore')
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning('Failed loading inactive features: %s', exc)
    symbol = os.path.splitext(os.path.basename(csv_path))[0]
    feat['sentiment'] = fetch_sentiment(symbol)
    regime = detect_regime(df)
    if isinstance(regime, pd.Series):
        regime = regime.iloc[-1]
    model = load_model(regime)
    expected_features = list(model.feature_names_in_)
    missing = set(expected_features) - set(feat.columns)
    if missing:
        raise ValueError(f'Missing features: {missing}')
    features = feat[expected_features].iloc[-1:].astype(dict.fromkeys(expected_features, 'float64'))
    try:
        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0][pred]
    except (ValueError, TypeError) as exc:
        logger.error('Prediction failed for %s: %s', symbol, exc)
        return (None, None)
    logger.info('Regime: %s, Prediction: %s, Probability: %.4f', regime, pred, proba)
    from ai_trading.logging import _get_metrics_logger
    _get_metrics_logger().log_metrics({'timestamp': pd.Timestamp.now(tz='UTC').isoformat(), 'symbol': symbol, 'regime': regime, 'prediction': int(pred), 'probability': float(proba)}, filename='metrics/predictions.csv')
    return (pred, proba)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict trade signal')
    parser.add_argument('csv', help='CSV file with OHLCV data')
    parser.add_argument('--freq', choices=['daily', 'intraday'], default='intraday')
    args = parser.parse_args()
    predict(args.csv, args.freq)