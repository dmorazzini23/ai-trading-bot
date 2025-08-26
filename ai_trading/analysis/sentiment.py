"""
Sentiment analysis module for AI trading bot.

This module provides sentiment analysis functionality using FinBERT and NewsAPI,
extracted from bot_engine.py to enable standalone imports and testing.
"""
import time as pytime
from datetime import datetime
from threading import Lock
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential, wait_random
from ai_trading.logging import logger
from ai_trading.settings import get_news_api_key
from ai_trading.config import get_settings
from ai_trading.utils.timing import HTTP_TIMEOUT
from ai_trading.config.management import get_env, validate_required_env
from ai_trading.utils.device import get_device, tensors_to_device  # AI-AGENT-REF: guard torch import

SENTIMENT_API_KEY = get_env("SENTIMENT_API_KEY", "")
DEVICE = None
_SENTIMENT_INITIALIZED = False


def _init_sentiment() -> None:
    """Initialize sentiment utilities once."""
    global _SENTIMENT_INITIALIZED, DEVICE
    if _SENTIMENT_INITIALIZED:
        return
    if not get_env("PYTEST_RUNNING", "0", cast=bool):
        validate_required_env()
    DEVICE = get_device()
    _SENTIMENT_INITIALIZED = True
    logger.debug("SENTIMENT_INIT")
_BS4 = None
_TRANSFORMERS = None
_SENT_DEPS_LOGGED: set[str] = set()

def _load_bs4(log=logger):
    global _BS4
    if _BS4 is not None:
        return _BS4
    try:
        from bs4 import BeautifulSoup
        _BS4 = BeautifulSoup
    except (ValueError, TypeError):
        if 'bs4' not in _SENT_DEPS_LOGGED:
            log.warning('SENTIMENT_OPTIONAL_DEP_MISSING', extra={'package': 'bs4'})
            _SENT_DEPS_LOGGED.add('bs4')
        _BS4 = None
    return _BS4

def _load_transformers(log=logger):
    global _TRANSFORMERS
    if _TRANSFORMERS is not None:
        return _TRANSFORMERS
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        model = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
        model.to(DEVICE)
        model.eval()
        _TRANSFORMERS = (torch, tokenizer, model)
    except (ValueError, TypeError):
        if 'transformers' not in _SENT_DEPS_LOGGED:
            log.warning('SENTIMENT_OPTIONAL_DEP_MISSING', extra={'package': 'transformers'})
            _SENT_DEPS_LOGGED.add('transformers')
        _TRANSFORMERS = None
    return _TRANSFORMERS
SENTIMENT_TTL_SEC = 600
SENTIMENT_RATE_LIMITED_TTL_SEC = 7200
SENTIMENT_FAILURE_THRESHOLD = 15
SENTIMENT_RECOVERY_TIMEOUT = 1800
SENTIMENT_MAX_RETRIES = 5
SENTIMENT_BASE_DELAY = 5
_SENTIMENT_CACHE: dict[str, tuple[float, float]] = {}
_SENTIMENT_CIRCUIT_BREAKER = {'failures': 0, 'last_failure': 0, 'state': 'closed'}
sentiment_lock = Lock()
__all__ = ['fetch_sentiment', 'predict_text_sentiment', 'analyze_text', 'sentiment_lock', '_SENTIMENT_CACHE', 'SENTIMENT_FAILURE_THRESHOLD', 'SENTIMENT_API_KEY']

def _check_sentiment_circuit_breaker() -> bool:
    """Check if sentiment circuit breaker allows requests."""
    global _SENTIMENT_CIRCUIT_BREAKER
    now = pytime.time()
    cb = _SENTIMENT_CIRCUIT_BREAKER
    if cb['state'] == 'open':
        if now - cb['last_failure'] > SENTIMENT_RECOVERY_TIMEOUT:
            cb['state'] = 'half-open'
            logger.info('Sentiment circuit breaker moved to half-open state')
            return True
        return False
    return True

def _record_sentiment_success():
    """Record successful sentiment API call."""
    global _SENTIMENT_CIRCUIT_BREAKER
    _SENTIMENT_CIRCUIT_BREAKER['failures'] = 0
    if _SENTIMENT_CIRCUIT_BREAKER['state'] == 'half-open':
        _SENTIMENT_CIRCUIT_BREAKER['state'] = 'closed'
        logger.info('Sentiment circuit breaker closed - service recovered')

def _record_sentiment_failure():
    """Record failed sentiment API call and update circuit breaker."""
    global _SENTIMENT_CIRCUIT_BREAKER
    cb = _SENTIMENT_CIRCUIT_BREAKER
    cb['failures'] += 1
    cb['last_failure'] = pytime.time()
    if cb['failures'] >= SENTIMENT_FAILURE_THRESHOLD:
        cb['state'] = 'open'
        logger.warning(f"Sentiment circuit breaker opened after {cb['failures']} failures")

@retry(stop=stop_after_attempt(SENTIMENT_MAX_RETRIES), wait=wait_exponential(multiplier=SENTIMENT_BASE_DELAY, min=SENTIMENT_BASE_DELAY, max=180) + wait_random(0, 5), retry=retry_if_exception_type((Exception,)))
def fetch_sentiment(ctx, ticker: str) -> float:
    """
    Fetch sentiment via NewsAPI + FinBERT + Form 4 signal.
    Uses a simple in-memory TTL cache to avoid hitting NewsAPI too often.
    If FinBERT isn't available, return neutral 0.0.

    Args:
        ctx: BotContext (for compatibility with bot_engine)
        ticker: Stock ticker symbol

    Returns:
        Sentiment score between -1.0 and 1.0
    """
    _init_sentiment()
    settings = get_settings()
    api_key = SENTIMENT_API_KEY or getattr(settings, 'sentiment_api_key', None) or get_news_api_key()
    if not api_key:
        logger.debug('No sentiment API key configured (checked settings.sentiment_api_key and news API key)')
        return 0.0
    now_ts = pytime.time()
    with sentiment_lock:
        cached = _SENTIMENT_CACHE.get(ticker)
        if cached:
            last_ts, last_score = cached
            cache_ttl = SENTIMENT_RATE_LIMITED_TTL_SEC if _SENTIMENT_CIRCUIT_BREAKER['state'] == 'open' else SENTIMENT_TTL_SEC
            if now_ts - last_ts < cache_ttl:
                logger.debug(f'Sentiment cache hit for {ticker} (age: {(now_ts - last_ts) / 60:.1f}m)')
                return last_score
    if not _check_sentiment_circuit_breaker():
        logger.info(f'Sentiment circuit breaker open, returning cached/neutral for {ticker}')
        with sentiment_lock:
            cached = _SENTIMENT_CACHE.get(ticker)
            if cached:
                _, last_score = cached
                logger.debug(f'Using stale cached sentiment {last_score} for {ticker}')
                return last_score
            _SENTIMENT_CACHE[ticker] = (now_ts, 0.0)
            return 0.0
    try:
        url = f'{settings.sentiment_api_url}?q={ticker}&sortBy=publishedAt&language=en&pageSize=5&apiKey={api_key}'
        resp = requests.get(url, timeout=HTTP_TIMEOUT)
        if resp.status_code == 429:
            logger.warning(f'fetch_sentiment({ticker}) rate-limited (429) → using enhanced fallback strategies')
            return _handle_rate_limit_with_enhanced_strategies(ticker)
        elif resp.status_code == 403:
            logger.warning(f'fetch_sentiment({ticker}) forbidden (403) - possible API key issue → using fallback')
            _record_sentiment_failure()
            return _get_cached_or_neutral_sentiment(ticker)
        elif resp.status_code >= 500:
            logger.warning(f'fetch_sentiment({ticker}) server error ({resp.status_code}) → using fallback')
            _record_sentiment_failure()
            return _get_cached_or_neutral_sentiment(ticker)
        resp.raise_for_status()
        payload = resp.json()
        articles = payload.get('articles', [])
        scores = []
        if articles:
            for art in articles:
                text = (art.get('title') or '') + '. ' + (art.get('description') or '')
                if text.strip():
                    scores.append(predict_text_sentiment(text))
        news_score = float(sum(scores) / len(scores)) if scores else 0.0
        form4_score = 0.0
        try:
            form4 = fetch_form4_filings(ticker)
            for filing in form4:
                if filing['type'] == 'buy' and filing['dollar_amount'] > 50000:
                    form4_score += 0.1
        except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
            logger.debug('Form4 fetch failed for %s - network error: %s', ticker, e)
        except (KeyError, ValueError, TypeError) as e:
            logger.debug('Form4 fetch failed for %s - data parsing error: %s', ticker, e)
        except (ValueError, TypeError) as e:
            logger.debug('Form4 fetch failed for %s - unexpected error: %s', ticker, e, extra={'component': 'sentiment', 'ticker': ticker, 'error_type': 'form4_fetch'})
        final_score = 0.8 * news_score + 0.2 * form4_score
        final_score = max(-1.0, min(1.0, final_score))
        _record_sentiment_success()
        with sentiment_lock:
            _SENTIMENT_CACHE[ticker] = (now_ts, final_score)
        return final_score
    except (ValueError, TypeError) as e:
        logger.warning(f'Sentiment API request failed for {ticker}: {e}')
        _record_sentiment_failure()
        with sentiment_lock:
            cached = _SENTIMENT_CACHE.get(ticker)
            if cached:
                _, last_score = cached
                logger.debug(f'Using cached sentiment fallback {last_score} for {ticker}')
                return last_score
            _SENTIMENT_CACHE[ticker] = (now_ts, 0.0)
            return 0.0
    except (ValueError, TypeError) as e:
        logger.error(f'Unexpected error fetching sentiment for {ticker}: {e}')
        _record_sentiment_failure()
        with sentiment_lock:
            _SENTIMENT_CACHE[ticker] = (now_ts, 0.0)
        return 0.0

def _handle_rate_limit_with_enhanced_strategies(ticker: str) -> float:
    """
    Enhanced rate limiting handling with multiple fallback strategies.

    AI-AGENT-REF: Implements exponential backoff and multiple fallback data sources for critical sentiment rate limiting fix.
    """
    _record_sentiment_failure()
    fallback_sources = [_try_alternative_sentiment_sources, _try_cached_similar_symbols, _try_sector_sentiment_proxy, _get_cached_or_neutral_sentiment]
    for fallback_func in fallback_sources:
        try:
            result = fallback_func(ticker)
            if result is not None:
                logger.info(f'SENTIMENT_FALLBACK_SUCCESS | ticker={ticker} source={fallback_func.__name__} value={result}')
                return result
        except (ValueError, TypeError) as e:
            logger.debug(f'SENTIMENT_FALLBACK_FAILED | ticker={ticker} source={fallback_func.__name__} error={e}')
            continue
    with sentiment_lock:
        cached = _SENTIMENT_CACHE.get(ticker)
        if cached:
            cache_ts, sentiment_val = cached
            if pytime.time() - cache_ts < SENTIMENT_RATE_LIMITED_TTL_SEC:
                logger.info(
                    f'SENTIMENT_RATE_LIMIT_USING_EXTENDED_CACHE | ticker={ticker} '
                    f'age_hours={int((pytime.time() - cache_ts) / 3600)}'
                )
                return sentiment_val
        now_ts = pytime.time()
        _SENTIMENT_CACHE[ticker] = (now_ts, 0.0)
    logger.warning('SENTIMENT_RATE_LIMIT_ALL_FALLBACKS_EXHAUSTED', extra={'ticker': ticker, 'fallback_strategies_tried': len(fallback_sources), 'cache_ttl_hours': SENTIMENT_RATE_LIMITED_TTL_SEC / 3600, 'recommendation': 'Consider upgrading NewsAPI plan, adding alternative sentiment sources, or reviewing rate limits'})
    return 0.0

def _try_alternative_sentiment_sources(ticker: str) -> float | None:
    """Try alternative sentiment data sources when primary is rate limited."""
    alt_api_key = get_env("ALTERNATIVE_SENTIMENT_API_KEY")
    alt_api_url = get_env("ALTERNATIVE_SENTIMENT_API_URL")
    primary_url = get_env("SENTIMENT_API_URL", "https://newsapi.org/v2/everything")
    primary_key = get_env("SENTIMENT_API_KEY")
    try:
        primary_url_full = f'{primary_url}?symbol={ticker}&apikey={primary_key}'
        timeout_v = HTTP_TIMEOUT
        primary_resp = requests.get(primary_url_full, timeout=timeout_v)
        if primary_resp.status_code in {429, 500, 502, 503, 504} and alt_api_key and alt_api_url:
            pytime.sleep(0.5)
            alt_url = f'{alt_api_url}?symbol={ticker}&apikey={alt_api_key}'
            alt_resp = requests.get(alt_url, timeout=timeout_v)
            if alt_resp.status_code == 200:
                data = alt_resp.json()
                sentiment_score = data.get('sentiment_score', 0.0)
                if -1.0 <= sentiment_score <= 1.0:
                    logger.info(f'ALTERNATIVE_SENTIMENT_SUCCESS | ticker={ticker} score={sentiment_score}')
                    return sentiment_score
        elif primary_resp.status_code == 200:
            data = primary_resp.json()
            sentiment_score = data.get('sentiment_score', 0.0)
            if -1.0 <= sentiment_score <= 1.0:
                return sentiment_score
    except (ValueError, TypeError) as e:
        logger.debug(f'Alternative sentiment source failed for {ticker}: {e}')
    return None

def _try_cached_similar_symbols(ticker: str) -> float | None:
    """Try to use sentiment from similar symbols (same sector/industry)."""
    symbol_correlations = {'AAPL': ['MSFT', 'GOOGL', 'AMZN', 'META'], 'MSFT': ['AAPL', 'GOOGL', 'AMZN', 'META'], 'GOOGL': ['AAPL', 'MSFT', 'AMZN', 'META'], 'META': ['AAPL', 'MSFT', 'GOOGL', 'AMZN'], 'AMZN': ['AAPL', 'MSFT', 'GOOGL', 'META'], 'NVDA': ['AMD', 'INTC', 'TSM'], 'AMD': ['NVDA', 'INTC', 'TSM'], 'JPM': ['BAC', 'WFC', 'C', 'GS'], 'BAC': ['JPM', 'WFC', 'C', 'GS'], 'SPY': ['QQQ', 'IWM', 'VTI'], 'QQQ': ['SPY', 'IWM', 'VTI'], 'IWM': ['SPY', 'QQQ', 'VTI']}
    similar_symbols = symbol_correlations.get(ticker, [])
    with sentiment_lock:
        for similar_symbol in similar_symbols:
            cached = _SENTIMENT_CACHE.get(similar_symbol)
            if cached:
                cache_ts, sentiment_val = cached
                if pytime.time() - cache_ts < SENTIMENT_TTL_SEC:
                    logger.info(f'SENTIMENT_SIMILAR_SYMBOL_PROXY | ticker={ticker} proxy={similar_symbol} score={sentiment_val}')
                    proxy_sentiment = sentiment_val * 0.8
                    return proxy_sentiment
    return None

def _try_sector_sentiment_proxy(ticker: str) -> float | None:
    """Try to derive sentiment from sector ETF or major index."""
    sector_proxies = {'XLK': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AMZN', 'CRM', 'NFLX', 'AMD'], 'XLF': ['JPM', 'BAC', 'WFC', 'C', 'GS'], 'XLY': ['AMZN', 'TSLA', 'HD', 'MCD'], 'SPY': ['*']}
    with sentiment_lock:
        for sector_etf, symbols in sector_proxies.items():
            if ticker in symbols or '*' in symbols:
                cached = _SENTIMENT_CACHE.get(sector_etf)
                if cached:
                    cache_ts, sentiment_val = cached
                    if pytime.time() - cache_ts < SENTIMENT_TTL_SEC * 2:
                        logger.info(f'SENTIMENT_SECTOR_PROXY | ticker={ticker} sector_etf={sector_etf} score={sentiment_val}')
                        sector_sentiment = sentiment_val * 0.6
                        return sector_sentiment
    return None

def _get_cached_or_neutral_sentiment(ticker: str) -> float:
    """Get cached sentiment or return neutral if no cache available."""
    with sentiment_lock:
        cached = _SENTIMENT_CACHE.get(ticker)
        if cached:
            cache_ts, sentiment_val = cached
            if pytime.time() - cache_ts < SENTIMENT_RATE_LIMITED_TTL_SEC:
                return sentiment_val
    return 0.0

def analyze_text(text: str, logger=logger) -> dict:
    """Return sentiment probabilities for ``text``.

    Falls back to neutral if transformers are unavailable.
    """
    _init_sentiment()
    deps = _load_transformers(logger)
    if deps is None:
        return {'available': False, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0}
    torch, tokenizer, model = deps
    try:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
        inputs = tensors_to_device(inputs, DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=0)
        neg, neu, pos = (float(x) for x in probs.tolist())
        return {'available': True, 'pos': pos, 'neg': neg, 'neu': neu}
    except (ValueError, TypeError) as exc:
        logger.warning('analyze_text inference failed: %s', exc)
        return {'available': False, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0}

def predict_text_sentiment(text: str) -> float:
    """Legacy float sentiment interface."""
    _init_sentiment()
    res = analyze_text(text)
    if not res.get('available'):
        return 0.0
    return float(res['pos'] - res['neg'])

def fetch_form4_filings(ticker: str) -> list[dict]:
    """
    Scrape SEC Form 4 filings for insider trade info.
    Returns a list of dicts: {"date": datetime, "type": "buy"/"sell", "dollar_amount": float}.

    Args:
        ticker: Stock ticker symbol

    Returns:
        List of insider trading filings
    """
    soup_cls = _load_bs4(logger)
    if soup_cls is None:
        logger.debug('BeautifulSoup not available, Form 4 parsing disabled')
        return []
    url = f'https://www.sec.gov/cgi-bin/own-disp?action=getowner&CIK={ticker}&type=4'
    try:
        headers = {'User-Agent': 'AI Trading Bot'}
        backoff = 0.5
        for attempt in range(3):
            r = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
            if r.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                pytime.sleep(backoff)
                backoff *= 2
                continue
            break
        r.raise_for_status()
        soup = soup_cls(r.content, 'lxml')
        filings = []
        table = soup.find('table', {'class': 'tableFile2'})
        if not table:
            return filings
        rows = table.find_all('tr')[1:]
        for row in rows:
            cols = row.find_all('td')
            if len(cols) < 6:
                continue
            date_str = cols[3].get_text(strip=True)
            try:
                datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                continue
        return filings
    except (ValueError, TypeError) as e:
        logger.debug(f'Error fetching Form 4 filings for {ticker}: {e}')
        return []
