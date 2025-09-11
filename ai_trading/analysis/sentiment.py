"""
Sentiment analysis module for AI trading bot.

This module provides sentiment analysis functionality using FinBERT and NewsAPI,
extracted from bot_engine.py to enable standalone imports and testing.
"""
import time as pytime
from datetime import datetime
from threading import Lock
from ai_trading.net.http import HTTPSession, get_http_session
from ai_trading.utils.http import clamp_request_timeout
from ai_trading.utils.retry import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)
from ai_trading.logging import logger
from ai_trading.settings import get_news_api_key
from ai_trading.config.settings import (
    get_settings,
    sentiment_backoff_base,
    sentiment_backoff_strategy,
    sentiment_retry_max,
)
from ai_trading.utils.timing import HTTP_TIMEOUT
from ai_trading.config.management import get_env, validate_required_env
from ai_trading.utils.device import get_device, tensors_to_device  # AI-AGENT-REF: guard torch import
from ai_trading.exc import RequestException, HTTPError
from ai_trading.metrics import get_counter, get_gauge
from ai_trading.data.provider_monitor import provider_monitor

SENTIMENT_API_KEY = get_env("SENTIMENT_API_KEY", "")
_http_session: HTTPSession = get_http_session()
_device = None
_sentiment_initialized = False


def _init_sentiment() -> None:
    """Initialize sentiment utilities once."""
    global _sentiment_initialized, _device
    if _sentiment_initialized:
        return
    if not get_env("PYTEST_RUNNING", "0", cast=bool):
        validate_required_env()
    _device = get_device()
    _sentiment_initialized = True
    logger.debug("SENTIMENT_INIT")
_bs4 = None
_transformers_bundle = None
_sentiment_deps_logged: set[str] = set()
_SENT_DEPS_LOGGED = False

def _load_bs4(log=logger):
    global _bs4, _SENT_DEPS_LOGGED
    if _bs4 is not None:
        return _bs4
    try:
        from bs4 import BeautifulSoup
        _bs4 = BeautifulSoup
    except (ImportError, ValueError, TypeError):
        if 'bs4' not in _sentiment_deps_logged:
            log.warning('SENTIMENT_OPTIONAL_DEP_MISSING', extra={'package': 'bs4'})
            _sentiment_deps_logged.add('bs4')
            _SENT_DEPS_LOGGED = True
        _bs4 = None
    return _bs4

def _load_transformers(log=logger):
    global _transformers_bundle, _SENT_DEPS_LOGGED
    if _transformers_bundle is not None:
        return _transformers_bundle
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        model = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
        model.to(_device)
        model.eval()
        _transformers_bundle = (torch, tokenizer, model)
    except (ImportError, ValueError, TypeError):
        if 'transformers' not in _sentiment_deps_logged:
            log.warning('SENTIMENT_OPTIONAL_DEP_MISSING', extra={'package': 'transformers'})
            _sentiment_deps_logged.add('transformers')
            _SENT_DEPS_LOGGED = True
        _transformers_bundle = None
    return _transformers_bundle
SENTIMENT_TTL_SEC = 600
SENTIMENT_RATE_LIMITED_TTL_SEC = 7200
SENTIMENT_FAILURE_THRESHOLD = 15
SENTIMENT_RECOVERY_TIMEOUT = 1800
_settings = get_settings()
SENTIMENT_MAX_RETRIES = sentiment_retry_max(_settings)
SENTIMENT_BACKOFF_BASE = sentiment_backoff_base(_settings)
SENTIMENT_BACKOFF_STRATEGY = sentiment_backoff_strategy(_settings)
if SENTIMENT_BACKOFF_STRATEGY.lower() == 'fixed':
    _SENTIMENT_WAIT = wait_random(SENTIMENT_BACKOFF_BASE, SENTIMENT_BACKOFF_BASE * 2)
else:
    _SENTIMENT_WAIT = wait_exponential(
        multiplier=SENTIMENT_BACKOFF_BASE,
        min=SENTIMENT_BACKOFF_BASE,
        max=180,
    ) + wait_random(0, SENTIMENT_BACKOFF_BASE)
SENTIMENT_NEWS_WEIGHT = 0.8
SENTIMENT_FORM4_WEIGHT = 0.2
_sentiment_cache: dict[str, tuple[float, float]] = {}
# track failures and progressive retry scheduling
_sentiment_circuit_breaker = {
    'failures': 0,
    'last_failure': 0,
    'state': 'closed',
    'next_retry': 0,
}
_METRICS_READY = False
sentiment_api_failures = None
sentiment_cb_state = None
sentiment_lock = Lock()
__all__ = [
    'fetch_sentiment',
    'analyze_text',
    'sentiment_lock',
    '_sentiment_cache',
    'SENTIMENT_FAILURE_THRESHOLD',
    'SENTIMENT_API_KEY',
    'SENTIMENT_NEWS_WEIGHT',
    'SENTIMENT_FORM4_WEIGHT',
]


def _init_metrics() -> None:
    """Register sentiment metrics once."""
    global _METRICS_READY, sentiment_api_failures, sentiment_cb_state
    if _METRICS_READY:
        return
    sentiment_api_failures = get_counter(
        "sentiment_api_failures_total",
        "Total sentiment API call failures",
    )
    sentiment_cb_state = get_gauge(
        "sentiment_circuit_breaker_state",
        "Sentiment circuit breaker state (0 closed, 1 half-open, 2 open)",
    )
    sentiment_cb_state.set(0)
    _METRICS_READY = True

def _check_sentiment_circuit_breaker() -> bool:
    """Check if sentiment circuit breaker allows requests."""
    global _sentiment_circuit_breaker
    _init_metrics()
    now = pytime.time()
    cb = _sentiment_circuit_breaker
    if cb['state'] == 'open':
        if now - cb['last_failure'] > SENTIMENT_RECOVERY_TIMEOUT:
            cb['state'] = 'half-open'
            sentiment_cb_state.set(1)
            logger.info('Sentiment circuit breaker moved to half-open state')
            return True
        sentiment_cb_state.set(2)
        return False
    if now < cb.get('next_retry', 0):
        logger.debug(
            'Sentiment retry delayed %.1fs', cb['next_retry'] - now
        )
        return False
    sentiment_cb_state.set(0)
    return True

def _record_sentiment_success() -> None:
    """Record successful sentiment API call."""
    global _sentiment_circuit_breaker
    _init_metrics()
    cb = _sentiment_circuit_breaker
    cb['failures'] = 0
    cb['next_retry'] = 0
    cb['last_failure'] = 0
    if cb['state'] != 'closed':
        cb['state'] = 'closed'
        logger.info('Sentiment circuit breaker closed - service recovered')
    sentiment_cb_state.set(0)
    provider_monitor.record_success('sentiment')

def _record_sentiment_failure(reason: str = 'error', error: str | None = None) -> None:
    """Record failed sentiment API call and update circuit breaker."""
    global _sentiment_circuit_breaker
    _init_metrics()
    sentiment_api_failures.inc()
    cb = _sentiment_circuit_breaker
    cb['failures'] += 1
    cb['last_failure'] = pytime.time()
    delay = min(
        SENTIMENT_BACKOFF_BASE * (2 ** (cb['failures'] - 1)),
        SENTIMENT_RECOVERY_TIMEOUT,
    )
    cb['next_retry'] = cb['last_failure'] + delay
    if cb['failures'] >= SENTIMENT_FAILURE_THRESHOLD:
        cb['state'] = 'open'
        logger.warning(
            f"Sentiment circuit breaker opened after {cb['failures']} failures"
        )
    else:
        logger.debug(
            'Sentiment failure %s; next retry in %.1fs', cb['failures'], delay
        )
    state_val = {'closed': 0, 'half-open': 1, 'open': 2}[cb['state']]
    sentiment_cb_state.set(state_val)
    provider_monitor.record_failure('sentiment', reason, error)

@retry(stop=stop_after_attempt(SENTIMENT_MAX_RETRIES), wait=_SENTIMENT_WAIT, retry=retry_if_exception_type((Exception,)))
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
    api_key = (
        SENTIMENT_API_KEY
        or getattr(settings, 'sentiment_api_key', None)
        or get_news_api_key()
    )
    if not api_key:
        logger.debug('No sentiment API key configured (checked settings.sentiment_api_key and news API key)')
        return 0.0
    now_ts = pytime.time()
    with sentiment_lock:
        cached = _sentiment_cache.get(ticker)
        if cached:
            last_ts, last_score = cached
            cache_ttl = (
                SENTIMENT_RATE_LIMITED_TTL_SEC
                if _sentiment_circuit_breaker['state'] == 'open'
                else SENTIMENT_TTL_SEC
            )
            if now_ts - last_ts < cache_ttl:
                logger.debug(
                    f'Sentiment cache hit for {ticker} (age: {(now_ts - last_ts) / 60:.1f}m)'
                )
                return last_score
    if not _check_sentiment_circuit_breaker():
        logger.info(f'Sentiment circuit breaker open, returning cached/neutral for {ticker}')
        with sentiment_lock:
            cached = _sentiment_cache.get(ticker)
            if cached:
                _, last_score = cached
                logger.debug(f'Using stale cached sentiment {last_score} for {ticker}')
                return last_score
            _sentiment_cache[ticker] = (now_ts, 0.0)
            return 0.0
    try:
        url = f'{settings.sentiment_api_url}?q={ticker}&sortBy=publishedAt&language=en&pageSize=5&apiKey={api_key}'
        resp = _http_session.get(url, timeout=clamp_request_timeout(HTTP_TIMEOUT))
        if resp.status_code == 429:
            logger.warning(f'fetch_sentiment({ticker}) rate-limited (429) → using enhanced fallback strategies')
            return _handle_rate_limit_with_enhanced_strategies(ticker)
        elif resp.status_code == 403:
            logger.warning(f'fetch_sentiment({ticker}) forbidden (403) - possible API key issue → using fallback')
            _record_sentiment_failure('forbidden')
            return _get_cached_or_neutral_sentiment(ticker)
        elif resp.status_code >= 500:
            logger.warning(f'fetch_sentiment({ticker}) server error ({resp.status_code}) → using fallback')
            _record_sentiment_failure('server_error', str(resp.status_code))
            return _get_cached_or_neutral_sentiment(ticker)
        resp.raise_for_status()
        payload = resp.json()
        articles = payload.get('articles', [])
        scores = []
        if articles:
            for art in articles:
                text = (art.get('title') or '') + '. ' + (art.get('description') or '')
                if text.strip():
                    res = analyze_text(text)
                    if res.get('available'):
                        scores.append(float(res['pos'] - res['neg']))
        news_score = float(sum(scores) / len(scores)) if scores else 0.0
        form4_score = 0.0
        try:
            form4 = fetch_form4_filings(ticker)
            for filing in form4:
                if filing['type'] == 'buy' and filing['dollar_amount'] > 50000:
                    form4_score += 0.1
        except (RequestException, HTTPError) as e:
            logger.debug('Form4 fetch failed for %s - network error: %s', ticker, e)
        except (KeyError, ValueError, TypeError) as e:
            logger.debug('Form4 fetch failed for %s - data parsing error: %s', ticker, e)
        except (ValueError, TypeError) as e:
            logger.debug('Form4 fetch failed for %s - unexpected error: %s', ticker, e, extra={'component': 'sentiment', 'ticker': ticker, 'error_type': 'form4_fetch'})
        final_score = (
            SENTIMENT_NEWS_WEIGHT * news_score
            + SENTIMENT_FORM4_WEIGHT * form4_score
        )
        final_score = max(-1.0, min(1.0, final_score))
        _record_sentiment_success()
        with sentiment_lock:
            _sentiment_cache[ticker] = (now_ts, final_score)
        return final_score
    except (ValueError, TypeError) as e:
        logger.warning(f'Sentiment API request failed for {ticker}: {e}')
        _record_sentiment_failure('api_error', str(e))
        with sentiment_lock:
            cached = _sentiment_cache.get(ticker)
            if cached:
                _, last_score = cached
                logger.debug(f'Using cached sentiment fallback {last_score} for {ticker}')
                return last_score
            _sentiment_cache[ticker] = (now_ts, 0.0)
            return 0.0
    except (ValueError, TypeError) as e:
        logger.error(f'Unexpected error fetching sentiment for {ticker}: {e}')
        _record_sentiment_failure('unexpected_error', str(e))
        with sentiment_lock:
            _sentiment_cache[ticker] = (now_ts, 0.0)
        return 0.0

def _handle_rate_limit_with_enhanced_strategies(ticker: str) -> float:
    """
    Enhanced rate limiting handling with multiple fallback strategies.

    AI-AGENT-REF: Implements exponential backoff and multiple fallback data sources for critical sentiment rate limiting fix.
    """
    _record_sentiment_failure('rate_limit')
    fallback_sources = [
        _try_alternative_sentiment_sources,
        _try_cached_similar_symbols,
        _try_sector_sentiment_proxy,
        _get_cached_or_neutral_sentiment,
    ]
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
        cached = _sentiment_cache.get(ticker)
        if cached:
            cache_ts, sentiment_val = cached
            if pytime.time() - cache_ts < SENTIMENT_RATE_LIMITED_TTL_SEC:
                logger.info(
                    f'SENTIMENT_RATE_LIMIT_USING_EXTENDED_CACHE | ticker={ticker} '
                    f'age_hours={int((pytime.time() - cache_ts) / 3600)}'
                )
                return sentiment_val
        now_ts = pytime.time()
        _sentiment_cache[ticker] = (now_ts, 0.0)
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
        primary_resp = _http_session.get(primary_url_full, timeout=clamp_request_timeout(timeout_v))
        if primary_resp.status_code in {429, 500, 502, 503, 504} and alt_api_key and alt_api_url:
            pytime.sleep(0.5)
            alt_url = f'{alt_api_url}?symbol={ticker}&apikey={alt_api_key}'
            alt_resp = _http_session.get(alt_url, timeout=clamp_request_timeout(timeout_v))
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
            cached = _sentiment_cache.get(similar_symbol)
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
                cached = _sentiment_cache.get(sector_etf)
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
        cached = _sentiment_cache.get(ticker)
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
        inputs = tensors_to_device(inputs, _device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=0)
        neg, neu, pos = (float(x) for x in probs.tolist())
        return {'available': True, 'pos': pos, 'neg': neg, 'neu': neu}
    except (ValueError, TypeError) as exc:
        logger.warning('analyze_text inference failed: %s', exc)
        return {'available': False, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0}

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
            r = _http_session.get(url, headers=headers, timeout=clamp_request_timeout(HTTP_TIMEOUT))
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
