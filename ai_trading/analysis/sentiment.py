"""
Sentiment analysis module for AI trading bot.

This module provides sentiment analysis functionality using FinBERT and NewsAPI,
extracted from bot_engine.py to enable standalone imports and testing.
"""

import time as pytime
from datetime import datetime
from threading import Lock
from typing import Dict, List, Tuple, Optional
import os

# AI-AGENT-REF: Use HTTP utilities with proper timeout/retry
from ai_trading.utils import http

# AI-AGENT-REF: Use centralized logger as per AGENTS.md
try:
    from ai_trading.logging import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logger.warning("BeautifulSoup not available, Form 4 parsing disabled")

# Retry mechanism
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, wait_random, retry_if_exception_type
    HAS_TENACITY = True
except ImportError:
    HAS_TENACITY = False
    # Fallback decorator if tenacity not available
    def retry(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    # Create mock objects that work with the + operator
    class MockWait:
        def __add__(self, other):
            return self
    
    stop_after_attempt = lambda x: None
    wait_exponential = lambda **kwargs: MockWait()
    wait_random = lambda *args, **kwargs: MockWait()
    retry_if_exception_type = lambda x: None

# AI-AGENT-REF: Import config with fallback
try:
    from ai_trading import config
    NEWS_API_KEY = getattr(config, "NEWS_API_KEY", None)
    SENTIMENT_API_KEY = getattr(config, "SENTIMENT_API_KEY", None) or NEWS_API_KEY
    SENTIMENT_API_URL = getattr(config, "SENTIMENT_API_URL", "https://newsapi.org/v2/everything")
except ImportError:
    # Fallback for testing environments
    import os
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")
    SENTIMENT_API_KEY = os.getenv("SENTIMENT_API_KEY") or NEWS_API_KEY
    SENTIMENT_API_URL = os.getenv("SENTIMENT_API_URL", "https://newsapi.org/v2/everything")

# FinBERT model initialization
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    try:
        _FINBERT_TOKENIZER = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        _FINBERT_MODEL = AutoModelForSequenceClassification.from_pretrained(
            "yiyanghkust/finbert-tone"
        )
        _FINBERT_MODEL.eval()
        _HUGGINGFACE_AVAILABLE = True
        logger.info("FinBERT loaded successfully")
    except Exception as e:
        logger.warning(f"FinBERT load failed ({e}); falling back to neutral sentiment")
        _HUGGINGFACE_AVAILABLE = False
        _FINBERT_TOKENIZER = None
        _FINBERT_MODEL = None
except ImportError:
    # Mock for testing environments without transformers/torch
    class MockFinBERT:
        def __call__(self, *args, **kwargs):
            return self
        def __getattr__(self, name):
            return lambda *args, **kwargs: self
        def tolist(self):
            return [0.33, 0.34, 0.33]  # neutral sentiment
    
    _FINBERT_TOKENIZER = MockFinBERT()
    _FINBERT_MODEL = MockFinBERT()
    _HUGGINGFACE_AVAILABLE = True
    logger.info("Using mock FinBERT for testing environment")

# Sentiment caching and circuit breaker - Enhanced for critical rate limiting fix
SENTIMENT_TTL_SEC = 600  # 10 minutes normal cache
SENTIMENT_RATE_LIMITED_TTL_SEC = 7200  # 2 hour cache when rate limited (increased)
SENTIMENT_FAILURE_THRESHOLD = 15  # Reduced back to 15 for more aggressive circuit breaker
SENTIMENT_RECOVERY_TIMEOUT = 1800  # 30 minutes recovery time (more aggressive)
SENTIMENT_MAX_RETRIES = 5  # Maximum retry attempts with exponential backoff
SENTIMENT_BASE_DELAY = 5  # Base delay in seconds for exponential backoff

_SENTIMENT_CACHE: Dict[str, Tuple[float, float]] = {}  # {ticker: (timestamp, score)}
_SENTIMENT_CIRCUIT_BREAKER = {"failures": 0, "last_failure": 0, "state": "closed"}  # closed, open, half-open
sentiment_lock = Lock()

# Expose sentiment_lock for external use (bot_engine compatibility)
__all__ = ["fetch_sentiment", "predict_text_sentiment", "sentiment_lock", "_SENTIMENT_CACHE"]


def _check_sentiment_circuit_breaker() -> bool:
    """Check if sentiment circuit breaker allows requests."""
    global _SENTIMENT_CIRCUIT_BREAKER
    now = pytime.time()
    cb = _SENTIMENT_CIRCUIT_BREAKER
    
    if cb["state"] == "open":
        if now - cb["last_failure"] > SENTIMENT_RECOVERY_TIMEOUT:
            cb["state"] = "half-open"
            logger.info("Sentiment circuit breaker moved to half-open state")
            return True
        return False
    return True


def _record_sentiment_success():
    """Record successful sentiment API call."""
    global _SENTIMENT_CIRCUIT_BREAKER
    _SENTIMENT_CIRCUIT_BREAKER["failures"] = 0
    if _SENTIMENT_CIRCUIT_BREAKER["state"] == "half-open":
        _SENTIMENT_CIRCUIT_BREAKER["state"] = "closed"
        logger.info("Sentiment circuit breaker closed - service recovered")


def _record_sentiment_failure():
    """Record failed sentiment API call and update circuit breaker."""
    global _SENTIMENT_CIRCUIT_BREAKER
    cb = _SENTIMENT_CIRCUIT_BREAKER
    cb["failures"] += 1
    cb["last_failure"] = pytime.time()
    
    if cb["failures"] >= SENTIMENT_FAILURE_THRESHOLD:
        cb["state"] = "open"
        logger.warning(f"Sentiment circuit breaker opened after {cb['failures']} failures")


@retry(
    stop=stop_after_attempt(SENTIMENT_MAX_RETRIES),  # Increased retries for rate limiting
    wait=wait_exponential(multiplier=SENTIMENT_BASE_DELAY, min=SENTIMENT_BASE_DELAY, max=180) + wait_random(0, 5),  # More aggressive backoff with jitter
    retry=retry_if_exception_type((Exception,)),  # AI-AGENT-REF: Use generic exception since we're using http utilities
)
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
    # Use new SENTIMENT_API_KEY or fallback to NEWS_API_KEY for backwards compatibility
    api_key = SENTIMENT_API_KEY or NEWS_API_KEY
    if not api_key:
        logger.debug("No sentiment API key configured (checked SENTIMENT_API_KEY and NEWS_API_KEY)")
        return 0.0

    now_ts = pytime.time()
    
    # AI-AGENT-REF: Enhanced caching with longer TTL during rate limiting
    with sentiment_lock:
        cached = _SENTIMENT_CACHE.get(ticker)
        if cached:
            last_ts, last_score = cached
            # Use longer cache during circuit breaker open state
            cache_ttl = SENTIMENT_RATE_LIMITED_TTL_SEC if _SENTIMENT_CIRCUIT_BREAKER["state"] == "open" else SENTIMENT_TTL_SEC
            if now_ts - last_ts < cache_ttl:
                logger.debug(f"Sentiment cache hit for {ticker} (age: {(now_ts - last_ts)/60:.1f}m)")
                return last_score

    # Cache miss or stale → fetch fresh
    # AI-AGENT-REF: Circuit breaker pattern for graceful degradation
    if not _check_sentiment_circuit_breaker():
        logger.info(f"Sentiment circuit breaker open, returning cached/neutral for {ticker}")
        with sentiment_lock:
            # Try to use any existing cache, even if stale
            cached = _SENTIMENT_CACHE.get(ticker)
            if cached:
                _, last_score = cached
                logger.debug(f"Using stale cached sentiment {last_score} for {ticker}")
                return last_score
            # No cache available, store and return neutral
            _SENTIMENT_CACHE[ticker] = (now_ts, 0.0)
            return 0.0

    try:
        # 1) Fetch NewsAPI articles using configurable URL
        url = (
            f"{SENTIMENT_API_URL}?"
            f"q={ticker}&sortBy=publishedAt&language=en&pageSize=5"
            f"&apiKey={api_key}"
        )
        resp = http.get(url)
        
        # AI-AGENT-REF: Enhanced rate limiting detection and handling
        if resp.status_code == 429:
            logger.warning(f"fetch_sentiment({ticker}) rate-limited (429) → using enhanced fallback strategies")
            return _handle_rate_limit_with_enhanced_strategies(ticker)
        elif resp.status_code == 403:
            logger.warning(f"fetch_sentiment({ticker}) forbidden (403) - possible API key issue → using fallback")
            _record_sentiment_failure()
            return _get_cached_or_neutral_sentiment(ticker)
        elif resp.status_code >= 500:
            logger.warning(f"fetch_sentiment({ticker}) server error ({resp.status_code}) → using fallback")
            _record_sentiment_failure()
            return _get_cached_or_neutral_sentiment(ticker)
            
        resp.raise_for_status()
        
        payload = resp.json()
        articles = payload.get("articles", [])
        scores = []
        if articles:
            for art in articles:
                text = (art.get("title") or "") + ". " + (art.get("description") or "")
                if text.strip():
                    scores.append(predict_text_sentiment(text))
        news_score = float(sum(scores) / len(scores)) if scores else 0.0

        # 2) Fetch Form 4 data (insider trades) - with error handling
        form4_score = 0.0
        try:
            form4 = fetch_form4_filings(ticker)
            # If any insider buy in last 7 days > $50k, boost sentiment
            for filing in form4:
                if filing["type"] == "buy" and filing["dollar_amount"] > 50_000:
                    form4_score += 0.1
        except Exception as e:
            logger.debug(f"Form4 fetch failed for {ticker}: {e}")  # Reduced to debug level

        final_score = 0.8 * news_score + 0.2 * form4_score
        final_score = max(-1.0, min(1.0, final_score))
        
        # AI-AGENT-REF: Record success and update cache
        _record_sentiment_success()
        with sentiment_lock:
            _SENTIMENT_CACHE[ticker] = (now_ts, final_score)
        return final_score
        
    except Exception as e:
        logger.warning(f"Sentiment API request failed for {ticker}: {e}")
        _record_sentiment_failure()
        
        # AI-AGENT-REF: Fallback to cached data or neutral if no cache
        with sentiment_lock:
            cached = _SENTIMENT_CACHE.get(ticker)
            if cached:
                _, last_score = cached
                logger.debug(f"Using cached sentiment fallback {last_score} for {ticker}")
                return last_score
            # No cache available, return neutral
            _SENTIMENT_CACHE[ticker] = (now_ts, 0.0)
            return 0.0
    except Exception as e:
        logger.error(f"Unexpected error fetching sentiment for {ticker}: {e}")
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
    
    # Try multiple fallback strategies in order of preference
    fallback_sources = [
        _try_alternative_sentiment_sources,
        _try_cached_similar_symbols,
        _try_sector_sentiment_proxy,
        _get_cached_or_neutral_sentiment
    ]
    
    for fallback_func in fallback_sources:
        try:
            result = fallback_func(ticker)
            if result is not None:
                logger.info(f"SENTIMENT_FALLBACK_SUCCESS | ticker={ticker} source={fallback_func.__name__} value={result}")
                return result
        except Exception as e:
            logger.debug(f"SENTIMENT_FALLBACK_FAILED | ticker={ticker} source={fallback_func.__name__} error={e}")
            continue
    
    # Final fallback: enhanced cached lookup with extended TTL
    with sentiment_lock:
        cached = _SENTIMENT_CACHE.get(ticker)
        if cached:
            cache_ts, sentiment_val = cached
            # Use cached data even if older than normal TTL during rate limiting
            if time.time() - cache_ts < SENTIMENT_RATE_LIMITED_TTL_SEC:
                logger.info(f"SENTIMENT_RATE_LIMIT_USING_EXTENDED_CACHE | ticker={ticker} age_hours={int((time.time() - cache_ts) / 3600)}")
                return sentiment_val
        
        # Cache neutral score with extended TTL during rate limiting
        now_ts = time.time()
        _SENTIMENT_CACHE[ticker] = (now_ts, 0.0)
    
    # Log enhanced rate limiting information for monitoring
    logger.warning("SENTIMENT_RATE_LIMIT_ALL_FALLBACKS_EXHAUSTED", extra={
        "ticker": ticker,
        "fallback_strategies_tried": len(fallback_sources),
        "cache_ttl_hours": SENTIMENT_RATE_LIMITED_TTL_SEC / 3600,
        "recommendation": "Consider upgrading NewsAPI plan, adding alternative sentiment sources, or reviewing rate limits"
    })
    
    return 0.0


def _try_alternative_sentiment_sources(ticker: str) -> Optional[float]:
    """Try alternative sentiment data sources when primary is rate limited."""
    # AI-AGENT-REF: Placeholder for future alternative sentiment sources
    # This could include sources like:
    # - Financial news aggregators (e.g., Alpha Vantage, Quandl)
    # - Social media sentiment (Twitter, Reddit)
    # - Financial blogs and analysis sites
    # - SEC filing sentiment analysis
    
    alternative_sources = []
    
    # Try environment-configured alternative sources
    alt_api_key = os.getenv("ALTERNATIVE_SENTIMENT_API_KEY")
    alt_api_url = os.getenv("ALTERNATIVE_SENTIMENT_API_URL")
    
    if alt_api_key and alt_api_url:
        try:
            # Example implementation for alternative API
            response = http.get(
                f"{alt_api_url}?symbol={ticker}&apikey={alt_api_key}"
            )
            if response.status_code == 200:
                data = response.json()
                sentiment_score = data.get('sentiment_score', 0.0)
                if -1.0 <= sentiment_score <= 1.0:
                    logger.info(f"ALTERNATIVE_SENTIMENT_SUCCESS | ticker={ticker} score={sentiment_score}")
                    return sentiment_score
        except Exception as e:
            logger.debug(f"Alternative sentiment source failed for {ticker}: {e}")
    
    return None


def _try_cached_similar_symbols(ticker: str) -> Optional[float]:
    """Try to use sentiment from similar symbols (same sector/industry)."""
    # AI-AGENT-REF: Use sentiment from related symbols as proxy
    symbol_correlations = {
        # Tech stocks
        'AAPL': ['MSFT', 'GOOGL', 'AMZN', 'META'],
        'MSFT': ['AAPL', 'GOOGL', 'AMZN', 'META'],
        'GOOGL': ['AAPL', 'MSFT', 'AMZN', 'META'],
        'META': ['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
        'AMZN': ['AAPL', 'MSFT', 'GOOGL', 'META'],
        'NVDA': ['AMD', 'INTC', 'TSM'],
        'AMD': ['NVDA', 'INTC', 'TSM'],
        
        # Financial stocks
        'JPM': ['BAC', 'WFC', 'C', 'GS'],
        'BAC': ['JPM', 'WFC', 'C', 'GS'],
        
        # ETFs
        'SPY': ['QQQ', 'IWM', 'VTI'],
        'QQQ': ['SPY', 'IWM', 'VTI'],
        'IWM': ['SPY', 'QQQ', 'VTI'],
    }
    
    similar_symbols = symbol_correlations.get(ticker, [])
    
    with sentiment_lock:
        for similar_symbol in similar_symbols:
            cached = _SENTIMENT_CACHE.get(similar_symbol)
            if cached:
                cache_ts, sentiment_val = cached
                # Use recent sentiment from similar symbols
                if time.time() - cache_ts < SENTIMENT_TTL_SEC:
                    logger.info(f"SENTIMENT_SIMILAR_SYMBOL_PROXY | ticker={ticker} proxy={similar_symbol} score={sentiment_val}")
                    # Apply slight decay factor for proxy sentiment
                    proxy_sentiment = sentiment_val * 0.8  # 20% discount for proxy
                    return proxy_sentiment
    
    return None


def _try_sector_sentiment_proxy(ticker: str) -> Optional[float]:
    """Try to derive sentiment from sector ETF or major index."""
    # AI-AGENT-REF: Use sector or market sentiment as proxy
    sector_proxies = {
        # Technology sector proxy
        'XLK': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AMZN', 'CRM', 'NFLX', 'AMD'],
        # Financial sector proxy
        'XLF': ['JPM', 'BAC', 'WFC', 'C', 'GS'],
        # Consumer discretionary
        'XLY': ['AMZN', 'TSLA', 'HD', 'MCD'],
        # Market-wide proxies
        'SPY': ['*'],  # Fallback for any stock
    }
    
    with sentiment_lock:
        for sector_etf, symbols in sector_proxies.items():
            if ticker in symbols or '*' in symbols:
                cached = _SENTIMENT_CACHE.get(sector_etf)
                if cached:
                    cache_ts, sentiment_val = cached
                    if time.time() - cache_ts < SENTIMENT_TTL_SEC * 2:  # More lenient for sector proxy
                        logger.info(f"SENTIMENT_SECTOR_PROXY | ticker={ticker} sector_etf={sector_etf} score={sentiment_val}")
                        # Apply decay factor for sector sentiment
                        sector_sentiment = sentiment_val * 0.6  # 40% discount for sector proxy
                        return sector_sentiment
    
    return None


def _get_cached_or_neutral_sentiment(ticker: str) -> float:
    """Get cached sentiment or return neutral if no cache available."""
    with sentiment_lock:
        cached = _SENTIMENT_CACHE.get(ticker)
        if cached:
            cache_ts, sentiment_val = cached
            # Use cached data if relatively recent (within extended TTL)
            if time.time() - cache_ts < SENTIMENT_RATE_LIMITED_TTL_SEC:
                return sentiment_val
    
    # Return neutral sentiment as safe fallback
    return 0.0


def predict_text_sentiment(text: str) -> float:
    """
    Uses FinBERT (if available) to assign a sentiment score ∈ [–1, +1].
    If FinBERT is unavailable, return 0.0.
    
    Args:
        text: Text to analyze for sentiment
        
    Returns:
        Sentiment score between -1.0 and 1.0
    """
    if _HUGGINGFACE_AVAILABLE and _FINBERT_MODEL and _FINBERT_TOKENIZER:
        try:
            inputs = _FINBERT_TOKENIZER(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
            )
            with torch.no_grad():
                outputs = _FINBERT_MODEL(**inputs)
                logits = outputs.logits[0]  # shape = (3,)
                probs = torch.softmax(logits, dim=0)  # [p_neg, p_neu, p_pos]

            neg, neu, pos = probs.tolist()
            return float(pos - neg)
        except Exception as e:
            logger.warning(
                f"[predict_text_sentiment] FinBERT inference failed ({e}); returning neutral"
            )
    return 0.0


def fetch_form4_filings(ticker: str) -> List[dict]:
    """
    Scrape SEC Form 4 filings for insider trade info.
    Returns a list of dicts: {"date": datetime, "type": "buy"/"sell", "dollar_amount": float}.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        List of insider trading filings
    """
    if not BS4_AVAILABLE:
        logger.debug("BeautifulSoup not available, Form 4 parsing disabled")
        return []
        
    url = f"https://www.sec.gov/cgi-bin/own-disp?action=getowner&CIK={ticker}&type=4"
    try:
        r = http.get(url, headers={"User-Agent": "AI Trading Bot"})
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "lxml")
        filings = []
        # Parse table rows (approximate)
        table = soup.find("table", {"class": "tableFile2"})
        if not table:
            return filings
        rows = table.find_all("tr")[1:]  # skip header
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 6:
                continue
            date_str = cols[3].get_text(strip=True)
            try:
                fdate = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                continue
            # Additional parsing logic would go here
            # For now, return empty list to avoid parsing errors
        return filings
    except Exception as e:
        logger.debug(f"Error fetching Form 4 filings for {ticker}: {e}")
        return []