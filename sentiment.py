"""
Sentiment analysis module for AI trading bot.

This module provides sentiment analysis functionality using FinBERT and NewsAPI,
extracted from bot_engine.py to enable standalone imports and testing.
"""

import time as pytime
import requests
from datetime import datetime
from threading import Lock
from typing import Dict, List, Tuple

# AI-AGENT-REF: Use centralized logger as per AGENTS.md
try:
    from logger import logger
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
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
except ImportError:
    # Fallback decorator if tenacity not available
    def retry(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    stop_after_attempt = lambda x: None
    wait_exponential = lambda **kwargs: None
    retry_if_exception_type = lambda x: None

# AI-AGENT-REF: Import config with fallback
try:
    import config
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

# Sentiment caching and circuit breaker
SENTIMENT_TTL_SEC = 600  # 10 minutes
SENTIMENT_RATE_LIMITED_TTL_SEC = 3600  # 1 hour cache when rate limited
SENTIMENT_FAILURE_THRESHOLD = 5  # AI-AGENT-REF: Increased from 3 to 5 for more resilience
SENTIMENT_RECOVERY_TIMEOUT = 600  # AI-AGENT-REF: Increased from 5 to 10 minutes for better recovery

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
    stop=stop_after_attempt(3),  # Allow 3 attempts total
    wait=wait_exponential(multiplier=2, min=5, max=60),  # AI-AGENT-REF: Better backoff: 5s, 10s, 20s, up to 60s max
    retry=retry_if_exception_type((requests.RequestException,)),
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
        resp = requests.get(url, timeout=10)
        
        # AI-AGENT-REF: Enhanced rate limiting detection and handling
        if resp.status_code == 429:
            logger.warning(f"fetch_sentiment({ticker}) rate-limited (429) → caching neutral with extended TTL")
            _record_sentiment_failure()
            with sentiment_lock:
                # Cache neutral score with extended TTL during rate limiting
                _SENTIMENT_CACHE[ticker] = (now_ts, 0.0)
            return 0.0
        elif resp.status_code == 403:
            logger.warning(f"fetch_sentiment({ticker}) forbidden (403) - possible API key issue → caching neutral")
            _record_sentiment_failure()
            with sentiment_lock:
                _SENTIMENT_CACHE[ticker] = (now_ts, 0.0)
            return 0.0
        elif resp.status_code >= 500:
            logger.warning(f"fetch_sentiment({ticker}) server error ({resp.status_code}) → caching neutral")
            _record_sentiment_failure()
            with sentiment_lock:
                _SENTIMENT_CACHE[ticker] = (now_ts, 0.0)
            return 0.0
            
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
        
    except requests.exceptions.RequestException as e:
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
        r = requests.get(url, headers={"User-Agent": "AI Trading Bot"}, timeout=10)
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
            except Exception:
                continue
            # Additional parsing logic would go here
            # For now, return empty list to avoid parsing errors
        return filings
    except Exception as e:
        logger.debug(f"Error fetching Form 4 filings for {ticker}: {e}")
        return []