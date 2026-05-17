"""
Sentiment analysis module for AI trading bot.

This module provides sentiment analysis functionality using FinBERT and NewsAPI,
extracted from bot_engine.py to enable standalone imports and testing.
"""
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS
import time as pytime
import json
from datetime import UTC, datetime, timedelta
from threading import Lock
from typing import Any, Literal, TypedDict, cast

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

SENTIMENT_API_KEY = ""
_http_session: HTTPSession | None = None
_device = None
_sentiment_initialized = False


def _init_sentiment() -> None:
    """Initialize sentiment utilities once."""
    global _sentiment_initialized, _device
    if _sentiment_initialized:
        return
    if not (
        get_env("PYTEST_RUNNING", "0", cast=bool)
        or get_env("AI_TRADING_HF_SENTIMENT_BENCHMARK_MODE", False, cast=bool)
    ):
        validate_required_env()
    _device = get_device()
    _sentiment_initialized = True
    logger.debug("SENTIMENT_INIT")
_bs4 = None
_transformers_bundle = None
_SENT_DEPS_LOGGED: set[str] = set()
_SENTIMENT_STUB_LOGGED = False


def _finbert_revision_kwargs() -> dict[str, str]:
    revision = str(
        get_env(
            "AI_TRADING_FINBERT_MODEL_REVISION",
            "",
            cast=str,
            resolve_aliases=False,
        )
        or ""
    ).strip()
    return {"revision": revision} if revision else {}


def _finbert_revision_required() -> bool:
    if get_env("PYTEST_RUNNING", "0", cast=bool) or get_env(
        "AI_TRADING_HF_SENTIMENT_BENCHMARK_MODE",
        False,
        cast=bool,
    ):
        return False
    if str(
        get_env("PYTEST_CURRENT_TEST", "", cast=str, resolve_aliases=False) or ""
    ).strip():
        return False
    if not _sentiment_fail_closed():
        return False
    if bool(
        get_env(
            "AI_TRADING_FINBERT_ALLOW_UNPINNED",
            False,
            cast=bool,
            resolve_aliases=False,
        )
    ):
        return False
    return _sentiment_runtime_enabled()


class SentimentEvidence(TypedDict, total=False):
    ticker: str
    score: float
    available: bool
    source: str
    authoritative: bool
    provenance: dict[str, Any]
    reason: str


_sentiment_evidence_cache: dict[str, SentimentEvidence] = {}
_sentiment_proxy_provenance: dict[str, dict[str, Any]] = {}

_DEFAULT_FORM4_CIK_MAP = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "GOOGL": "0001652044",
    "GOOG": "0001652044",
    "AMZN": "0001018724",
    "META": "0001326801",
    "NVDA": "0001045810",
    "TSLA": "0001318605",
    "JPM": "0000019617",
    "BAC": "0000070858",
    "WFC": "0000072971",
    "C": "0000831001",
    "GS": "0000886982",
}


def _default_fail_closed_outside_tests() -> bool:
    return not bool(
        str(get_env("PYTEST_CURRENT_TEST", "", cast=str) or "").strip()
        or bool(get_env("PYTEST_RUNNING", False, cast=bool))
    )


def _sentiment_fail_closed() -> bool:
    return bool(
        get_env(
            "AI_TRADING_SENTIMENT_FAIL_CLOSED",
            _default_fail_closed_outside_tests(),
            cast=bool,
        )
    )


def _sentiment_runtime_enabled() -> bool:
    return bool(get_env("AI_TRADING_SENTIMENT_ENABLED", True, cast=bool))


def _sentiment_min_scored_articles() -> int:
    return max(
        1,
        int(
            get_env(
                "AI_TRADING_SENTIMENT_MIN_SCORED_ARTICLES",
                1,
                cast=int,
                resolve_aliases=False,
            )
            or 1
        ),
    )


def _form4_recency_days() -> int:
    return max(
        1,
        int(
            get_env(
                "AI_TRADING_FORM4_RECENCY_DAYS",
                30,
                cast=int,
                resolve_aliases=False,
            )
                or 30
        ),
    )


def _record_sentiment_evidence(
    ticker: str,
    score: float,
    *,
    source: str,
    authoritative: bool,
    reason: str,
    provenance: dict[str, Any] | None = None,
) -> None:
    evidence: SentimentEvidence = {
        "ticker": ticker,
        "score": float(max(-1.0, min(1.0, score))),
        "available": bool(authoritative),
        "source": source,
        "authoritative": bool(authoritative),
        "reason": reason,
        "provenance": dict(provenance or {}),
    }
    _sentiment_evidence_cache[ticker] = evidence


def get_sentiment_evidence(ticker: str) -> SentimentEvidence | None:
    """Return the last structured sentiment evidence recorded for ``ticker``."""
    evidence = _sentiment_evidence_cache.get(ticker)
    return cast(SentimentEvidence, dict(evidence)) if evidence is not None else None


def _raise_sentiment_unavailable(reason: str) -> None:
    raise RuntimeError(
        "Sentiment unavailable: "
        f"{reason}. Set AI_TRADING_SENTIMENT_FAIL_CLOSED=0 to allow neutral fallback."
    )


def _neutral_sentiment_payload(reason: str) -> dict[str, float | bool]:
    if _sentiment_fail_closed():
        _raise_sentiment_unavailable(reason)
    return {"available": False, "pos": 0.0, "neg": 0.0, "neu": 1.0}


def _resolve_sentiment_api_key(settings: Any) -> str:
    """Resolve sentiment API key from current settings/config each call."""
    return str(
        getattr(settings, 'sentiment_api_key', None)
        or get_news_api_key()
        or get_env("SENTIMENT_API_KEY", "", cast=str, resolve_aliases=False)
        or ""
    )


def _get_sentiment_http_session() -> HTTPSession:
    """Return the sentiment HTTP session, creating it only on first use."""
    global _http_session
    if _http_session is None:
        _http_session = get_http_session()
    return _http_session


def _set_sentiment_http_session_for_tests(session: HTTPSession | None) -> None:
    """Install or clear the cached sentiment HTTP session for focused tests."""
    global _http_session
    _http_session = session


def reset_sentiment_runtime_cache() -> None:
    """Clear lazily initialized sentiment runtime resources."""
    _set_sentiment_http_session_for_tests(None)
    _sentiment_evidence_cache.clear()
    _sentiment_proxy_provenance.clear()

def _load_bs4(log=logger):
    global _bs4, _SENT_DEPS_LOGGED
    if _bs4 is not None:
        return _bs4
    try:
        from bs4 import BeautifulSoup
        _bs4 = BeautifulSoup
    except (ImportError, ValueError, TypeError):
        if "bs4" not in _SENT_DEPS_LOGGED:
            log.warning("SENTIMENT_OPTIONAL_DEP_MISSING", extra={"package": "bs4"})
            _SENT_DEPS_LOGGED.add("bs4")
        _bs4 = None
    return _bs4

def _load_transformers(log=logger):
    global _transformers_bundle, _SENT_DEPS_LOGGED
    if _transformers_bundle is not None:
        return _transformers_bundle
    revision_kwargs = _finbert_revision_kwargs()
    if not revision_kwargs and _finbert_revision_required():
        raise RuntimeError(
            "FinBERT runtime model revision is not pinned. Set "
            "AI_TRADING_FINBERT_MODEL_REVISION to the reviewed local-cache revision "
            "or AI_TRADING_FINBERT_ALLOW_UNPINNED=1 for explicit research-only use."
        )
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        model_name = str(
            get_env(
                "AI_TRADING_FINBERT_MODEL_NAME",
                "yiyanghkust/finbert-tone",
                cast=str,
                resolve_aliases=False,
            )
            or "yiyanghkust/finbert-tone"
        )
        tokenizer_cls = cast(Any, AutoTokenizer)
        model_cls = cast(Any, AutoModelForSequenceClassification)
        try:
            tokenizer = tokenizer_cls.from_pretrained(
                model_name,
                local_files_only=True,
                **revision_kwargs,
            )
            model = model_cls.from_pretrained(
                model_name,
                local_files_only=True,
                **revision_kwargs,
            )
        except ValueError:
            from transformers import BertForSequenceClassification, BertTokenizer

            bert_tokenizer_cls = cast(Any, BertTokenizer)
            bert_model_cls = cast(Any, BertForSequenceClassification)
            tokenizer = bert_tokenizer_cls.from_pretrained(
                model_name,
                local_files_only=True,
                **revision_kwargs,
            )
            model = bert_model_cls.from_pretrained(
                model_name,
                local_files_only=True,
                **revision_kwargs,
            )
        model.to(_device)
        model.eval()
        _transformers_bundle = (torch, tokenizer, model)
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        if "transformers" not in _SENT_DEPS_LOGGED:
            log.warning(
                "SENTIMENT_OPTIONAL_DEP_MISSING", extra={"package": "transformers"}
            )
            _SENT_DEPS_LOGGED.add("transformers")
        _transformers_bundle = None
    return _transformers_bundle


def _sentiment_label_key(raw: Any) -> str | None:
    token = str(raw or "").strip().lower().replace("_", "-")
    if "positive" in token or token in {"pos", "bull", "bullish"}:
        return "pos"
    if "negative" in token or token in {"neg", "bear", "bearish"}:
        return "neg"
    if "neutral" in token or token == "neu":
        return "neu"
    return None


def _sentiment_probability_payload(probs: Any, model: Any) -> dict[str, float | bool]:
    values = [float(x) for x in probs.tolist()]
    id2label = getattr(getattr(model, "config", None), "id2label", {}) or {}
    mapped = {"pos": 0.0, "neg": 0.0, "neu": 0.0}
    used_mapping = False
    for index, value in enumerate(values):
        label = id2label.get(index)
        if label is None:
            label = id2label.get(str(index)) if isinstance(id2label, dict) else None
        key = _sentiment_label_key(label)
        if key is None:
            continue
        mapped[key] = value
        used_mapping = True
    if used_mapping:
        return {"available": True, "pos": mapped["pos"], "neg": mapped["neg"], "neu": mapped["neu"]}
    if len(values) >= 3:
        neg, neu, pos = values[:3]
        return {"available": True, "pos": pos, "neg": neg, "neu": neu}
    return _neutral_sentiment_payload("sentiment_label_mapping_unavailable")
SENTIMENT_TTL_SEC = 600
SENTIMENT_RATE_LIMITED_TTL_SEC = 7200
SENTIMENT_FAILURE_THRESHOLD = 15
SENTIMENT_RECOVERY_TIMEOUT = 1800
SENTIMENT_MAX_RETRIES = 5
SENTIMENT_BACKOFF_BASE = 5.0
SENTIMENT_BACKOFF_STRATEGY = "exponential"
SENTIMENT_NEWS_WEIGHT = 0.8
SENTIMENT_FORM4_WEIGHT = 0.2
_sentiment_cache: dict[str, tuple[float, float]] = {}


class _SentimentCircuitBreaker(TypedDict):
    failures: int
    last_failure: float
    state: str
    next_retry: float


# track failures and progressive retry scheduling
_sentiment_circuit_breaker: _SentimentCircuitBreaker = {
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
    '_sentiment_evidence_cache',
    'get_sentiment_evidence',
    'SENTIMENT_FAILURE_THRESHOLD',
    'SENTIMENT_API_KEY',
    'SENTIMENT_NEWS_WEIGHT',
    'SENTIMENT_FORM4_WEIGHT',
]


def _current_sentiment_retry_policy() -> tuple[int, float, str]:
    """Resolve sentiment retry policy from current settings at call time."""
    settings = get_settings()
    max_retries = max(1, int(sentiment_retry_max(settings)))
    backoff_base = max(0.0, float(sentiment_backoff_base(settings)))
    backoff_strategy = str(sentiment_backoff_strategy(settings) or "exponential")
    return max_retries, backoff_base, backoff_strategy


def _sentiment_retry_wait(backoff_base: float, backoff_strategy: str):
    if backoff_strategy.lower() == 'fixed':
        return wait_random(backoff_base, backoff_base * 2)
    return wait_exponential(
        multiplier=backoff_base,
        min=backoff_base,
        max=180,
    ) + wait_random(0, backoff_base)


def _sentiment_retry_decorator():
    max_retries, backoff_base, backoff_strategy = _current_sentiment_retry_policy()
    return retry(
        stop=stop_after_attempt(max_retries),
        wait=_sentiment_retry_wait(backoff_base, backoff_strategy),
        retry=retry_if_exception_type((RequestException, HTTPError, ValueError, TypeError)),
        reraise=True,
    )


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
    _, backoff_base, _ = _current_sentiment_retry_policy()
    delay = min(backoff_base * (2 ** (cb['failures'] - 1)), SENTIMENT_RECOVERY_TIMEOUT)
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

def fetch_sentiment(ctx, ticker: str) -> float:
    """Fetch sentiment with the current retry policy resolved at call time."""
    return cast(float, _sentiment_retry_decorator()(_fetch_sentiment_once)(ctx, ticker))


def _fetch_sentiment_once(ctx, ticker: str) -> float:
    """
    Fetch sentiment via NewsAPI + FinBERT + Form 4 signal.
    Uses a simple in-memory TTL cache to avoid hitting NewsAPI too often.
    If sentiment is unavailable, cached sentiment may still be used. Neutral
    fallback is allowed only when explicit degraded mode is enabled.

    Args:
        ctx: BotContext (for compatibility with bot_engine)
        ticker: Stock ticker symbol

    Returns:
        Sentiment score between -1.0 and 1.0
    """
    _init_sentiment()
    if not _sentiment_runtime_enabled():
        with sentiment_lock:
            cached = _sentiment_cache.get(ticker)
        score = float(cached[1]) if cached else 0.0
        _record_sentiment_evidence(
            ticker,
            score,
            source="disabled",
            authoritative=False,
            reason="sentiment_disabled",
        )
        return score
    settings = get_settings()
    api_key = _resolve_sentiment_api_key(settings)
    if not api_key:
        logger.debug('No sentiment API key configured (checked settings.sentiment_api_key and news API key)')
        _raise_sentiment_unavailable("missing_api_key")
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
                _record_sentiment_evidence(
                    ticker,
                    last_score,
                    source="cache",
                    authoritative=False,
                    reason="cache_hit",
                    provenance={"cache_age_seconds": now_ts - last_ts},
                )
                return last_score
    if not _check_sentiment_circuit_breaker():
        logger.info(f'Sentiment circuit breaker open, returning cached/neutral for {ticker}')
        fallback_score = _get_cached_or_neutral_sentiment(
            ticker,
            reason="circuit_breaker_open_without_cache",
        )
        with sentiment_lock:
            cached = _sentiment_cache.get(ticker)
            if cached:
                _, last_score = cached
                logger.debug(f'Using stale cached sentiment {last_score} for {ticker}')
                _record_sentiment_evidence(
                    ticker,
                    last_score,
                    source="cache",
                    authoritative=False,
                    reason="circuit_breaker_stale_cache",
                    provenance={"circuit_breaker_state": _sentiment_circuit_breaker["state"]},
                )
                return last_score
            _sentiment_cache[ticker] = (now_ts, fallback_score)
        _record_sentiment_evidence(
            ticker,
            fallback_score,
            source="neutral_fallback",
            authoritative=False,
            reason="circuit_breaker_open_without_cache",
        )
        return fallback_score
    try:
        url = f'{settings.sentiment_api_url}?q={ticker}&sortBy=publishedAt&language=en&pageSize=5&apiKey={api_key}'
        resp = _get_sentiment_http_session().get(url, timeout=clamp_request_timeout(HTTP_TIMEOUT))
        if resp.status_code == 429:
            logger.warning(f'fetch_sentiment({ticker}) rate-limited (429) → using enhanced fallback strategies')
            return _handle_rate_limit_with_enhanced_strategies(ticker)
        elif resp.status_code == 403:
            logger.warning(f'fetch_sentiment({ticker}) forbidden (403) - possible API key issue → using fallback')
            _record_sentiment_failure('forbidden')
            return _get_cached_or_neutral_sentiment(ticker, reason="http_403")
        elif resp.status_code >= 500:
            logger.warning(f'fetch_sentiment({ticker}) server error ({resp.status_code}) → using fallback')
            _record_sentiment_failure('server_error', str(resp.status_code))
            return _get_cached_or_neutral_sentiment(
                ticker,
                reason=f"http_{resp.status_code}",
            )
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, dict):
            raise TypeError("sentiment API payload must be a JSON object")
        articles = payload.get('articles', [])
        if not isinstance(articles, list):
            raise TypeError("sentiment API articles must be a list")
        scores = []
        if articles:
            for art in articles:
                if not isinstance(art, dict):
                    raise TypeError("sentiment API article must be a JSON object")
                title = art.get('title') or ''
                description = art.get('description') or ''
                if not isinstance(title, str) or not isinstance(description, str):
                    raise TypeError("sentiment API article text fields must be strings")
                if title.strip() or description.strip():
                    text = title + '. ' + description
                    res = analyze_text(text)
                    if res.get('available'):
                        scores.append(float(res['pos'] - res['neg']))
        min_scored_articles = _sentiment_min_scored_articles()
        if len(scores) < min_scored_articles:
            reason = "insufficient_newsapi_evidence"
            logger.info(
                "SENTIMENT_INSUFFICIENT_NEWSAPI_EVIDENCE",
                extra={
                    "ticker": ticker,
                    "scored_article_count": len(scores),
                    "min_scored_articles": min_scored_articles,
                    "article_count": len(articles),
                },
            )
            return _get_cached_or_neutral_sentiment(ticker, reason=reason)
        news_score = float(sum(scores) / len(scores))
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
        _record_sentiment_evidence(
            ticker,
            final_score,
            source="newsapi_finbert_form4",
            authoritative=True,
            reason="provider_success",
            provenance={
                "article_count": len(articles),
                "scored_article_count": len(scores),
                "min_scored_article_count": min_scored_articles,
                "news_score": news_score,
                "form4_score": form4_score,
                "form4_count": len(form4) if "form4" in locals() else 0,
                "news_weight": SENTIMENT_NEWS_WEIGHT,
                "form4_weight": SENTIMENT_FORM4_WEIGHT,
                "text_model": "finbert_local_files_only",
            },
        )
        return final_score
    except (RequestException, ValueError, TypeError) as e:
        logger.warning(f'Sentiment API request failed for {ticker}: {e}')
        _record_sentiment_failure('api_error', str(e))
        with sentiment_lock:
            cached = _sentiment_cache.get(ticker)
            if cached:
                _, last_score = cached
                logger.debug(f'Using cached sentiment fallback {last_score} for {ticker}')
                _record_sentiment_evidence(
                    ticker,
                    last_score,
                    source="cache",
                    authoritative=False,
                    reason="api_error_cached_fallback",
                    provenance={"error": str(e)},
                )
                return last_score
        fallback_score = _get_cached_or_neutral_sentiment(
            ticker,
            reason="api_error_without_cache",
        )
        with sentiment_lock:
            _sentiment_cache[ticker] = (now_ts, fallback_score)
        _record_sentiment_evidence(
            ticker,
            fallback_score,
            source="neutral_fallback",
            authoritative=False,
            reason="api_error_without_cache",
            provenance={"error": str(e)},
        )
        return fallback_score

def _handle_rate_limit_with_enhanced_strategies(ticker: str) -> float:
    """
    Enhanced rate limiting handling with multiple fallback strategies.

    AI-AGENT-REF: Implements exponential backoff and multiple fallback data sources for critical sentiment rate limiting fix.
    """
    _record_sentiment_failure('rate_limit')
    _sentiment_proxy_provenance.pop(ticker, None)
    fallback_sources = [
        (
            _try_alternative_sentiment_sources,
            lambda: _try_alternative_sentiment_sources(ticker),
        ),
        (
            _try_cached_similar_symbols,
            lambda: _try_cached_similar_symbols(ticker),
        ),
        (
            _try_sector_sentiment_proxy,
            lambda: _try_sector_sentiment_proxy(ticker),
        ),
    ]
    for fallback_func, fallback_call in fallback_sources:
        try:
            result = fallback_call()
            if result is not None:
                logger.info(f'SENTIMENT_FALLBACK_SUCCESS | ticker={ticker} source={fallback_func.__name__} value={result}')
                source_name: Literal[
                    "alternative_provider",
                    "similar_symbol_proxy",
                    "sector_proxy",
                    "neutral_fallback",
                ]
                if fallback_func is _try_alternative_sentiment_sources:
                    source_name = "alternative_provider"
                elif fallback_func is _try_cached_similar_symbols:
                    source_name = "similar_symbol_proxy"
                elif fallback_func is _try_sector_sentiment_proxy:
                    source_name = "sector_proxy"
                else:
                    source_name = "neutral_fallback"
                value = float(result)
                _record_sentiment_evidence(
                    ticker,
                    value,
                    source=source_name,
                    authoritative=False,
                    reason="rate_limit_fallback",
                    provenance={
                        "fallback": fallback_func.__name__,
                        **_sentiment_proxy_provenance.get(ticker, {}),
                    },
                )
                return value
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
                _record_sentiment_evidence(
                    ticker,
                    sentiment_val,
                    source="cache",
                    authoritative=False,
                    reason="rate_limit_extended_cache",
                    provenance={"cache_age_seconds": pytime.time() - cache_ts},
                )
                return sentiment_val
    logger.warning('SENTIMENT_RATE_LIMIT_ALL_FALLBACKS_EXHAUSTED', extra={'ticker': ticker, 'fallback_strategies_tried': len(fallback_sources), 'cache_ttl_hours': SENTIMENT_RATE_LIMITED_TTL_SEC / 3600, 'recommendation': 'Consider upgrading NewsAPI plan, adding alternative sentiment sources, or reviewing rate limits'})
    return _get_cached_or_neutral_sentiment(
        ticker,
        reason="rate_limit_without_cache",
    )

def _try_alternative_sentiment_sources(ticker: str) -> float | None:
    """Try alternative sentiment data sources when primary is rate limited."""
    alt_api_key = get_env(
        "ALTERNATIVE_SENTIMENT_API_KEY",
        None,
        cast=str,
        resolve_aliases=False,
    )
    alt_api_url = get_env(
        "ALTERNATIVE_SENTIMENT_API_URL",
        None,
        cast=str,
        resolve_aliases=False,
    )
    if not alt_api_key or not alt_api_url:
        return None
    try:
        timeout_v = HTTP_TIMEOUT
        session = _get_sentiment_http_session()
        alt_url = f'{alt_api_url}?symbol={ticker}&apikey={alt_api_key}'
        alt_resp = session.get(alt_url, timeout=clamp_request_timeout(timeout_v))
        if alt_resp.status_code == 200:
            data = alt_resp.json()
            if not isinstance(data, dict):
                return None
            sentiment_score = float(data.get('sentiment_score', 0.0) or 0.0)
            if -1.0 <= sentiment_score <= 1.0:
                logger.info(f'ALTERNATIVE_SENTIMENT_SUCCESS | ticker={ticker} score={sentiment_score}')
                _sentiment_proxy_provenance[ticker] = {
                    "provider": "alternative_sentiment",
                    "alternative_status_code": alt_resp.status_code,
                }
                return sentiment_score
    except (RequestException, HTTPError, ValueError, TypeError) as e:
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
                    _sentiment_proxy_provenance[ticker] = {
                        "proxy_symbol": similar_symbol,
                        "proxy_multiplier": 0.8,
                        "proxy_cache_age_seconds": pytime.time() - cache_ts,
                    }
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
                        _sentiment_proxy_provenance[ticker] = {
                            "proxy_symbol": sector_etf,
                            "proxy_type": "sector_etf",
                            "proxy_multiplier": 0.6,
                            "proxy_cache_age_seconds": pytime.time() - cache_ts,
                        }
                        return sector_sentiment
    return None

def _get_cached_or_neutral_sentiment(ticker: str, *, reason: str) -> float:
    """Get cached sentiment or fail closed / return neutral if no cache exists."""
    with sentiment_lock:
        cached = _sentiment_cache.get(ticker)
        if cached:
            cache_ts, sentiment_val = cached
            if pytime.time() - cache_ts < SENTIMENT_RATE_LIMITED_TTL_SEC:
                return sentiment_val
    if _sentiment_fail_closed():
        _raise_sentiment_unavailable(reason)
    _record_sentiment_evidence(
        ticker,
        0.0,
        source="neutral_fallback",
        authoritative=False,
        reason=reason,
    )
    return 0.0

def analyze_text(text: str, logger=logger) -> dict:
    """Return sentiment probabilities for ``text``.

    Fails closed by default outside tests if transformers/model weights are
    unavailable. Raises a ``RuntimeError`` with offline instructions if an SSL
    handshake fails while fetching model weights.
    """
    _init_sentiment()
    global _SENTIMENT_STUB_LOGGED
    try:
        deps = _load_transformers(logger)
    except HTTPError as exc:
        raise RuntimeError(
            "SSL error loading FinBERT; download the model and set "
            "TRANSFORMERS_OFFLINE=1 to run without internet"
        ) from exc
    except (RequestException, OSError) as exc:
        if not _SENTIMENT_STUB_LOGGED:
            logger.warning('SENTIMENT_FALLBACK_STUB', extra={'error': type(exc).__name__})
            _SENTIMENT_STUB_LOGGED = True
        return _neutral_sentiment_payload(type(exc).__name__)
    if deps is None:
        if not _SENTIMENT_STUB_LOGGED:
            logger.warning('SENTIMENT_FALLBACK_STUB', extra={'error': 'missing_dependencies'})
            _SENTIMENT_STUB_LOGGED = True
        return _neutral_sentiment_payload("missing_dependencies")
    torch, tokenizer, model = deps
    try:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
        inputs = tensors_to_device(inputs, _device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=0)
        return _sentiment_probability_payload(probs, model)
    except (ValueError, TypeError) as exc:
        logger.warning('analyze_text inference failed: %s', exc)
        if not _SENTIMENT_STUB_LOGGED:
            logger.warning('SENTIMENT_FALLBACK_STUB', extra={'error': type(exc).__name__})
            _SENTIMENT_STUB_LOGGED = True
        return _neutral_sentiment_payload(type(exc).__name__)


def _parse_form4_number(raw: Any) -> float | None:
    text = str(raw or "").strip()
    if not text or "%" in text:
        return None
    negative = text.startswith("(") and text.endswith(")")
    cleaned = (
        text.replace("$", "")
        .replace(",", "")
        .replace("+", "")
        .replace("(", "")
        .replace(")", "")
        .strip()
    )
    try:
        value = float(cleaned)
    except (TypeError, ValueError):
        return None
    if negative:
        value = -value
    return value


def _form4_transaction_type(cells: list[str]) -> str | None:
    for cell in cells:
        token = cell.strip().upper()
        if token in {"P", "BUY", "PURCHASE", "PURCHASED", "ACQUIRED", "A"}:
            return "buy"
        if token in {"S", "SELL", "SALE", "SOLD", "DISPOSED", "D"}:
            return "sell"
    return None


def _form4_dollar_amount(cells: list[str], *, date_index: int) -> float | None:
    dollar_values: list[float] = []
    plain_values: list[float] = []
    for index, cell in enumerate(cells):
        if index == date_index:
            continue
        value = _parse_form4_number(cell)
        if value is None or value <= 0:
            continue
        if "$" in cell:
            dollar_values.append(value)
        else:
            plain_values.append(value)
    if dollar_values and plain_values:
        return float(max(plain_values) * max(dollar_values))
    if dollar_values:
        return float(max(dollar_values))
    if len(plain_values) >= 2:
        return float(plain_values[-2] * plain_values[-1])
    if plain_values:
        return float(plain_values[0])
    return None


def _parse_form4_cells(cells: list[str]) -> dict[str, Any] | None:
    parsed_date: datetime | None = None
    date_index = -1
    for index, cell in enumerate(cells):
        try:
            parsed_date = datetime.strptime(cell.strip(), "%Y-%m-%d")
        except ValueError:
            continue
        date_index = index
        break
    if parsed_date is None:
        return None
    transaction_type = _form4_transaction_type(cells)
    if transaction_type is None:
        return None
    dollar_amount = _form4_dollar_amount(cells, date_index=date_index)
    if dollar_amount is None:
        return None
    return {
        "date": parsed_date,
        "type": transaction_type,
        "dollar_amount": float(dollar_amount),
    }


def _resolve_form4_cik(ticker: str) -> tuple[str | None, dict[str, Any]]:
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return None, {"reason": "empty_ticker"}
    if symbol.isdigit():
        return symbol.zfill(10), {"cik_source": "numeric_input"}
    env_map_raw = str(
        get_env(
            "AI_TRADING_FORM4_CIK_MAP",
            "",
            cast=str,
            resolve_aliases=False,
        )
        or ""
    ).strip()
    if env_map_raw:
        try:
            env_map = json.loads(env_map_raw)
        except json.JSONDecodeError:
            logger.warning("FORM4_CIK_MAP_INVALID_JSON", extra={"ticker": symbol})
            env_map = {}
        if isinstance(env_map, dict):
            cik = env_map.get(symbol) or env_map.get(symbol.replace(".", "-"))
            if cik:
                return str(cik).strip().zfill(10), {"cik_source": "env_map"}
    cik = _DEFAULT_FORM4_CIK_MAP.get(symbol)
    if cik:
        return cik, {"cik_source": "built_in_map"}
    return None, {"reason": "cik_unresolved"}


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
    cik, cik_provenance = _resolve_form4_cik(ticker)
    if cik is None:
        logger.debug("FORM4_CIK_UNRESOLVED", extra={"ticker": ticker, **cik_provenance})
        return []
    url = f'https://www.sec.gov/cgi-bin/own-disp?action=getowner&CIK={cik}&type=4'
    try:
        headers = {'User-Agent': 'AI Trading Bot'}
        cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=_form4_recency_days())
        backoff = 0.5
        for attempt in range(3):
            r = _get_sentiment_http_session().get(url, headers=headers, timeout=clamp_request_timeout(HTTP_TIMEOUT))
            if r.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                pytime.sleep(backoff)
                backoff *= 2
                continue
            break
        r.raise_for_status()
        soup = soup_cls(r.content, 'lxml')
        filings: list[dict[str, Any]] = []
        table = soup.find('table', {'class': 'tableFile2'})
        if not table:
            return filings
        rows = table.find_all('tr')[1:]
        for row in rows:
            cols = row.find_all('td')
            if len(cols) < 6:
                continue
            cells = [col.get_text(strip=True) for col in cols]
            filing = _parse_form4_cells(cells)
            if filing is not None:
                if filing["date"] < cutoff:
                    continue
                filing["ticker"] = ticker
                filing["cik"] = cik
                filing["provenance"] = {
                    "source": "sec_own_disp_form4",
                    "url": url,
                    **cik_provenance,
                }
                filings.append(filing)
        return filings
    except (ValueError, TypeError) as e:
        logger.debug(f'Error fetching Form 4 filings for {ticker}: {e}')
        return []
