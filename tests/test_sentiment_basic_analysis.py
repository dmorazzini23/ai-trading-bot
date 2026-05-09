from types import SimpleNamespace

import pytest

from ai_trading.analysis import sentiment


@pytest.fixture(autouse=True)
def _sentiment_setup(monkeypatch):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setenv("SENTIMENT_API_KEY", "test")
    monkeypatch.setattr(sentiment, "SENTIMENT_API_KEY", "test", raising=False)
    sentiment.reset_sentiment_runtime_cache()
    sentiment._sentiment_cache.clear()
    sentiment._sentiment_circuit_breaker = {
        "failures": 0,
        "last_failure": 0,
        "state": "closed",
        "next_retry": 0,
    }
    monkeypatch.setattr(sentiment, "get_device", lambda: "cpu")
    yield
    sentiment.reset_sentiment_runtime_cache()


def test_fetch_sentiment_weighting(monkeypatch):
    class DummySettings:
        sentiment_api_url = "http://example.com"

    monkeypatch.setattr(sentiment, "get_settings", lambda: DummySettings())

    def fake_analyze(_text: str):
        return {"available": True, "pos": 0.6, "neg": 0.2, "neu": 0.2}

    monkeypatch.setattr(sentiment, "analyze_text", fake_analyze)
    monkeypatch.setattr(
        sentiment,
        "fetch_form4_filings",
        lambda _t: [{"type": "buy", "dollar_amount": 60000}],
    )

    class Resp(SimpleNamespace):
        status_code: int = 200

        def json(self):
            return {"articles": [{"title": "t", "description": "d"}]}

        def raise_for_status(self):
            return None

    sentiment._set_sentiment_http_session_for_tests(
        SimpleNamespace(get=lambda *a, **kw: Resp())
    )

    score = sentiment.fetch_sentiment(None, "AAPL")
    expected = (
        sentiment.SENTIMENT_NEWS_WEIGHT * 0.4
        + sentiment.SENTIMENT_FORM4_WEIGHT * 0.1
    )
    assert abs(score - expected) < 1e-9


def test_analyze_text_neutral(monkeypatch):
    monkeypatch.setenv("AI_TRADING_SENTIMENT_FAIL_CLOSED", "0")
    monkeypatch.setattr(sentiment, "_load_transformers", lambda log=None: None)
    res = sentiment.analyze_text("anything")
    assert res == {"available": False, "pos": 0.0, "neg": 0.0, "neu": 1.0}


def test_analyze_text_fails_closed_outside_pytest(monkeypatch):
    monkeypatch.delenv("PYTEST_RUNNING", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("AI_TRADING_SENTIMENT_FAIL_CLOSED", raising=False)
    monkeypatch.setattr(sentiment, "_load_transformers", lambda log=None: None)

    with pytest.raises(RuntimeError, match="Sentiment unavailable"):
        sentiment.analyze_text("anything")


def test_neutral_caching_after_failure(monkeypatch):
    monkeypatch.setenv("AI_TRADING_SENTIMENT_FAIL_CLOSED", "0")
    class DummySettings:
        sentiment_api_url = "http://example.com"

    monkeypatch.setattr(sentiment, "get_settings", lambda: DummySettings())

    calls = {"count": 0}

    class Resp:
        status_code = 500

        def json(self):
            return {}

    def fake_get(*_a, **_k):
        calls["count"] += 1
        return Resp()

    sentiment._set_sentiment_http_session_for_tests(SimpleNamespace(get=fake_get))
    monkeypatch.setattr(sentiment.pytime, "time", lambda: 0)

    score1 = sentiment.fetch_sentiment(None, "MSFT")
    score2 = sentiment.fetch_sentiment(None, "MSFT")

    assert score1 == 0.0
    assert score2 == 0.0
    assert calls["count"] == 1


def test_fetch_sentiment_missing_api_key_fails_closed_outside_pytest(monkeypatch):
    monkeypatch.delenv("PYTEST_RUNNING", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("AI_TRADING_SENTIMENT_FAIL_CLOSED", raising=False)
    monkeypatch.delenv("SENTIMENT_API_KEY", raising=False)
    monkeypatch.setattr(sentiment, "SENTIMENT_API_KEY", "", raising=False)

    class DummySettings:
        sentiment_api_url = "http://example.com"
        sentiment_api_key = None

    monkeypatch.setattr(sentiment, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(sentiment, "get_news_api_key", lambda: "")

    with pytest.raises(RuntimeError, match="missing_api_key"):
        sentiment.fetch_sentiment(None, "AAPL")


def test_fetch_sentiment_enabled_without_key_fails_closed_in_tests(monkeypatch):
    monkeypatch.setenv("AI_TRADING_SENTIMENT_FAIL_CLOSED", "0")
    monkeypatch.setenv("AI_TRADING_SENTIMENT_ENABLED", "1")
    monkeypatch.delenv("SENTIMENT_API_KEY", raising=False)
    monkeypatch.setattr(sentiment, "SENTIMENT_API_KEY", "", raising=False)

    class DummySettings:
        sentiment_api_url = "http://example.com"
        sentiment_api_key = None

    monkeypatch.setattr(sentiment, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(sentiment, "get_news_api_key", lambda: "")

    with pytest.raises(RuntimeError, match="missing_api_key"):
        sentiment.fetch_sentiment(None, "AAPL")


def test_fetch_sentiment_disabled_without_key_returns_non_authoritative_neutral(monkeypatch):
    monkeypatch.setenv("AI_TRADING_SENTIMENT_FAIL_CLOSED", "1")
    monkeypatch.setenv("AI_TRADING_SENTIMENT_ENABLED", "0")
    monkeypatch.delenv("SENTIMENT_API_KEY", raising=False)

    assert sentiment.fetch_sentiment(None, "AAPL") == 0.0
    evidence = sentiment.get_sentiment_evidence("AAPL")
    assert evidence is not None
    assert evidence["source"] == "disabled"
    assert evidence["authoritative"] is False


def test_fetch_sentiment_fail_closed_runtime_error_is_not_retried(monkeypatch):
    calls = {"count": 0}

    class DummySettings:
        sentiment_api_url = "http://example.com"
        sentiment_api_key = None

    def fail_closed(_ticker, *, reason):
        calls["count"] += 1
        raise RuntimeError(f"Sentiment unavailable: {reason}")

    monkeypatch.setattr(sentiment, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(sentiment, "get_news_api_key", lambda: "")
    monkeypatch.delenv("SENTIMENT_API_KEY", raising=False)
    monkeypatch.setattr(sentiment, "_get_cached_or_neutral_sentiment", fail_closed)

    with pytest.raises(RuntimeError, match="missing_api_key"):
        sentiment.fetch_sentiment(None, "AAPL")

    assert calls["count"] == 0


def test_fetch_sentiment_resolves_fresh_key_over_import_constant(monkeypatch):
    class DummySettings:
        sentiment_api_url = "http://example.com"
        sentiment_api_key = "fresh-settings-key"

    captured = {}
    monkeypatch.setattr(sentiment, "SENTIMENT_API_KEY", "stale-import-key", raising=False)
    monkeypatch.setattr(sentiment, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(sentiment, "get_news_api_key", lambda: "fresh-env-key")
    monkeypatch.setattr(
        sentiment,
        "analyze_text",
        lambda _text: {"available": True, "pos": 0.5, "neg": 0.2, "neu": 0.3},
    )
    monkeypatch.setattr(sentiment, "fetch_form4_filings", lambda _ticker: [])

    class Resp:
        status_code = 200

        def json(self):
            return {"articles": [{"title": "t", "description": "d"}]}

        def raise_for_status(self):
            return None

    def fake_get(url, **_kwargs):
        captured["url"] = url
        return Resp()

    sentiment._set_sentiment_http_session_for_tests(SimpleNamespace(get=fake_get))

    sentiment.fetch_sentiment(None, "AAPL")

    assert "apiKey=fresh-settings-key" in captured["url"]
    assert "stale-import-key" not in captured["url"]


def test_circuit_breaker_recovers(monkeypatch):
    class DummySettings:
        sentiment_api_url = "http://example.com"

    monkeypatch.setattr(sentiment, "get_settings", lambda: DummySettings())

    def fake_analyze(_text: str):
        return {"available": True, "pos": 0.7, "neg": 0.1, "neu": 0.2}

    monkeypatch.setattr(sentiment, "analyze_text", fake_analyze)
    monkeypatch.setattr(sentiment, "fetch_form4_filings", lambda _t: [])

    class Resp:
        status_code = 200

        def json(self):
            return {"articles": [{"title": "t", "description": "d"}]}

        def raise_for_status(self):
            return None

    sentiment._set_sentiment_http_session_for_tests(
        SimpleNamespace(get=lambda *a, **k: Resp())
    )

    # Open circuit breaker and advance time beyond recovery timeout
    sentiment._sentiment_circuit_breaker = {
        "failures": sentiment.SENTIMENT_FAILURE_THRESHOLD,
        "last_failure": 0,
        "state": "open",
        "next_retry": 0,
    }

    monkeypatch.setattr(
        sentiment.pytime, "time", lambda: sentiment.SENTIMENT_RECOVERY_TIMEOUT + 1
    )

    score = sentiment.fetch_sentiment(None, "AAPL")

    assert score != 0.0
    cb = sentiment._sentiment_circuit_breaker
    assert cb["state"] == "closed"
    assert cb["failures"] == 0
