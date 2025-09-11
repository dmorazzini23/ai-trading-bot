from types import SimpleNamespace

import pytest

from ai_trading.analysis import sentiment


@pytest.fixture(autouse=True)
def _sentiment_setup(monkeypatch):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setattr(sentiment, "SENTIMENT_API_KEY", "test", raising=False)
    sentiment._sentiment_cache.clear()
    sentiment._sentiment_circuit_breaker = {
        "failures": 0,
        "last_failure": 0,
        "state": "closed",
        "next_retry": 0,
    }
    monkeypatch.setattr(sentiment, "get_device", lambda: "cpu")
    yield


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

    monkeypatch.setattr(sentiment._http_session, "get", lambda *a, **kw: Resp())

    score = sentiment.fetch_sentiment(None, "AAPL")
    expected = (
        sentiment.SENTIMENT_NEWS_WEIGHT * 0.4
        + sentiment.SENTIMENT_FORM4_WEIGHT * 0.1
    )
    assert abs(score - expected) < 1e-9


def test_analyze_text_neutral(monkeypatch):
    monkeypatch.setattr(sentiment, "_load_transformers", lambda log=None: None)
    res = sentiment.analyze_text("anything")
    assert res == {"available": False, "pos": 0.0, "neg": 0.0, "neu": 1.0}


def test_neutral_caching_after_failure(monkeypatch):
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

    monkeypatch.setattr(sentiment._http_session, "get", fake_get)
    monkeypatch.setattr(sentiment.pytime, "time", lambda: 0)

    score1 = sentiment.fetch_sentiment(None, "MSFT")
    score2 = sentiment.fetch_sentiment(None, "MSFT")

    assert score1 == 0.0
    assert score2 == 0.0
    assert calls["count"] == 1


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

    monkeypatch.setattr(sentiment._http_session, "get", lambda *a, **k: Resp())

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

