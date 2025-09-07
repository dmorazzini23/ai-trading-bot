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

