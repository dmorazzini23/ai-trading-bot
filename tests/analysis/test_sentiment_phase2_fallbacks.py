from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ai_trading.analysis import sentiment


@pytest.fixture(autouse=True)
def _reset_sentiment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setenv("AI_TRADING_SENTIMENT_FAIL_CLOSED", "0")
    monkeypatch.setattr(sentiment, "_sentiment_initialized", True)
    monkeypatch.setattr(sentiment, "_device", "cpu")
    sentiment._sentiment_cache.clear()
    sentiment._SENT_DEPS_LOGGED.clear()
    sentiment._SENTIMENT_STUB_LOGGED = False
    sentiment._sentiment_circuit_breaker = {
        "failures": 0,
        "last_failure": 0.0,
        "state": "closed",
        "next_retry": 0.0,
    }

    class Counter:
        def __init__(self) -> None:
            self.count = 0

        def inc(self) -> None:
            self.count += 1

    class Gauge:
        def __init__(self) -> None:
            self.values: list[int] = []

        def set(self, value: int) -> None:
            self.values.append(value)

    counter = Counter()
    gauge = Gauge()
    monkeypatch.setattr(sentiment, "sentiment_api_failures", counter)
    monkeypatch.setattr(sentiment, "sentiment_cb_state", gauge)
    monkeypatch.setattr(sentiment, "_METRICS_READY", True)
    monkeypatch.setattr(
        sentiment.provider_monitor,
        "record_failure",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(sentiment.provider_monitor, "record_success", lambda *_args: None)


def test_circuit_breaker_blocks_delayed_retry_and_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    sentiment._sentiment_circuit_breaker.update(
        {"state": "closed", "failures": 1, "next_retry": 50.0}
    )
    monkeypatch.setattr(sentiment.pytime, "time", lambda: 10.0)

    assert sentiment._check_sentiment_circuit_breaker() is False

    sentiment._sentiment_circuit_breaker.update(
        {"state": "open", "last_failure": 0.0, "next_retry": 0.0}
    )
    monkeypatch.setattr(
        sentiment.pytime,
        "time",
        lambda: sentiment.SENTIMENT_RECOVERY_TIMEOUT + 1.0,
    )

    assert sentiment._check_sentiment_circuit_breaker() is True
    assert sentiment._sentiment_circuit_breaker["state"] == "half-open"


def test_record_failure_opens_circuit_and_success_resets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sentiment.pytime, "time", lambda: 100.0)

    for _ in range(sentiment.SENTIMENT_FAILURE_THRESHOLD):
        sentiment._record_sentiment_failure("rate_limit", "429")

    assert sentiment._sentiment_circuit_breaker["state"] == "open"
    assert sentiment._sentiment_circuit_breaker["next_retry"] > 100.0

    sentiment._record_sentiment_success()

    assert sentiment._sentiment_circuit_breaker == {
        "failures": 0,
        "last_failure": 0,
        "state": "closed",
        "next_retry": 0,
    }


def test_rate_limit_fallback_uses_similar_symbol_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sentiment.pytime, "time", lambda: 100.0)
    sentiment._sentiment_cache["MSFT"] = (99.0, 0.5)

    score = sentiment._handle_rate_limit_with_enhanced_strategies("AAPL")

    assert score == pytest.approx(0.4)


def test_rate_limit_fallback_uses_sector_proxy_and_final_neutral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sentiment.pytime, "time", lambda: 200.0)
    sentiment._sentiment_cache["XLK"] = (190.0, 0.6)

    assert sentiment._handle_rate_limit_with_enhanced_strategies("NVDA") == pytest.approx(0.36)

    sentiment._sentiment_cache.clear()
    monkeypatch.setattr(sentiment, "_try_alternative_sentiment_sources", lambda _ticker: None)
    monkeypatch.setattr(sentiment, "_try_cached_similar_symbols", lambda _ticker: None)
    monkeypatch.setattr(sentiment, "_try_sector_sentiment_proxy", lambda _ticker: None)

    assert sentiment._handle_rate_limit_with_enhanced_strategies("UNKNOWN") == 0.0
    assert sentiment._sentiment_cache["UNKNOWN"] == (200.0, 0.0)


def test_alternative_sentiment_sources_primary_and_alt(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        SimpleNamespace(status_code=429, json=lambda: {}),
        SimpleNamespace(status_code=200, json=lambda: {"sentiment_score": "0.7"}),
    ]
    urls: list[str] = []

    def fake_get(url: str, **_kwargs: Any) -> Any:
        urls.append(url)
        return responses.pop(0)

    monkeypatch.setenv("ALTERNATIVE_SENTIMENT_API_KEY", "alt-key")
    monkeypatch.setenv("ALTERNATIVE_SENTIMENT_API_URL", "https://alt.example/sent")
    monkeypatch.setenv("SENTIMENT_API_URL", "https://primary.example/sent")
    monkeypatch.setenv("SENTIMENT_API_KEY", "primary-key")
    monkeypatch.setattr(sentiment._http_session, "get", fake_get)
    monkeypatch.setattr(sentiment.pytime, "sleep", lambda _seconds: None)

    assert sentiment._try_alternative_sentiment_sources("SPY") == 0.7
    assert urls == [
        "https://primary.example/sent?symbol=SPY&apikey=primary-key",
        "https://alt.example/sent?symbol=SPY&apikey=alt-key",
    ]

    monkeypatch.setattr(
        sentiment._http_session,
        "get",
        lambda *_args, **_kwargs: SimpleNamespace(
            status_code=200,
            json=lambda: {"sentiment_score": -0.25},
        ),
    )
    assert sentiment._try_alternative_sentiment_sources("SPY") == -0.25


def test_analyze_text_success_with_fake_transformer_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    class Torch:
        @staticmethod
        def no_grad() -> Any:
            class Context:
                def __enter__(self) -> None:
                    return None

                def __exit__(self, *_args: object) -> None:
                    return None

            return Context()

        @staticmethod
        def softmax(_logits: Any, dim: int) -> Any:
            assert dim == 0
            return SimpleNamespace(tolist=lambda: [0.2, 0.3, 0.5])

    class Model:
        def __call__(self, **_inputs: Any) -> Any:
            return SimpleNamespace(logits=[[1.0, 2.0, 3.0]])

    monkeypatch.setattr(
        sentiment,
        "_load_transformers",
        lambda _logger=None: (Torch, lambda *_args, **_kwargs: {"ids": [1]}, Model()),
    )
    monkeypatch.setattr(sentiment, "tensors_to_device", lambda inputs, _device: inputs)

    assert sentiment.analyze_text("earnings look strong") == {
        "available": True,
        "pos": 0.5,
        "neg": 0.2,
        "neu": 0.3,
    }


def test_analyze_text_ssl_error_becomes_actionable_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sentiment,
        "_load_transformers",
        lambda _logger=None: (_ for _ in ()).throw(sentiment.HTTPError("ssl")),
    )

    with pytest.raises(RuntimeError, match="TRANSFORMERS_OFFLINE"):
        sentiment.analyze_text("text")


def test_fetch_form4_filings_retries_and_parses_table(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code
            self.content = b"<html></html>"

        def raise_for_status(self) -> None:
            return None

    class Column:
        def __init__(self, text: str) -> None:
            self.text = text

        def get_text(self, *, strip: bool = False) -> str:
            return self.text.strip() if strip else self.text

    class Row:
        def __init__(self, date_text: str) -> None:
            self.date_text = date_text

        def find_all(self, _tag: str) -> list[Column]:
            return [Column("") for _ in range(3)] + [Column(self.date_text)] + [
                Column("")
                for _ in range(3)
            ]

    class Table:
        def find_all(self, _tag: str) -> list[Row]:
            return [Row("header"), Row("2026-04-20"), Row("bad-date")]

    class Soup:
        def __init__(self, content: bytes, parser: str) -> None:
            assert content == b"<html></html>"
            assert parser == "lxml"

        def find(self, _tag: str, _attrs: dict[str, str]) -> Table:
            return Table()

    responses = [Response(429), Response(200)]
    monkeypatch.setattr(sentiment, "_load_bs4", lambda _logger=None: Soup)
    monkeypatch.setattr(sentiment._http_session, "get", lambda *_args, **_kwargs: responses.pop(0))
    monkeypatch.setattr(sentiment.pytime, "sleep", lambda _seconds: None)

    assert sentiment.fetch_form4_filings("SPY") == []
    assert responses == []
