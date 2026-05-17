from __future__ import annotations

import builtins
from types import SimpleNamespace
from typing import Any

import pytest

from ai_trading.analysis import sentiment


class _Response:
    def __init__(self, status_code: int = 200, payload: Any | None = None) -> None:
        self.status_code = status_code
        self._payload = payload if payload is not None else {"articles": []}

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise sentiment.HTTPError(str(self.status_code))


@pytest.fixture(autouse=True)
def _sentiment_test_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setenv("AI_TRADING_SENTIMENT_FAIL_CLOSED", "0")
    monkeypatch.setattr(sentiment, "SENTIMENT_API_KEY", "module-key")
    monkeypatch.setattr(sentiment, "_sentiment_initialized", True)
    monkeypatch.setattr(sentiment, "_device", "cpu")
    monkeypatch.setattr(
        sentiment,
        "get_settings",
        lambda: SimpleNamespace(
            sentiment_api_key="settings-key",
            sentiment_api_url="https://news.example/everything",
        ),
    )
    monkeypatch.setattr(sentiment.provider_monitor, "record_success", lambda *_args: None)
    monkeypatch.setattr(sentiment.provider_monitor, "record_failure", lambda *_args: None)
    monkeypatch.setattr(sentiment, "_record_sentiment_success", lambda: None)
    sentiment._sentiment_cache.clear()
    sentiment._sentiment_circuit_breaker = {
        "failures": 0,
        "last_failure": 0.0,
        "state": "closed",
        "next_retry": 0.0,
    }
    sentiment._SENT_DEPS_LOGGED.clear()
    sentiment._SENTIMENT_STUB_LOGGED = False
    sentiment._bs4 = None
    sentiment._transformers_bundle = None
    sentiment.reset_sentiment_runtime_cache()
    yield
    sentiment.reset_sentiment_runtime_cache()


def test_fetch_sentiment_aggregates_news_and_form4(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "articles": [
            {"title": "strong beat", "description": "guidance raised"},
            {"title": "margin pressure", "description": "costs rose"},
            {"title": "", "description": ""},
        ]
    }
    analyzed: list[str] = []

    def fake_analyze(text: str) -> dict[str, float | bool]:
        analyzed.append(text)
        if "strong" in text:
            return {"available": True, "pos": 0.8, "neg": 0.2, "neu": 0.0}
        return {"available": True, "pos": 0.3, "neg": 0.5, "neu": 0.2}

    sentiment._set_sentiment_http_session_for_tests(
        SimpleNamespace(get=lambda *_a, **_k: _Response(payload=payload))
    )
    monkeypatch.setattr(sentiment, "analyze_text", fake_analyze)
    monkeypatch.setattr(
        sentiment,
        "fetch_form4_filings",
        lambda _ticker: [
            {"type": "buy", "dollar_amount": 75_000},
            {"type": "buy", "dollar_amount": 49_999},
            {"type": "sell", "dollar_amount": 200_000},
        ],
    )
    monkeypatch.setattr(sentiment.pytime, "time", lambda: 1000.0)

    assert sentiment.fetch_sentiment(None, "AAPL") == pytest.approx(0.18)
    assert analyzed == [
        "strong beat. guidance raised",
        "margin pressure. costs rose",
    ]
    assert sentiment._sentiment_cache["AAPL"] == (1000.0, pytest.approx(0.18))
    evidence = sentiment.get_sentiment_evidence("AAPL")
    assert evidence is not None
    assert evidence["source"] == "newsapi_finbert_form4"
    assert evidence["authoritative"] is True
    assert evidence["provenance"]["scored_article_count"] == 2


def test_fetch_sentiment_bad_payload_falls_back_and_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failures: list[tuple[str, str | None]] = []

    sentiment._set_sentiment_http_session_for_tests(
        SimpleNamespace(
            get=lambda *_a, **_k: _Response(payload={"articles": {"title": "not-a-list"}})
        )
    )
    monkeypatch.setattr(sentiment, "_record_sentiment_failure", lambda reason, error=None: failures.append((reason, error)))
    monkeypatch.setattr(sentiment.pytime, "time", lambda: 42.0)

    assert sentiment.fetch_sentiment(None, "MSFT") == 0.0
    assert failures == [("api_error", "sentiment API articles must be a list")]
    assert sentiment._sentiment_cache["MSFT"] == (42.0, 0.0)


def test_fetch_sentiment_provider_exception_uses_existing_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failures: list[tuple[str, str | None]] = []
    sentiment._sentiment_cache["TSLA"] = (100.0, -0.35)

    def fail_get(*_args: object, **_kwargs: object) -> object:
        raise sentiment.RequestException("provider offline")

    sentiment._set_sentiment_http_session_for_tests(SimpleNamespace(get=fail_get))
    monkeypatch.setattr(sentiment, "_record_sentiment_failure", lambda reason, error=None: failures.append((reason, error)))
    monkeypatch.setattr(sentiment.pytime, "time", lambda: 1000.0)

    assert sentiment.fetch_sentiment(None, "TSLA") == -0.35
    assert failures == [("api_error", "provider offline")]


def test_fetch_sentiment_http_provider_fallback_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failures: list[tuple[str, str | None]] = []
    monkeypatch.setattr(sentiment, "_record_sentiment_failure", lambda reason, error=None: failures.append((reason, error)))
    monkeypatch.setattr(sentiment.pytime, "time", lambda: 10.0)

    sentiment._set_sentiment_http_session_for_tests(
        SimpleNamespace(get=lambda *_a, **_k: _Response(403))
    )
    assert sentiment.fetch_sentiment(None, "META") == 0.0

    sentiment._set_sentiment_http_session_for_tests(
        SimpleNamespace(get=lambda *_a, **_k: _Response(503))
    )
    assert sentiment.fetch_sentiment(None, "NVDA") == 0.0

    assert failures == [("forbidden", None), ("server_error", "503")]


def test_fetch_sentiment_rate_limit_delegates_to_enhanced_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[str] = []

    sentiment._set_sentiment_http_session_for_tests(
        SimpleNamespace(get=lambda *_a, **_k: _Response(429))
    )
    monkeypatch.setattr(
        sentiment,
        "_handle_rate_limit_with_enhanced_strategies",
        lambda ticker: called.append(ticker) or 0.27,
    )

    assert sentiment.fetch_sentiment(None, "QQQ") == 0.27
    assert called == ["QQQ"]


def test_fetch_sentiment_requires_minimum_scored_news_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentiment._set_sentiment_http_session_for_tests(
        SimpleNamespace(
            get=lambda *_a, **_k: _Response(
                payload={"articles": [{"title": "", "description": ""}]}
            )
        )
    )
    monkeypatch.setattr(sentiment, "fetch_form4_filings", lambda _ticker: [])
    monkeypatch.setattr(sentiment.pytime, "time", lambda: 1000.0)

    assert sentiment.fetch_sentiment(None, "AAPL") == 0.0
    assert "AAPL" not in sentiment._sentiment_cache
    evidence = sentiment.get_sentiment_evidence("AAPL")
    assert evidence is not None
    assert evidence["authoritative"] is False
    assert evidence["reason"] == "insufficient_newsapi_evidence"


def test_fetch_sentiment_preserves_special_ticker_cache_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentiment._sentiment_cache["BRK.B"] = (980.0, 0.44)

    def fail_if_called(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("cached ticker should not call provider")

    sentiment._set_sentiment_http_session_for_tests(SimpleNamespace(get=fail_if_called))
    monkeypatch.setattr(sentiment.pytime, "time", lambda: 1000.0)

    assert sentiment.fetch_sentiment(None, "BRK.B") == 0.44


def test_analyze_text_missing_dependencies_and_inference_errors_are_neutral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sentiment, "_load_transformers", lambda _logger=None: None)

    assert sentiment.analyze_text("") == {
        "available": False,
        "pos": 0.0,
        "neg": 0.0,
        "neu": 1.0,
    }

    class Torch:
        @staticmethod
        def no_grad() -> Any:
            class Context:
                def __enter__(self) -> None:
                    return None

                def __exit__(self, *_args: object) -> None:
                    return None

            return Context()

    class Model:
        def __call__(self, **_inputs: Any) -> Any:
            raise TypeError("bad logits")

    sentiment._SENTIMENT_STUB_LOGGED = False
    monkeypatch.setattr(
        sentiment,
        "_load_transformers",
        lambda _logger=None: (Torch, lambda *_a, **_k: {"ids": [1]}, Model()),
    )
    monkeypatch.setattr(sentiment, "tensors_to_device", lambda inputs, _device: inputs)

    assert sentiment.analyze_text("   ") == {
        "available": False,
        "pos": 0.0,
        "neg": 0.0,
        "neu": 1.0,
    }


def test_analyze_text_dependency_loader_provider_failure_is_neutral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sentiment,
        "_load_transformers",
        lambda _logger=None: (_ for _ in ()).throw(OSError("model unavailable")),
    )

    assert sentiment.analyze_text("earnings call transcript") == {
        "available": False,
        "pos": 0.0,
        "neg": 0.0,
        "neu": 1.0,
    }


def test_optional_dependency_loaders_cache_success_and_log_missing_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[dict[str, str]] = []
    original_import = builtins.__import__

    class Log:
        @staticmethod
        def warning(_message: str, *, extra: dict[str, str]) -> None:
            warnings.append(extra)

    class Tokenizer:
        @staticmethod
        def from_pretrained(name: str, *, local_files_only: bool = False) -> str:
            assert name == "yiyanghkust/finbert-tone"
            assert local_files_only is True
            return "tokenizer"

    class ModelFactory:
        @staticmethod
        def from_pretrained(name: str, *, local_files_only: bool = False) -> Any:
            assert name == "yiyanghkust/finbert-tone"
            assert local_files_only is True

            class Model:
                def to(self, device: str) -> None:
                    assert device == "cpu"

                def eval(self) -> None:
                    return None

            return Model()

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "bs4":
            return SimpleNamespace(BeautifulSoup="soup")
        if name == "torch":
            return "torch"
        if name == "transformers":
            return SimpleNamespace(
                AutoTokenizer=Tokenizer,
                AutoModelForSequenceClassification=ModelFactory,
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert sentiment._load_bs4(Log) == "soup"
    assert sentiment._load_bs4(Log) == "soup"
    loaded_transformers = sentiment._load_transformers(Log)
    assert loaded_transformers[:2] == ("torch", "tokenizer")
    assert sentiment._load_transformers(Log) is loaded_transformers
    assert warnings == []

    sentiment._bs4 = None
    sentiment._transformers_bundle = None

    def missing_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in {"bs4", "torch", "transformers"}:
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_import)

    assert sentiment._load_bs4(Log) is None
    assert sentiment._load_bs4(Log) is None
    assert sentiment._load_transformers(Log) is None
    assert sentiment._load_transformers(Log) is None
    assert warnings == [{"package": "bs4"}, {"package": "transformers"}]


def test_transformer_loader_requires_pinned_revision_outside_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sentiment, "_transformers_bundle", None)
    monkeypatch.setattr(sentiment, "_finbert_revision_required", lambda: True)
    monkeypatch.delenv("AI_TRADING_FINBERT_MODEL_REVISION", raising=False)

    with pytest.raises(RuntimeError, match="revision is not pinned"):
        sentiment._load_transformers()


def test_transformer_loader_falls_back_to_explicit_bert_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    class Log:
        @staticmethod
        def warning(_message: str, *, extra: dict[str, str]) -> None:
            raise AssertionError(extra)

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(_name: str, *, local_files_only: bool = False) -> str:
            assert local_files_only is True
            raise ValueError("auto tokenizer unavailable")

    class AutoModel:
        @staticmethod
        def from_pretrained(_name: str, *, local_files_only: bool = False) -> str:
            raise AssertionError("auto model should not be used after tokenizer failure")

    class BertTokenizer:
        @staticmethod
        def from_pretrained(name: str, *, local_files_only: bool = False) -> str:
            assert name == "yiyanghkust/finbert-tone"
            assert local_files_only is True
            return "bert-tokenizer"

    class BertModel:
        @staticmethod
        def from_pretrained(name: str, *, local_files_only: bool = False) -> Any:
            assert name == "yiyanghkust/finbert-tone"
            assert local_files_only is True

            class Model:
                def to(self, device: str) -> None:
                    assert device == "cpu"

                def eval(self) -> None:
                    return None

            return Model()

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch":
            return "torch"
        if name == "transformers":
            return SimpleNamespace(
                AutoTokenizer=AutoTokenizer,
                AutoModelForSequenceClassification=AutoModel,
                BertTokenizer=BertTokenizer,
                BertForSequenceClassification=BertModel,
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sentiment._transformers_bundle = None

    loaded = sentiment._load_transformers(Log)

    assert loaded[:2] == ("torch", "bert-tokenizer")
