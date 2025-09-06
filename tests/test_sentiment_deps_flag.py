import importlib
import sys
import types
from unittest.mock import patch


def test_sentiment_dep_flag_updates(monkeypatch):
    sentiment = importlib.reload(importlib.import_module("ai_trading.analysis.sentiment"))
    assert hasattr(sentiment, "_SENT_DEPS_LOGGED")
    assert sentiment._SENT_DEPS_LOGGED is False

    dummy_bs4 = types.ModuleType("bs4")
    monkeypatch.setitem(sys.modules, "bs4", dummy_bs4)
    sentiment._sentiment_deps_logged.clear()
    with patch.object(sentiment.logger, "warning") as warn:
        sentiment._load_bs4()
        assert warn.called
    assert sentiment._SENT_DEPS_LOGGED is True
