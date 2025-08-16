import builtins
import sys


def test_optional_ml_imports(monkeypatch):
    for mod in ("bs4", "transformers"):
        sys.modules.pop(mod, None)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("bs4") or name.startswith("transformers"):
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from ai_trading.analysis import sentiment

    res = sentiment.analyze_text("hello")
    sentiment.fetch_form4_filings("AAPL")

    assert res == {"available": False, "pos": 0.0, "neg": 0.0, "neu": 1.0}
    assert sentiment._SENT_DEPS_LOGGED == {"bs4", "transformers"}
