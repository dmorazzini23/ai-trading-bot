from __future__ import annotations

import importlib
import sys
import types


def test_get_test_client_handles_missing_flask_testing(monkeypatch):
    """`get_test_client` returns ``None`` when `flask.testing` is missing."""
    flask_stub = types.ModuleType("flask")
    flask_stub.__path__ = []  # mark as package
    flask_app_stub = types.ModuleType("flask.app")

    class Flask:  # minimal stub
        def __init__(self, *a, **k):
            pass

    flask_app_stub.Flask = Flask
    flask_stub.Flask = Flask
    flask_stub.jsonify = lambda *a, **k: {}

    monkeypatch.setitem(sys.modules, "flask", flask_stub)
    monkeypatch.setitem(sys.modules, "flask.app", flask_app_stub)
    monkeypatch.delitem(sys.modules, "flask.testing", raising=False)
    monkeypatch.setitem(sys.modules, "testing", types.ModuleType("testing"))

    app_module = importlib.reload(importlib.import_module("ai_trading.app"))

    assert app_module.get_test_client() is None
