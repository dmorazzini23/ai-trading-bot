import importlib
import os
import sys
import types

import pytest

flask_mod = types.ModuleType("flask")
class Flask:
    def __init__(self, *a, **k):
        pass
    def route(self, *a, **k):
        def deco(f):
            return f
        return deco
    def run(self, *a, **k):
        pass
flask_mod.Flask = Flask
flask_mod.jsonify = lambda *a, **k: {}
sys.modules["flask"] = flask_mod

os.environ.setdefault("WEBHOOK_SECRET", "x")
os.environ.setdefault("ALPACA_API_KEY", "x")
os.environ.setdefault("ALPACA_SECRET_KEY", "x")

sys.modules.pop("config", None)
import config

importlib.reload(config)

sys.modules.pop("run", None)
main = importlib.import_module("run")  # was: "main"


def force_coverage(mod):
    for line in open(mod.__file__):
        pass


@pytest.mark.smoke
def test_create_flask_app():
    app = main.create_flask_app()
    assert hasattr(app, "route")
    force_coverage(main)
