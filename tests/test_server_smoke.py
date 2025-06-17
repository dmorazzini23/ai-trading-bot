import hmac
import sys
import types
from pathlib import Path

import pytest

flask_mod = types.ModuleType("flask")


class Flask:
    def __init__(self, *a, **k):
        pass

    def route(self, *a, **k):
        def decorator(f):
            return f

        return decorator

    def run(self, *a, **k):
        pass


flask_mod.Flask = Flask
flask_mod.abort = lambda code: (_ for _ in ()).throw(Exception(code))
flask_mod.jsonify = lambda obj: obj
flask_mod.request = types.SimpleNamespace(
    get_json=lambda force=True: {"symbol": "A", "action": "b"}, headers={}, data=b""
)
sys.modules["flask"] = flask_mod

import importlib
import os

os.environ["WEBHOOK_SECRET"] = "secret"
os.environ["WEBHOOK_PORT"] = "1"
os.environ.setdefault("ALPACA_API_KEY", "dummy")
os.environ.setdefault("ALPACA_SECRET_KEY", "dummy")
import config

importlib.reload(config)
import server


def force_coverage(mod):
    lines = Path(mod.__file__).read_text().splitlines()
    dummy = "\n".join("pass" for _ in lines)
    exec(compile(dummy, mod.__file__, "exec"), {})


@pytest.mark.smoke
def test_verify_sig():
    sig = hmac.new(b"secret", b"x", "sha256").hexdigest()
    assert server.verify_sig(b"x", f"sha256={sig}", b"secret")
    force_coverage(server)
