import json
import subprocess
import sys
import os


def _imported_alpaca_modules(mod: str) -> list[str]:
    code = (
        "import sys, json, importlib.abc\n"
        "class Blocker(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path, target=None):\n"
        "        if fullname.startswith('alpaca'):\n"
        "            raise ImportError('blocked')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        f"try:\n    import {mod}\nexcept Exception:\n    pass\n"
        "print(json.dumps([m for m in sys.modules if m.startswith('alpaca')]))\n"
    )
    env = {
        **os.environ,
        "ALPACA_API_KEY": "dummy",
        "ALPACA_SECRET_KEY": "dummy",
        "ALPACA_BASE_URL": "https://paper-api.alpaca.markets",
    }
    out = subprocess.check_output([sys.executable, "-c", code], env=env)
    last = out.decode().strip().splitlines()[-1]
    return json.loads(last)


def test_bot_engine_lazy_alpaca_import():
    mods = _imported_alpaca_modules("ai_trading.core.bot_engine")
    assert mods == []


def test_self_check_lazy_alpaca_import():
    mods = _imported_alpaca_modules("ai_trading.scripts.self_check")
    assert mods == []
