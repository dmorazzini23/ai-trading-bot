import json
import os
import subprocess
import sys
from typing import cast


def _imported_alpaca_modules(mod: str) -> list[str]:
    code = (
        "import sys, json, importlib.abc\n"
        "class Blocker(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path, target=None):\n"
        "        if fullname.startswith('alpaca'):\n"
        "            raise ImportError('blocked')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        f"import {mod}\n"
        "print(json.dumps([m for m in sys.modules if m.startswith('alpaca')]))\n"
    )
    env = os.environ.copy()
    for deprecated_key in ("ALPACA_BASE_URL", "ALPACA_API_URL", "TRADING_MODE"):
        env.pop(deprecated_key, None)
    env.update(
        {
            "ALPACA_API_KEY": "dummy",
            "ALPACA_SECRET_KEY": "dummy",
            "ALPACA_TRADING_BASE_URL": "https://paper-api.alpaca.markets",
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    last = result.stdout.strip().splitlines()[-1]
    return cast(list[str], json.loads(last))


def test_bot_engine_lazy_alpaca_import():
    mods = _imported_alpaca_modules("ai_trading.core.bot_engine")
    assert mods == []


def test_self_check_lazy_alpaca_import():
    mods = _imported_alpaca_modules("ai_trading.scripts.self_check")
    assert mods == []
