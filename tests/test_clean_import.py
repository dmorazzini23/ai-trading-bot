import importlib
import os
import subprocess
import sys


def test_package_import_has_no_cli_side_effects(monkeypatch):
    for name in list(sys.modules):
        if name.startswith("ai_trading"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    importlib.import_module("ai_trading")
    heavy_modules = {
        "ai_trading.__main__",
        "ai_trading.app",
        "ai_trading.main",
        "ai_trading.production_system",
        "ai_trading.core.run_all_trades",
    }
    assert heavy_modules.isdisjoint(sys.modules)


def test_bot_engine_import_is_quiet() -> None:
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = env.get("PYTHONWARNINGS", "ignore")

    proc = subprocess.run(
        [sys.executable, "-c", "import ai_trading.core.bot_engine"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    combined = f"{proc.stdout}\n{proc.stderr}"
    assert proc.returncode == 0, combined
    assert "INDICATOR_IMPORT_OK" not in combined
    assert "RL_IMPORT_OK" not in combined
    assert "RUNTIME_SETTINGS_RESOLVED" not in combined
    assert "TRADING_MODE_EFFECTIVE" not in combined
