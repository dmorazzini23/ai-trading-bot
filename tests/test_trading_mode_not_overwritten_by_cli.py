from __future__ import annotations

import os
import sys
import types

import ai_trading
from ai_trading import __main__ as cli


def test_cli_paper_sets_execution_mode_only(monkeypatch) -> None:
    monkeypatch.setenv("TRADING_MODE", "balanced")
    monkeypatch.setenv("EXECUTION_MODE", "live")
    monkeypatch.delenv("PYTEST_RUNNING", raising=False)

    env_module = types.ModuleType("ai_trading.env")
    env_module.ensure_dotenv_loaded = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ai_trading.env", env_module)

    main_module = types.ModuleType("ai_trading.main")
    main_module.preflight_import_health = lambda: True  # type: ignore[attr-defined]
    main_module.should_enforce_strict_import_preflight = lambda: False  # type: ignore[attr-defined]
    main_module.main = lambda _argv=None: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ai_trading.main", main_module)
    monkeypatch.setattr(ai_trading, "main", main_module, raising=False)

    monkeypatch.setattr(cli, "_validate_startup_config", lambda: None)

    rc = cli.main(["--paper", "--once", "--interval", "0"])
    assert rc == 0
    assert os.environ["TRADING_MODE"] == "balanced"
    assert os.environ["EXECUTION_MODE"] == "paper"
