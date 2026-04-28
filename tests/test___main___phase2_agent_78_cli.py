from __future__ import annotations

import sys
import types
from argparse import Namespace

import pytest

import ai_trading
from ai_trading import __main__ as cli


def test_validate_env_alias_consistency_raises_on_numeric_conflict(monkeypatch) -> None:
    values = {
        "MAX_ORDER_DOLLARS": "100",
        "AI_TRADING_MAX_ORDER_DOLLARS": "101",
    }
    monkeypatch.setattr(cli, "_env_text", lambda name, default="": values.get(name, default))

    with pytest.raises(SystemExit, match="MAX_ORDER_DOLLARS=100 conflicts"):
        cli._validate_env_alias_consistency()


def test_validate_startup_config_rejects_live_sqlite_database(monkeypatch) -> None:
    values = {
        "EXECUTION_MODE": "live",
        "AI_TRADING_OMS_INTENT_STORE_ENABLED": "1",
        "DATABASE_URL": "sqlite:///orders.db",
        "BACKUP_DATA_PROVIDER": "alpaca",
        "TIMEFRAME": "1Min",
    }
    monkeypatch.setattr(cli, "_env_text", lambda name, default="": values.get(name, default))
    monkeypatch.setattr(cli, "_validate_env_alias_consistency", lambda: None)

    management = types.ModuleType("ai_trading.config.management")
    management.validate_no_deprecated_env = lambda: None  # type: ignore[attr-defined]
    settings = types.ModuleType("ai_trading.settings")
    settings.get_settings = lambda: types.SimpleNamespace(alpaca_data_feed="iex")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ai_trading.config.management", management)
    monkeypatch.setitem(sys.modules, "ai_trading.settings", settings)

    with pytest.raises(SystemExit, match="non-sqlite database"):
        cli._validate_startup_config()


def test_run_loop_keyboard_interrupt_requests_stop_and_exits(monkeypatch) -> None:
    requests: list[str] = []
    monkeypatch.setattr(cli, "should_stop", lambda: False)
    monkeypatch.setattr(cli, "request_stop", lambda reason: requests.append(reason))

    with pytest.raises(SystemExit) as exc:
        cli._run_loop(
            lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
            Namespace(once=False, interval=0),
            "Trade",
        )

    assert exc.value.code == 0
    assert requests == ["keyboard-interrupt"]


def test_main_maps_once_interval_and_cancels_timer(monkeypatch) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setattr(cli, "_env_text", lambda name, default="": "")

    env_module = types.ModuleType("ai_trading.env")
    env_module.ensure_dotenv_loaded = lambda: calls.setdefault("dotenv", True)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ai_trading.env", env_module)

    main_module = types.ModuleType("ai_trading.main")
    main_module.preflight_import_health = lambda: True  # type: ignore[attr-defined]
    main_module.should_enforce_strict_import_preflight = lambda: False  # type: ignore[attr-defined]

    def _main(argv=None):
        calls["mapped_argv"] = list(argv or [])

    main_module.main = _main  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ai_trading.main", main_module)
    monkeypatch.setattr(ai_trading, "main", main_module, raising=False)
    monkeypatch.setattr(cli, "_validate_startup_config", lambda: calls.setdefault("validated", True))
    monkeypatch.setattr(cli, "_set_execution_mode", lambda *, paper: calls.setdefault("paper", paper))
    monkeypatch.setattr(cli.stop_event, "clear", lambda: calls.setdefault("stop_cleared", True))

    class _Timer:
        def cancel(self) -> None:
            calls["timer_cancelled"] = True

    monkeypatch.setattr(cli, "install_runtime_timer", lambda seconds: _Timer())
    monkeypatch.setattr(cli, "request_stop", lambda reason: calls.setdefault("stop_reason", reason))

    rc = cli.main(["--live", "--once", "--interval", "0.25", "--max-runtime-seconds", "3"])

    assert rc == 0
    assert calls["mapped_argv"] == ["--interval", "0.25", "--iterations", "1"]
    assert calls["paper"] is False
    assert calls["timer_cancelled"] is True


def test_main_without_mode_flag_does_not_override_execution_mode(monkeypatch) -> None:
    calls: dict[str, object] = {}
    mode_calls: list[bool] = []
    monkeypatch.setattr(cli, "_env_text", lambda name, default="": "")

    env_module = types.ModuleType("ai_trading.env")
    env_module.ensure_dotenv_loaded = lambda: calls.setdefault("dotenv", True)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ai_trading.env", env_module)

    main_module = types.ModuleType("ai_trading.main")
    main_module.preflight_import_health = lambda: True  # type: ignore[attr-defined]
    main_module.should_enforce_strict_import_preflight = lambda: False  # type: ignore[attr-defined]
    main_module.main = lambda argv=None: calls.setdefault("mapped_argv", list(argv or []))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ai_trading.main", main_module)
    monkeypatch.setattr(ai_trading, "main", main_module, raising=False)
    monkeypatch.setattr(cli, "_validate_startup_config", lambda: calls.setdefault("validated", True))
    monkeypatch.setattr(cli, "_set_execution_mode", lambda *, paper: mode_calls.append(paper))
    monkeypatch.setattr(cli.stop_event, "clear", lambda: calls.setdefault("stop_cleared", True))

    rc = cli.main(["--once", "--interval", "0"])

    assert rc == 0
    assert calls["mapped_argv"] == ["--interval", "0.0", "--iterations", "1"]
    assert mode_calls == []


def test_main_legacy_apca_runtime_error_returns_config_code(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_env_text", lambda name, default="": "")
    env_module = types.ModuleType("ai_trading.env")
    env_module.ensure_dotenv_loaded = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ai_trading.env", env_module)

    main_module = types.ModuleType("ai_trading.main")
    main_module.preflight_import_health = lambda: True  # type: ignore[attr-defined]
    main_module.should_enforce_strict_import_preflight = lambda: False  # type: ignore[attr-defined]
    main_module.main = lambda _argv=None: (_ for _ in ()).throw(RuntimeError("APCA_API_KEY"))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ai_trading.main", main_module)
    monkeypatch.setattr(ai_trading, "main", main_module, raising=False)
    monkeypatch.setattr(cli, "_validate_startup_config", lambda: None)

    assert cli.main(["--once", "--interval", "0"]) == 78
