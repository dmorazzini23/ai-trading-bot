from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace


def _override_env(env_overrides: dict[str, str | None]) -> dict[str, str | None]:
    original: dict[str, str | None] = {}
    for key, value in env_overrides.items():
        original[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    return original


def _restore_env(snapshot: dict[str, str | None]) -> None:
    for key, value in snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_execution_engine_instantiated_without_stub(request) -> None:
    """Simulate production configuration and ensure the real execution engine is attached."""

    overrides = {
        "APP_ENV": "prod",
        "EXECUTION_MODE": "sim",
        "AI_TRADING_MAX_POSITION_SIZE": "10000",
    }
    snapshot = _override_env(overrides)

    reload_func = None
    try:
        from ai_trading.config.management import reload_trading_config
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency missing
        if getattr(exc, "name", "") not in {"pydantic", "pydantic_settings"}:
            raise
    else:
        reload_trading_config()
        reload_func = reload_trading_config

    def _finalize_env() -> None:
        _restore_env(snapshot)
        if reload_func is not None:
            reload_func()

    request.addfinalizer(_finalize_env)

    portalocker_stub = types.ModuleType("portalocker")
    portalocker_stub.LOCK_EX = 1

    def _noop(*_args, **_kwargs) -> None:
        return None

    portalocker_stub.lock = _noop  # type: ignore[attr-defined]
    portalocker_stub.unlock = _noop  # type: ignore[attr-defined]
    sys.modules.setdefault("portalocker", portalocker_stub)

    bs4_stub = types.ModuleType("bs4")

    class _BeautifulSoup:  # pragma: no cover - test stub
        def __init__(self, *_args, **_kwargs) -> None:  # noqa: D401 - simple stub
            """Placeholder BeautifulSoup stub used for import compatibility."""

    bs4_stub.BeautifulSoup = _BeautifulSoup  # type: ignore[attr-defined]
    sys.modules.setdefault("bs4", bs4_stub)

    from ai_trading.core import bot_engine

    original_exec_engine = getattr(bot_engine, "_exec_engine", None)

    def _restore_exec_engine() -> None:
        from ai_trading.core import bot_engine as _bot_engine

        _bot_engine._exec_engine = original_exec_engine

    request.addfinalizer(_restore_exec_engine)

    bot_engine._exec_engine = None
    runtime = SimpleNamespace()
    bot_engine._ensure_execution_engine(runtime)

    exec_engine = getattr(runtime, "execution_engine", None)
    assert exec_engine is not None, "execution engine should be attached"
    assert not getattr(exec_engine, "_IS_STUB", False), "stub execution engine must not be used"
    assert hasattr(exec_engine, "check_stops"), "risk-stop hook should be available"
