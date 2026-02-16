from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest


def _reload_bot_engine():
    return importlib.reload(__import__("ai_trading.core.bot_engine", fromlist=["dummy"]))


def test_order_type_failfast_requires_capabilities_in_paper(monkeypatch) -> None:
    be = _reload_bot_engine()
    runtime = SimpleNamespace(cfg=SimpleNamespace(execution_mode="paper", testing=False))
    monkeypatch.setattr(be, "_is_testing_env", lambda: False)
    monkeypatch.setattr(be, "is_runtime_contract_testing_mode", lambda: False)
    monkeypatch.setenv("AI_TRADING_ORDER_TYPES_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_ORDER_TYPE_FAILFAST_ON_UNSUPPORTED", "1")
    monkeypatch.setenv("AI_TRADING_DEFAULT_ENTRY_ORDER_TYPE", "limit")
    monkeypatch.setenv("AI_TRADING_DEFAULT_EXIT_ORDER_TYPE", "stop_limit")
    monkeypatch.setenv("AI_TRADING_ALLOW_BRACKET_ORDERS", "0")
    monkeypatch.setenv("AI_TRADING_ALLOW_OCO_OTO", "0")

    with pytest.raises(RuntimeError, match="ORDER_TYPE_CAPABILITIES_MISSING"):
        be._enforce_order_type_failfast(runtime, cfg=runtime.cfg, execution_mode="paper")


def test_order_type_failfast_accepts_supported_configuration(monkeypatch) -> None:
    be = _reload_bot_engine()
    runtime = SimpleNamespace(
        cfg=SimpleNamespace(execution_mode="paper", testing=False),
        broker_order_type_capabilities={
            "limit": True,
            "market": True,
            "stop": True,
            "stop_limit": True,
            "trailing_stop": True,
            "bracket": True,
            "oco": True,
            "oto": True,
        },
    )
    monkeypatch.setattr(be, "_is_testing_env", lambda: False)
    monkeypatch.setattr(be, "is_runtime_contract_testing_mode", lambda: False)
    monkeypatch.setenv("AI_TRADING_ORDER_TYPES_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_ORDER_TYPE_FAILFAST_ON_UNSUPPORTED", "1")
    monkeypatch.setenv("AI_TRADING_DEFAULT_ENTRY_ORDER_TYPE", "limit")
    monkeypatch.setenv("AI_TRADING_DEFAULT_EXIT_ORDER_TYPE", "stop_limit")
    monkeypatch.setenv("AI_TRADING_ALLOW_BRACKET_ORDERS", "0")
    monkeypatch.setenv("AI_TRADING_ALLOW_OCO_OTO", "0")

    be._enforce_order_type_failfast(runtime, cfg=runtime.cfg, execution_mode="paper")


def test_order_type_failfast_infers_live_engine_capabilities(monkeypatch) -> None:
    be = _reload_bot_engine()

    class _LiveExec:
        pass

    _LiveExec.__module__ = "ai_trading.execution.live_trading"

    runtime = SimpleNamespace(
        cfg=SimpleNamespace(execution_mode="paper", testing=False),
        execution_engine=_LiveExec(),
    )
    monkeypatch.setattr(be, "_is_testing_env", lambda: False)
    monkeypatch.setattr(be, "is_runtime_contract_testing_mode", lambda: False)
    monkeypatch.setenv("AI_TRADING_ORDER_TYPES_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_ORDER_TYPE_FAILFAST_ON_UNSUPPORTED", "1")
    monkeypatch.setenv("AI_TRADING_DEFAULT_ENTRY_ORDER_TYPE", "limit")
    monkeypatch.setenv("AI_TRADING_DEFAULT_EXIT_ORDER_TYPE", "stop_limit")
    monkeypatch.setenv("AI_TRADING_ALLOW_BRACKET_ORDERS", "0")
    monkeypatch.setenv("AI_TRADING_ALLOW_OCO_OTO", "0")

    be._enforce_order_type_failfast(runtime, cfg=runtime.cfg, execution_mode="paper")
    assert runtime.broker_order_type_capabilities["limit"] is True
