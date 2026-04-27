from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

from ai_trading import production_system as ps
from ai_trading.core.enums import OrderSide, OrderType, RiskLevel


def _system_alerts(system: ps.ProductionTradingSystem) -> list[tuple[object, ...]]:
    return cast(list[tuple[object, ...]], cast(Any, system.alert_manager).system_alerts)


class _FakeAlertManager:
    def __init__(self):
        self.processing = False
        self.system_alerts: list[tuple[object, ...]] = []
        self.trading_alerts: list[tuple[object, ...]] = []

    def start_processing(self):
        self.processing = True

    def stop_processing(self):
        self.processing = False

    async def send_system_alert(self, *args):
        self.system_alerts.append(args)

    async def send_trading_alert(self, *args):
        self.trading_alerts.append(args)

    def get_alert_stats(self):
        return {"processing_active": self.processing, "queue_size": 0}


class _FakeSyncAlertManager(_FakeAlertManager):
    def send_system_alert(self, *args):
        self.system_alerts.append(args)
        return "system-alert"

    def send_trading_alert(self, *args):
        self.trading_alerts.append(args)
        return "trading-alert"


class _FakeHaltManager:
    def __init__(self):
        self.allowed = True
        self.reasons: list[str] = []
        self.equity_updates: list[float] = []
        self.emergency_reasons: list[str] = []

    def is_trading_allowed(self):
        return {"trading_allowed": self.allowed, "reasons": list(self.reasons)}

    def update_equity(self, equity):
        self.equity_updates.append(equity)

    def emergency_stop_all(self, reason):
        self.emergency_reasons.append(reason)
        self.allowed = False


class _FakeRiskManager:
    def __init__(self, _risk_level):
        self.approved = True

    def assess_trade_risk(self, *_args):
        if self.approved:
            return {"approved": True, "recommended_size": 7, "warnings": []}
        return {"approved": False, "recommended_size": 0, "warnings": ["risk blocked"]}


class _FakeDashboard:
    def __init__(self, _alert_manager):
        self.positions: list[tuple[object, ...]] = []

    def update_position(self, *args):
        self.positions.append(args)

    def get_dashboard_summary(self):
        return {"performance_metrics": {"pnl": 12.5}, "portfolio": "ok"}


class _FakeExecutionCoordinator:
    def __init__(self, account_equity, risk_level):
        self.account_equity = account_equity
        self.risk_level = risk_level
        self.next_result: object = {
            "status": "success",
            "symbol": "AAPL",
            "quantity": 7,
            "fill_price": 101.5,
        }
        self.equity_updates: list[float] = []
        self.submitted: list[tuple[object, ...]] = []

    async def submit_order(self, *_args):
        self.submitted.append(_args)
        return self.next_result

    def get_execution_summary(self):
        return {"execution_stats": {"total_orders": 1}}

    def update_account_equity(self, equity):
        self.equity_updates.append(equity)


class _FakeLiquidityManager:
    def __init__(self):
        self.next_liquidity = {
            "liquidity_level": SimpleNamespace(value="normal"),
            "execution_recommendations": {
                "recommended_order_type": OrderType.LIMIT,
                "execution_strategy": "patient",
                "risk_warnings": ["wide spread"],
            },
        }

    def update_symbol_liquidity(self, *_args):
        return self.next_liquidity

    def get_portfolio_liquidity_summary(self):
        return {"liquidity": "ok"}


class _FakeMtfAnalyzer:
    def __init__(self):
        self.next_analysis = {"recommendation": {"action": "BUY", "confidence": 0.8}}

    def analyze_symbol(self, *_args):
        return self.next_analysis


class _FakeRegimeDetector:
    def __init__(self):
        self.multiplier = 1.0

    def detect_regime(self, price_data):
        return {"regime": "trend", "price_points": len(price_data)}

    def get_regime_recommendations(self):
        return {"position_size_multiplier": self.multiplier}

    def get_regime_summary(self):
        return {"regime": "trend"}


def _build_system(monkeypatch, *, equity: float = 100_000.0):
    monkeypatch.setattr(ps, "AlertManager", _FakeAlertManager)
    monkeypatch.setattr(ps, "TradingHaltManager", _FakeHaltManager)
    monkeypatch.setattr(ps, "RiskManager", _FakeRiskManager)
    monkeypatch.setattr(ps, "DynamicPositionSizer", lambda _risk_level: object())
    monkeypatch.setattr(ps, "PerformanceDashboard", _FakeDashboard)
    monkeypatch.setattr(ps, "ProductionExecutionCoordinator", _FakeExecutionCoordinator)
    monkeypatch.setattr(ps, "LiquidityManager", _FakeLiquidityManager)
    monkeypatch.setattr(ps, "MultiTimeframeAnalyzer", _FakeMtfAnalyzer)
    monkeypatch.setattr(ps, "RegimeDetector", _FakeRegimeDetector)
    return ps.ProductionTradingSystem(equity, RiskLevel.MODERATE)


def test_start_and_stop_system_success(monkeypatch):
    system = _build_system(monkeypatch)

    started = asyncio.run(system.start_system())
    stopped = asyncio.run(system.stop_system("done"))

    assert started["status"] == "success"
    assert started["risk_level"] == RiskLevel.MODERATE.value
    assert stopped["status"] == "success"
    assert stopped["shutdown_reason"] == "done"
    assert not system.is_active
    assert _system_alerts(system)[-1][1] == "System Stopped"


def test_start_system_accepts_sync_alert_manager(monkeypatch):
    monkeypatch.setattr(ps, "AlertManager", _FakeSyncAlertManager)
    monkeypatch.setattr(ps, "TradingHaltManager", _FakeHaltManager)
    monkeypatch.setattr(ps, "RiskManager", _FakeRiskManager)
    monkeypatch.setattr(ps, "DynamicPositionSizer", lambda _risk_level: object())
    monkeypatch.setattr(ps, "PerformanceDashboard", _FakeDashboard)
    monkeypatch.setattr(ps, "ProductionExecutionCoordinator", _FakeExecutionCoordinator)
    monkeypatch.setattr(ps, "LiquidityManager", _FakeLiquidityManager)
    monkeypatch.setattr(ps, "MultiTimeframeAnalyzer", _FakeMtfAnalyzer)
    monkeypatch.setattr(ps, "RegimeDetector", _FakeRegimeDetector)
    system = ps.ProductionTradingSystem(100_000.0, RiskLevel.MODERATE)

    result = asyncio.run(system.start_system())

    assert result["status"] == "success"
    assert _system_alerts(system)[-1][1] == "System Started"


def test_start_system_fails_when_health_check_reports_issue(monkeypatch):
    system = _build_system(monkeypatch, equity=0.0)

    result = asyncio.run(system.start_system())

    assert result == {"status": "failed", "reason": "Health check failed"}
    assert _system_alerts(system)[-1][1] == "Startup Health Check Failed"


def test_analyze_trading_opportunity_halted_and_full_path(monkeypatch):
    system = _build_system(monkeypatch)
    system.halt_manager.allowed = False
    system.halt_manager.reasons = ["maintenance"]

    halted = asyncio.run(system.analyze_trading_opportunity("AAPL", {}))
    assert halted["recommendation"] == "NO_TRADE"
    assert "maintenance" in halted["reason"]

    system.halt_manager.allowed = True
    system.halt_manager.reasons = []
    full = asyncio.run(
        system.analyze_trading_opportunity(
            "AAPL",
            {
                "price_data": [1, 2, 3],
                "timeframe_data": {"1m": []},
                "current_price": 101.5,
            },
        )
    )

    assert full["trading_allowed"] is True
    assert full["regime_analysis"]["regime"] == "trend"
    assert full["final_recommendation"]["action"] == "BUY"
    assert full["final_recommendation"]["recommended_quantity"] == 7
    assert full["final_recommendation"]["execution_strategy"] == "patient"


def test_execute_trade_rejects_no_trade_and_records_success(monkeypatch):
    system = _build_system(monkeypatch)
    system.risk_manager.approved = False

    rejected = asyncio.run(
        system.execute_trade(
            "AAPL",
            OrderSide.BUY,
            5,
            market_data={"current_price": 100.0},
        )
    )

    assert rejected["status"] == "rejected"
    assert rejected["analysis"]["final_recommendation"]["action"] == "NO_TRADE"

    system.risk_manager.approved = True
    result = asyncio.run(system.execute_trade("AAPL", OrderSide.BUY, 5, price=101.5))

    assert result["status"] == "success"
    assert result["total_execution_time_seconds"] >= 0
    assert len(system.session_trades) == 1
    assert system.performance_dashboard.positions == [("AAPL", 7, 101.5, 101.5)]


def test_execute_trade_rejects_limit_without_price(monkeypatch):
    system = _build_system(monkeypatch)

    result = asyncio.run(system.execute_trade("AAPL", OrderSide.BUY, 5, order_type=OrderType.LIMIT))

    assert result == {"status": "rejected", "reason": "Limit price required for limit order"}
    assert system.execution_coordinator.submitted == []


def test_execute_trade_normalizes_object_execution_result(monkeypatch):
    system = _build_system(monkeypatch)
    system.execution_coordinator.next_result = SimpleNamespace(status="SUCCESS")

    result = asyncio.run(system.execute_trade("MSFT", OrderSide.SELL, 3, price=50.0))

    assert result["status"] == "success"
    assert result["symbol"] == "MSFT"
    assert result["quantity"] == 3
    assert system.session_trades[-1]["side"] == OrderSide.SELL.value


def test_health_status_session_summary_and_system_status(monkeypatch):
    system = _build_system(monkeypatch)
    assert asyncio.run(system.get_session_summary()) == {"error": "No active session"}

    asyncio.run(system.start_system())
    asyncio.run(system.execute_trade("AAPL", OrderSide.BUY, 7, price=101.5))
    status = asyncio.run(system.get_system_status())
    summary = asyncio.run(system.get_session_summary())

    assert status["system_active"] is True
    assert status["session_trades_count"] == 1
    assert status["portfolio_summary"]["portfolio"] == "ok"
    assert summary["total_trades"] == 1
    assert summary["successful_trades"] == 1
    assert summary["success_rate"] == 100


def test_integrate_analyses_sell_hold_low_liquidity_and_final_recommendation(monkeypatch):
    system = _build_system(monkeypatch)

    sell = asyncio.run(
        system._integrate_analyses(
            "AAPL",
            {},
            {"recommendation": {"action": "SELL", "confidence": 0.6}},
            {"liquidity_level": SimpleNamespace(value="low"), "execution_recommendations": {}},
        )
    )
    hold = asyncio.run(
        system._integrate_analyses(
            "AAPL",
            {},
            {"recommendation": {"action": "BUY", "confidence": 0.9}},
            {"liquidity_level": SimpleNamespace(value="normal"), "execution_recommendations": {}},
        )
    )
    blocked = asyncio.run(
        system._generate_final_recommendation(
            {"action": "BUY", "confidence": 0.7, "reasoning": []},
            {"approved": False, "warnings": ["too risky"]},
            {},
        )
    )

    assert sell["action"] == "SELL"
    assert sell["confidence"] == 0.3
    assert "Reduced confidence" in sell["reasoning"][-1]
    assert hold["action"] == "BUY"
    assert blocked["action"] == "NO_TRADE"
    assert blocked["warnings"] == ["too risky"]


def test_update_equity_emergency_shutdown_and_health(monkeypatch):
    system = _build_system(monkeypatch)
    system.is_active = True

    system.update_account_equity(125_000.0)
    assert system.account_equity == 125_000.0
    assert system.execution_coordinator.equity_updates == [125_000.0]
    assert system.halt_manager.equity_updates == [125_000.0]
    assert system.is_healthy()

    asyncio.run(system.emergency_shutdown("risk"))

    assert system.halt_manager.emergency_reasons == ["risk"]
    assert _system_alerts(system)[-2][1] == "EMERGENCY SHUTDOWN"
    assert not system.is_healthy()
