"""Real-time risk monitoring system for institutional trading."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Callable, Set
from uuid import UUID, uuid4
import numpy as np

from ..core.models import (
    TradePosition, PortfolioMetrics, RiskLimits, AlertEvent,
    TradingSignal, MarketData
)
from ..core.enums import RiskLevel, TradingSide, AssetClass
from ..core.exceptions import RiskLimitExceededError
from ..core.config import RiskManagementConfig
from ..core.logging import get_trading_logger


logger = get_trading_logger(__name__)


class RiskMonitor:
    """Real-time risk monitoring and alerting system."""
    
    def __init__(
        self,
        risk_config: RiskManagementConfig,
        alert_callback: Optional[Callable[[AlertEvent], None]] = None
    ):
        self.risk_config = risk_config
        self.alert_callback = alert_callback
        
        # Risk tracking
        self._positions: Dict[str, TradePosition] = {}
        self._portfolio_metrics: Optional[PortfolioMetrics] = None
        self._risk_violations: Set[str] = set()
        self._last_var_calculation = None
        self._var_history: List[float] = []
        
        # Monitoring state
        self._is_monitoring = False
        self._monitoring_task: Optional[asyncio.Task] = None
        
        logger.info("Risk monitor initialized with institutional risk limits")
    
    async def start_monitoring(self) -> None:
        """Start real-time risk monitoring."""
        if self._is_monitoring:
            logger.warning("Risk monitoring already active")
            return
        
        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Real-time risk monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop risk monitoring."""
        self._is_monitoring = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Risk monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._is_monitoring:
            try:
                await self._perform_risk_checks()
                await asyncio.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Error in risk monitoring loop: {e}")
                await asyncio.sleep(10)  # Wait longer on errors
    
    async def _perform_risk_checks(self) -> None:
        """Perform all risk checks."""
        if not self._portfolio_metrics:
            return
        
        # Portfolio-level checks
        await self._check_portfolio_exposure()
        await self._check_leverage_limits()
        await self._check_drawdown_limits()
        await self._check_var_limits()
        
        # Position-level checks
        await self._check_position_limits()
        await self._check_correlation_limits()
        await self._check_sector_exposure()
    
    def update_positions(self, positions: Dict[str, TradePosition]) -> None:
        """Update current positions for monitoring."""
        self._positions = positions.copy()
        logger.debug(f"Updated positions for risk monitoring: {len(positions)} positions")
    
    def update_portfolio_metrics(self, metrics: PortfolioMetrics) -> None:
        """Update portfolio metrics for monitoring."""
        self._portfolio_metrics = metrics
        logger.debug("Updated portfolio metrics for risk monitoring")
    
    async def check_trade_signal(self, signal: TradingSignal) -> bool:
        """Check if a trade signal violates risk limits.
        
        Args:
            signal: Trading signal to check
            
        Returns:
            True if signal is safe to execute
            
        Raises:
            RiskLimitExceededError: If signal violates risk limits
        """
        try:
            # Check if we have a position in this symbol
            existing_position = self._positions.get(signal.symbol)
            
            # Check position size limits
            if not existing_position:
                await self._check_new_position_limits(signal)
            else:
                await self._check_position_increase_limits(signal, existing_position)
            
            # Check portfolio exposure
            await self._check_signal_portfolio_impact(signal)
            
            # Check correlation with existing positions
            await self._check_signal_correlation_impact(signal)
            
            return True
            
        except RiskLimitExceededError:
            raise
        except Exception as e:
            logger.error(f"Error checking trade signal risk: {e}")
            # Err on the side of caution
            raise RiskLimitExceededError(
                limit_type="signal_check_error",
                current_value=0,
                limit_value=0,
                symbol=signal.symbol
            )
    
    async def _check_new_position_limits(self, signal: TradingSignal) -> None:
        """Check limits for new position."""
        if not self._portfolio_metrics:
            return
        
        # Estimate position size based on current portfolio
        estimated_position_value = (
            float(self._portfolio_metrics.total_value) * 
            float(self.risk_config.max_position_size)
        )
        
        current_exposure = (
            float(self._portfolio_metrics.gross_exposure) / 
            float(self._portfolio_metrics.total_value)
        )
        
        new_exposure = current_exposure + float(self.risk_config.max_position_size)
        
        if new_exposure > float(self.risk_config.max_portfolio_exposure):
            await self._emit_alert(
                "portfolio_exposure_limit",
                RiskLevel.HIGH,
                f"New position would exceed portfolio exposure limit: "
                f"{new_exposure:.2%} > {self.risk_config.max_portfolio_exposure:.2%}",
                new_exposure,
                float(self.risk_config.max_portfolio_exposure),
                signal.symbol
            )
            
            raise RiskLimitExceededError(
                limit_type="portfolio_exposure",
                current_value=new_exposure,
                limit_value=float(self.risk_config.max_portfolio_exposure),
                symbol=signal.symbol
            )
    
    async def _check_position_increase_limits(
        self,
        signal: TradingSignal,
        position: TradePosition
    ) -> None:
        """Check limits for increasing existing position."""
        if signal.side == position.side:
            # Increasing position in same direction
            current_position_pct = (
                float(abs(position.market_value)) / 
                float(self._portfolio_metrics.total_value)
            )
            
            if current_position_pct >= float(self.risk_config.max_position_size):
                await self._emit_alert(
                    "position_size_limit",
                    RiskLevel.HIGH,
                    f"Position already at maximum size: {current_position_pct:.2%}",
                    current_position_pct,
                    float(self.risk_config.max_position_size),
                    signal.symbol
                )
                
                raise RiskLimitExceededError(
                    limit_type="position_size",
                    current_value=current_position_pct,
                    limit_value=float(self.risk_config.max_position_size),
                    symbol=signal.symbol
                )
    
    async def _check_portfolio_exposure(self) -> None:
        """Check portfolio-level exposure limits."""
        if not self._portfolio_metrics:
            return
        
        exposure = (
            float(self._portfolio_metrics.gross_exposure) / 
            float(self._portfolio_metrics.total_value)
        )
        
        limit = float(self.risk_config.max_portfolio_exposure)
        
        if exposure > limit:
            alert_key = "portfolio_exposure_violation"
            if alert_key not in self._risk_violations:
                await self._emit_alert(
                    "portfolio_exposure",
                    RiskLevel.CRITICAL,
                    f"Portfolio exposure exceeds limit: {exposure:.2%} > {limit:.2%}",
                    exposure,
                    limit
                )
                self._risk_violations.add(alert_key)
        else:
            self._risk_violations.discard("portfolio_exposure_violation")
    
    async def _check_leverage_limits(self) -> None:
        """Check leverage limits."""
        if not self._portfolio_metrics:
            return
        
        leverage = float(self._portfolio_metrics.leverage)
        limit = float(self.risk_config.leverage_limit)
        
        if leverage > limit:
            alert_key = "leverage_violation"
            if alert_key not in self._risk_violations:
                await self._emit_alert(
                    "leverage_limit",
                    RiskLevel.HIGH,
                    f"Portfolio leverage exceeds limit: {leverage:.2f}x > {limit:.2f}x",
                    leverage,
                    limit
                )
                self._risk_violations.add(alert_key)
        else:
            self._risk_violations.discard("leverage_violation")
    
    async def _check_drawdown_limits(self) -> None:
        """Check drawdown limits."""
        if not self._portfolio_metrics or not self._portfolio_metrics.max_drawdown:
            return
        
        drawdown = abs(self._portfolio_metrics.max_drawdown)
        limit = float(self.risk_config.max_drawdown)
        
        if drawdown > limit:
            alert_key = "drawdown_violation"
            if alert_key not in self._risk_violations:
                await self._emit_alert(
                    "max_drawdown",
                    RiskLevel.CRITICAL,
                    f"Portfolio drawdown exceeds limit: {drawdown:.2%} > {limit:.2%}",
                    drawdown,
                    limit
                )
                self._risk_violations.add(alert_key)
        else:
            self._risk_violations.discard("drawdown_violation")
    
    async def _check_var_limits(self) -> None:
        """Check Value at Risk limits."""
        if not self._portfolio_metrics:
            return
        
        # Check 95% VaR
        if self._portfolio_metrics.var_95:
            var_95 = float(abs(self._portfolio_metrics.var_95)) / float(self._portfolio_metrics.total_value)
            limit_95 = float(self.risk_config.var_limit_95)
            
            if var_95 > limit_95:
                alert_key = "var_95_violation"
                if alert_key not in self._risk_violations:
                    await self._emit_alert(
                        "var_95_limit",
                        RiskLevel.HIGH,
                        f"95% VaR exceeds limit: {var_95:.2%} > {limit_95:.2%}",
                        var_95,
                        limit_95
                    )
                    self._risk_violations.add(alert_key)
            else:
                self._risk_violations.discard("var_95_violation")
        
        # Check 99% VaR
        if self._portfolio_metrics.var_99:
            var_99 = float(abs(self._portfolio_metrics.var_99)) / float(self._portfolio_metrics.total_value)
            limit_99 = float(self.risk_config.var_limit_99)
            
            if var_99 > limit_99:
                alert_key = "var_99_violation"
                if alert_key not in self._risk_violations:
                    await self._emit_alert(
                        "var_99_limit",
                        RiskLevel.CRITICAL,
                        f"99% VaR exceeds limit: {var_99:.2%} > {limit_99:.2%}",
                        var_99,
                        limit_99
                    )
                    self._risk_violations.add(alert_key)
            else:
                self._risk_violations.discard("var_99_violation")
    
    async def _check_position_limits(self) -> None:
        """Check individual position limits."""
        if not self._portfolio_metrics:
            return
        
        total_value = float(self._portfolio_metrics.total_value)
        max_position_size = float(self.risk_config.max_position_size)
        
        for symbol, position in self._positions.items():
            position_pct = float(abs(position.market_value)) / total_value
            
            if position_pct > max_position_size:
                alert_key = f"position_size_{symbol}"
                if alert_key not in self._risk_violations:
                    await self._emit_alert(
                        "position_size",
                        RiskLevel.HIGH,
                        f"Position size exceeds limit for {symbol}: "
                        f"{position_pct:.2%} > {max_position_size:.2%}",
                        position_pct,
                        max_position_size,
                        symbol
                    )
                    self._risk_violations.add(alert_key)
            else:
                self._risk_violations.discard(f"position_size_{symbol}")
    
    async def _check_correlation_limits(self) -> None:
        """Check position correlation limits."""
        if len(self._positions) < 2:
            return
        
        # This is a simplified correlation check
        # In practice, you'd calculate actual correlations from historical returns
        correlation_limit = float(self.risk_config.correlation_limit)
        
        # Group positions by asset class to check for concentration
        asset_class_exposure = {}
        total_value = float(self._portfolio_metrics.total_value) if self._portfolio_metrics else 1
        
        for position in self._positions.values():
            asset_class = position.asset_class
            exposure = float(abs(position.market_value)) / total_value
            
            if asset_class not in asset_class_exposure:
                asset_class_exposure[asset_class] = 0
            asset_class_exposure[asset_class] += exposure
        
        # Check if any asset class exceeds correlation limits
        for asset_class, exposure in asset_class_exposure.items():
            if exposure > correlation_limit:
                alert_key = f"correlation_{asset_class}"
                if alert_key not in self._risk_violations:
                    await self._emit_alert(
                        "correlation_limit",
                        RiskLevel.MEDIUM,
                        f"Asset class concentration exceeds limit for {asset_class}: "
                        f"{exposure:.2%} > {correlation_limit:.2%}",
                        exposure,
                        correlation_limit
                    )
                    self._risk_violations.add(alert_key)
            else:
                self._risk_violations.discard(f"correlation_{asset_class}")
    
    async def _check_sector_exposure(self) -> None:
        """Check sector exposure limits."""
        # This would require sector classification of symbols
        # Simplified implementation checking max_sector_exposure config
        pass
    
    async def _check_signal_portfolio_impact(self, signal: TradingSignal) -> None:
        """Check how signal would impact portfolio risk."""
        # Placeholder for more sophisticated portfolio impact analysis
        pass
    
    async def _check_signal_correlation_impact(self, signal: TradingSignal) -> None:
        """Check correlation impact of new signal."""
        # Placeholder for correlation analysis
        pass
    
    async def _emit_alert(
        self,
        alert_type: str,
        severity: RiskLevel,
        message: str,
        current_value: float,
        threshold_value: float,
        symbol: Optional[str] = None
    ) -> None:
        """Emit a risk alert."""
        alert = AlertEvent(
            correlation_id=uuid4(),
            alert_type=alert_type,
            severity=severity,
            message=message,
            source="risk_monitor",
            symbol=symbol,
            threshold_value=Decimal(str(threshold_value)),
            current_value=Decimal(str(current_value))
        )
        
        # Log the alert
        logger.risk_alert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            current_value=current_value,
            threshold_value=threshold_value,
            symbol=symbol
        )
        
        # Call alert callback if provided
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get current risk summary."""
        if not self._portfolio_metrics:
            return {"status": "no_data"}
        
        exposure = (
            float(self._portfolio_metrics.gross_exposure) / 
            float(self._portfolio_metrics.total_value)
        )
        
        leverage = float(self._portfolio_metrics.leverage)
        
        drawdown = (
            abs(self._portfolio_metrics.max_drawdown) 
            if self._portfolio_metrics.max_drawdown else 0
        )
        
        return {
            "status": "active" if self._is_monitoring else "inactive",
            "portfolio_exposure": {
                "current": exposure,
                "limit": float(self.risk_config.max_portfolio_exposure),
                "utilization": exposure / float(self.risk_config.max_portfolio_exposure)
            },
            "leverage": {
                "current": leverage,
                "limit": float(self.risk_config.leverage_limit),
                "utilization": leverage / float(self.risk_config.leverage_limit)
            },
            "drawdown": {
                "current": drawdown,
                "limit": float(self.risk_config.max_drawdown),
                "utilization": drawdown / float(self.risk_config.max_drawdown) if drawdown > 0 else 0
            },
            "active_violations": len(self._risk_violations),
            "violations": list(self._risk_violations),
            "position_count": len(self._positions),
            "last_check": datetime.now(timezone.utc).isoformat()
        }


class CircuitBreaker:
    """Trading circuit breaker for extreme risk events."""
    
    def __init__(
        self,
        daily_loss_limit: float = 0.05,  # 5% daily loss
        rapid_loss_limit: float = 0.02,  # 2% loss in 15 minutes
        rapid_loss_window: int = 900      # 15 minutes in seconds
    ):
        self.daily_loss_limit = daily_loss_limit
        self.rapid_loss_limit = rapid_loss_limit
        self.rapid_loss_window = rapid_loss_window
        
        self._is_tripped = False
        self._trip_reason = None
        self._trip_time = None
        self._pnl_history: List[tuple[datetime, float]] = []
        self._daily_start_value = None
        
        logger.info("Circuit breaker initialized")
    
    def update_portfolio_value(self, current_value: float, daily_pnl: float) -> None:
        """Update portfolio value for circuit breaker monitoring."""
        now = datetime.now(timezone.utc)
        
        # Track P&L history
        self._pnl_history.append((now, daily_pnl))
        
        # Clean old history
        cutoff = now - timedelta(seconds=self.rapid_loss_window)
        self._pnl_history = [
            (time, pnl) for time, pnl in self._pnl_history if time >= cutoff
        ]
        
        # Check for circuit breaker conditions
        self._check_daily_loss(daily_pnl)
        self._check_rapid_loss()
    
    def _check_daily_loss(self, daily_pnl: float) -> None:
        """Check daily loss limit."""
        if self._is_tripped:
            return
        
        daily_loss_pct = abs(daily_pnl) if daily_pnl < 0 else 0
        
        if daily_loss_pct > self.daily_loss_limit:
            self._trip_circuit_breaker(
                f"Daily loss limit exceeded: {daily_loss_pct:.2%} > {self.daily_loss_limit:.2%}"
            )
    
    def _check_rapid_loss(self) -> None:
        """Check rapid loss limit."""
        if self._is_tripped or len(self._pnl_history) < 2:
            return
        
        # Calculate loss over the window
        oldest_pnl = self._pnl_history[0][1]
        latest_pnl = self._pnl_history[-1][1]
        
        rapid_loss = oldest_pnl - latest_pnl if latest_pnl < oldest_pnl else 0
        
        if rapid_loss > self.rapid_loss_limit:
            self._trip_circuit_breaker(
                f"Rapid loss limit exceeded: {rapid_loss:.2%} in "
                f"{self.rapid_loss_window/60:.0f} minutes"
            )
    
    def _trip_circuit_breaker(self, reason: str) -> None:
        """Trip the circuit breaker."""
        self._is_tripped = True
        self._trip_reason = reason
        self._trip_time = datetime.now(timezone.utc)
        
        logger.critical(f"CIRCUIT BREAKER TRIPPED: {reason}")
    
    def reset_circuit_breaker(self, reason: str) -> None:
        """Reset the circuit breaker."""
        self._is_tripped = False
        self._trip_reason = None
        self._trip_time = None
        
        logger.warning(f"Circuit breaker reset: {reason}")
    
    def is_tripped(self) -> bool:
        """Check if circuit breaker is tripped."""
        return self._is_tripped
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "is_tripped": self._is_tripped,
            "trip_reason": self._trip_reason,
            "trip_time": self._trip_time.isoformat() if self._trip_time else None,
            "daily_loss_limit": self.daily_loss_limit,
            "rapid_loss_limit": self.rapid_loss_limit,
            "rapid_loss_window_minutes": self.rapid_loss_window / 60
        }