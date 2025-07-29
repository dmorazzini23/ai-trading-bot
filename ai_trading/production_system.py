"""
Production trading system integration module.

Integrates all production-ready components into a unified trading system
with comprehensive risk management, monitoring, and execution capabilities.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging

# Use the centralized logger as per AGENTS.md
try:
    from logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from ..core.enums import RiskLevel, OrderSide, OrderType
from ..risk import (
    DynamicPositionSizer,
    TradingHaltManager,
    RiskManager
)
from ..monitoring import (
    AlertManager,
    PerformanceDashboard,
    AlertSeverity
)
from ..execution.production_engine import ProductionExecutionCoordinator
from ..execution.liquidity import LiquidityManager
from ..strategies import (
    MultiTimeframeAnalyzer,
    RegimeDetector
)


class ProductionTradingSystem:
    """
    Comprehensive production trading system.
    
    Integrates all production-ready components into a unified system
    with comprehensive safety, monitoring, and execution capabilities.
    """
    
    def __init__(self, account_equity: float, risk_level: RiskLevel = RiskLevel.MODERATE,
                 config: Dict = None):
        """Initialize production trading system."""
        # AI-AGENT-REF: Comprehensive production trading system integration
        self.account_equity = account_equity
        self.risk_level = risk_level
        self.config = config or {}
        
        # Initialize core components
        self.risk_manager = RiskManager(risk_level)
        self.halt_manager = TradingHaltManager()
        self.position_sizer = DynamicPositionSizer(risk_level)
        
        # Initialize monitoring and alerting
        self.alert_manager = AlertManager()
        self.performance_dashboard = PerformanceDashboard(self.alert_manager)
        
        # Initialize execution components
        self.execution_coordinator = ProductionExecutionCoordinator(account_equity, risk_level)
        self.liquidity_manager = LiquidityManager()
        
        # Initialize strategy components
        self.mtf_analyzer = MultiTimeframeAnalyzer()
        self.regime_detector = RegimeDetector()
        
        # System state
        self.is_active = False
        self.last_health_check = datetime.now()
        self.system_errors = []
        
        # Trading session state
        self.session_start_time = None
        self.session_trades = []
        self.session_pnl = 0.0
        
        logger.info(f"ProductionTradingSystem initialized with equity=${account_equity:,.2f}, "
                   f"risk_level={risk_level}")
    
    async def start_system(self) -> Dict[str, Any]:
        """Start the production trading system."""
        try:
            logger.info("Starting production trading system...")
            
            # Start monitoring components
            self.alert_manager.start_processing()
            
            # Initialize session
            self.session_start_time = datetime.now()
            self.is_active = True
            
            # Perform initial health check
            health_status = await self.perform_health_check()
            
            if not health_status["healthy"]:
                await self.alert_manager.send_system_alert(
                    "Trading System", "Startup Health Check Failed",
                    f"Issues: {', '.join(health_status['issues'])}",
                    AlertSeverity.CRITICAL
                )
                return {"status": "failed", "reason": "Health check failed"}
            
            # Send startup notification
            await self.alert_manager.send_system_alert(
                "Trading System", "System Started",
                f"Production trading system started successfully with ${self.account_equity:,.2f} equity",
                AlertSeverity.INFO
            )
            
            logger.info("Production trading system started successfully")
            
            return {
                "status": "success",
                "session_start_time": self.session_start_time,
                "account_equity": self.account_equity,
                "risk_level": self.risk_level.value,
                "health_status": health_status
            }
            
        except Exception as e:
            logger.error(f"Error starting production trading system: {e}")
            await self.alert_manager.send_system_alert(
                "Trading System", "Startup Failed",
                f"System startup error: {e}",
                AlertSeverity.EMERGENCY
            )
            return {"status": "error", "message": str(e)}
    
    async def stop_system(self, reason: str = "Manual shutdown") -> Dict[str, Any]:
        """Stop the production trading system."""
        try:
            logger.info(f"Stopping production trading system: {reason}")
            
            self.is_active = False
            
            # Get final session summary
            session_summary = await self.get_session_summary()
            
            # Stop monitoring components
            self.alert_manager.stop_processing()
            
            # Send shutdown notification
            await self.alert_manager.send_system_alert(
                "Trading System", "System Stopped",
                f"Production trading system stopped: {reason}",
                AlertSeverity.INFO
            )
            
            logger.info("Production trading system stopped successfully")
            
            return {
                "status": "success",
                "session_summary": session_summary,
                "shutdown_reason": reason,
                "shutdown_time": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error stopping production trading system: {e}")
            return {"status": "error", "message": str(e)}
    
    async def analyze_trading_opportunity(self, symbol: str, market_data: Dict) -> Dict[str, Any]:
        """Comprehensive analysis of a trading opportunity."""
        try:
            analysis_start = datetime.now()
            
            # Check if trading is allowed
            trading_status = self.halt_manager.is_trading_allowed()
            if not trading_status["trading_allowed"]:
                return {
                    "symbol": symbol,
                    "recommendation": "NO_TRADE",
                    "reason": f"Trading halted: {', '.join(trading_status['reasons'])}",
                    "timestamp": datetime.now()
                }
            
            # Perform regime detection
            regime_analysis = self.regime_detector.detect_regime(market_data.get("price_data", {}))
            
            # Perform multi-timeframe analysis
            mtf_analysis = self.mtf_analyzer.analyze_symbol(symbol, market_data.get("timeframe_data", {}))
            
            # Perform liquidity analysis
            liquidity_analysis = self.liquidity_manager.update_symbol_liquidity(
                symbol, market_data, market_data.get("current_price")
            )
            
            # Integrate analyses
            integrated_recommendation = await self._integrate_analyses(
                symbol, regime_analysis, mtf_analysis, liquidity_analysis
            )
            
            # Risk assessment
            risk_assessment = self.risk_manager.assess_trade_risk(
                symbol,
                integrated_recommendation.get("recommended_quantity", 0),
                market_data.get("current_price", 0),
                self.account_equity,
                []  # Position history would be retrieved here
            )
            
            # Final recommendation
            final_recommendation = await self._generate_final_recommendation(
                integrated_recommendation, risk_assessment, liquidity_analysis
            )
            
            analysis_time = (datetime.now() - analysis_start).total_seconds()
            
            return {
                "symbol": symbol,
                "timestamp": datetime.now(),
                "analysis_time_seconds": analysis_time,
                "regime_analysis": regime_analysis,
                "mtf_analysis": mtf_analysis,
                "liquidity_analysis": liquidity_analysis,
                "risk_assessment": risk_assessment,
                "final_recommendation": final_recommendation,
                "trading_allowed": trading_status["trading_allowed"]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trading opportunity for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}
    
    async def execute_trade(self, symbol: str, side: OrderSide, quantity: int,
                          order_type: OrderType = OrderType.MARKET,
                          price: Optional[float] = None,
                          market_data: Dict = None) -> Dict[str, Any]:
        """Execute a trade with comprehensive safety checks."""
        try:
            execution_start = datetime.now()
            
            # Perform pre-trade analysis
            if market_data:
                opportunity_analysis = await self.analyze_trading_opportunity(symbol, market_data)
                
                # Check if analysis recommends the trade
                final_rec = opportunity_analysis.get("final_recommendation", {})
                if final_rec.get("action") == "NO_TRADE":
                    return {
                        "status": "rejected",
                        "reason": "Analysis recommends no trade",
                        "analysis": opportunity_analysis
                    }
            
            # Execute through production coordinator
            execution_result = await self.execution_coordinator.submit_order(
                symbol, side, quantity, order_type, price, "production_system"
            )
            
            # Update performance tracking
            if execution_result["status"] == "success":
                await self._update_performance_tracking(execution_result)
                
                # Record session trade
                self.session_trades.append({
                    "timestamp": datetime.now(),
                    "symbol": symbol,
                    "side": side.value,
                    "quantity": execution_result.get("quantity", quantity),
                    "fill_price": execution_result.get("fill_price", price),
                    "execution_result": execution_result
                })
            
            execution_time = (datetime.now() - execution_start).total_seconds()
            execution_result["total_execution_time_seconds"] = execution_time
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            await self.alert_manager.send_trading_alert(
                "Trade Execution Error", symbol,
                {"error": str(e), "side": side.value, "quantity": quantity},
                AlertSeverity.CRITICAL
            )
            return {"status": "error", "message": str(e)}
    
    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        try:
            health_status = {
                "healthy": True,
                "timestamp": datetime.now(),
                "issues": [],
                "warnings": [],
                "component_status": {}
            }
            
            # Check trading halt manager
            trading_status = self.halt_manager.is_trading_allowed()
            health_status["component_status"]["halt_manager"] = trading_status
            if not trading_status["trading_allowed"]:
                health_status["warnings"].append("Trading currently halted")
            
            # Check alert manager
            alert_stats = self.alert_manager.get_alert_stats()
            health_status["component_status"]["alert_manager"] = {
                "processing_active": alert_stats.get("processing_active", False),
                "queue_size": alert_stats.get("queue_size", 0)
            }
            
            if not alert_stats.get("processing_active", False):
                health_status["issues"].append("Alert manager not processing")
                health_status["healthy"] = False
            
            # Check execution coordinator
            execution_stats = self.execution_coordinator.get_execution_summary()
            health_status["component_status"]["execution_coordinator"] = execution_stats
            
            if "error" in execution_stats:
                health_status["issues"].append("Execution coordinator error")
                health_status["healthy"] = False
            
            # Check account equity
            if self.account_equity <= 0:
                health_status["issues"].append("Invalid account equity")
                health_status["healthy"] = False
            
            # Update last health check time
            self.last_health_check = datetime.now()
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error performing health check: {e}")
            return {
                "healthy": False,
                "timestamp": datetime.now(),
                "issues": [f"Health check error: {e}"],
                "warnings": [],
                "component_status": {}
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            # Get component statuses
            trading_status = self.halt_manager.is_trading_allowed()
            execution_summary = self.execution_coordinator.get_execution_summary()
            portfolio_summary = self.performance_dashboard.get_dashboard_summary()
            liquidity_summary = self.liquidity_manager.get_portfolio_liquidity_summary()
            regime_summary = self.regime_detector.get_regime_summary()
            
            # Calculate uptime
            uptime_seconds = (datetime.now() - self.session_start_time).total_seconds() if self.session_start_time else 0
            
            return {
                "system_active": self.is_active,
                "session_start_time": self.session_start_time,
                "uptime_seconds": uptime_seconds,
                "account_equity": self.account_equity,
                "risk_level": self.risk_level.value,
                "trading_status": trading_status,
                "execution_summary": execution_summary,
                "portfolio_summary": portfolio_summary,
                "liquidity_summary": liquidity_summary,
                "regime_summary": regime_summary,
                "session_trades_count": len(self.session_trades),
                "last_health_check": self.last_health_check,
                "system_errors_count": len(self.system_errors)
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"error": str(e)}
    
    async def get_session_summary(self) -> Dict[str, Any]:
        """Get current trading session summary."""
        try:
            if not self.session_start_time:
                return {"error": "No active session"}
            
            session_duration = datetime.now() - self.session_start_time
            
            # Calculate session P&L
            total_trades = len(self.session_trades)
            successful_trades = len([t for t in self.session_trades 
                                   if t["execution_result"]["status"] == "success"])
            
            # Get portfolio performance
            portfolio_summary = self.performance_dashboard.get_dashboard_summary()
            
            return {
                "session_start_time": self.session_start_time,
                "session_duration_seconds": session_duration.total_seconds(),
                "total_trades": total_trades,
                "successful_trades": successful_trades,
                "success_rate": (successful_trades / total_trades * 100) if total_trades > 0 else 0,
                "session_pnl": self.session_pnl,
                "portfolio_performance": portfolio_summary.get("performance_metrics", {}),
                "account_equity": self.account_equity,
                "risk_level": self.risk_level.value
            }
            
        except Exception as e:
            logger.error(f"Error getting session summary: {e}")
            return {"error": str(e)}
    
    async def _integrate_analyses(self, symbol: str, regime_analysis: Dict, 
                                mtf_analysis: Dict, liquidity_analysis: Dict) -> Dict[str, Any]:
        """Integrate multiple analyses into unified recommendation."""
        try:
            integrated = {
                "symbol": symbol,
                "action": "HOLD",
                "confidence": 0.0,
                "recommended_quantity": 0,
                "reasoning": []
            }
            
            # Get regime recommendation
            regime_recommendations = self.regime_detector.get_regime_recommendations()
            regime_multiplier = regime_recommendations.get("position_size_multiplier", 1.0)
            
            # Get MTF recommendation
            mtf_recommendation = mtf_analysis.get("recommendation", {})
            mtf_action = mtf_recommendation.get("action", "HOLD")
            mtf_confidence = mtf_recommendation.get("confidence", 0.0)
            
            # Get liquidity constraints
            liquidity_level = liquidity_analysis.get("liquidity_level")
            execution_recs = liquidity_analysis.get("execution_recommendations", {})
            liquidity_multiplier = execution_recs.get("max_participation_rate", 0.1)
            
            # Integrate decisions
            if mtf_action in ["BUY", "WEAK_BUY"] and regime_multiplier > 0.5:
                integrated["action"] = "BUY"
                integrated["confidence"] = mtf_confidence * regime_multiplier
                integrated["reasoning"].append(f"MTF analysis suggests {mtf_action}")
                integrated["reasoning"].append(f"Regime allows {regime_multiplier:.0%} position size")
                
            elif mtf_action in ["SELL", "WEAK_SELL"] and regime_multiplier > 0.5:
                integrated["action"] = "SELL"
                integrated["confidence"] = mtf_confidence * regime_multiplier
                integrated["reasoning"].append(f"MTF analysis suggests {mtf_action}")
                
            else:
                integrated["action"] = "HOLD"
                integrated["confidence"] = 0.3
                integrated["reasoning"].append("No clear signal or regime restricts trading")
            
            # Apply liquidity constraints
            if liquidity_level and hasattr(liquidity_level, 'value') and liquidity_level.value in ["very_low", "low"]:
                integrated["confidence"] *= 0.5
                integrated["reasoning"].append("Reduced confidence due to low liquidity")
            
            return integrated
            
        except Exception as e:
            logger.error(f"Error integrating analyses: {e}")
            return {"symbol": symbol, "action": "HOLD", "confidence": 0.0}
    
    async def _generate_final_recommendation(self, integrated_rec: Dict, 
                                           risk_assessment: Dict, 
                                           liquidity_analysis: Dict) -> Dict[str, Any]:
        """Generate final trading recommendation."""
        try:
            final_rec = {
                "action": integrated_rec.get("action", "HOLD"),
                "confidence": integrated_rec.get("confidence", 0.0),
                "recommended_quantity": 0,
                "max_quantity": 0,
                "order_type": OrderType.MARKET,
                "execution_strategy": "standard",
                "warnings": [],
                "reasoning": integrated_rec.get("reasoning", [])
            }
            
            # Apply risk constraints
            if not risk_assessment.get("approved", False):
                final_rec["action"] = "NO_TRADE"
                final_rec["warnings"].extend(risk_assessment.get("warnings", []))
                final_rec["reasoning"].append("Trade rejected by risk assessment")
                return final_rec
            
            # Calculate position size
            recommended_size = risk_assessment.get("recommended_size", 0)
            final_rec["recommended_quantity"] = recommended_size
            final_rec["max_quantity"] = recommended_size
            
            # Apply liquidity constraints
            execution_recs = liquidity_analysis.get("execution_recommendations", {})
            final_rec["order_type"] = execution_recs.get("recommended_order_type", OrderType.MARKET)
            final_rec["execution_strategy"] = execution_recs.get("execution_strategy", "standard")
            
            # Add warnings from liquidity analysis
            liquidity_warnings = execution_recs.get("risk_warnings", [])
            final_rec["warnings"].extend(liquidity_warnings)
            
            return final_rec
            
        except Exception as e:
            logger.error(f"Error generating final recommendation: {e}")
            return {"action": "NO_TRADE", "warnings": [f"Recommendation error: {e}"]}
    
    async def _update_performance_tracking(self, execution_result: Dict):
        """Update performance tracking with execution result."""
        try:
            # Extract trade details
            symbol = execution_result.get("symbol", "")
            quantity = execution_result.get("quantity", 0)
            fill_price = execution_result.get("fill_price", 0)
            
            # Update position in performance dashboard
            self.performance_dashboard.update_position(symbol, quantity, fill_price, fill_price)
            
            # Update account equity (simplified - would need actual P&L calculation)
            # self.account_equity += trade_pnl
            # self.execution_coordinator.update_account_equity(self.account_equity)
            
        except Exception as e:
            logger.error(f"Error updating performance tracking: {e}")
    
    def update_account_equity(self, new_equity: float):
        """Update account equity across all components."""
        try:
            self.account_equity = new_equity
            self.execution_coordinator.update_account_equity(new_equity)
            self.halt_manager.update_equity(new_equity)
            logger.info(f"Account equity updated to ${new_equity:,.2f}")
        except Exception as e:
            logger.error(f"Error updating account equity: {e}")
    
    async def emergency_shutdown(self, reason: str = "Emergency"):
        """Emergency system shutdown with immediate halt."""
        try:
            logger.critical(f"EMERGENCY SHUTDOWN INITIATED: {reason}")
            
            # Activate emergency stop
            self.halt_manager.emergency_stop_all(reason)
            
            # Send emergency alert
            await self.alert_manager.send_system_alert(
                "Trading System", "EMERGENCY SHUTDOWN",
                f"Emergency shutdown activated: {reason}",
                AlertSeverity.EMERGENCY
            )
            
            # Stop system
            await self.stop_system(f"Emergency: {reason}")
            
        except Exception as e:
            logger.error(f"Error during emergency shutdown: {e}")
    
    def is_healthy(self) -> bool:
        """Quick health check for external monitoring."""
        try:
            return (self.is_active and 
                   self.halt_manager.is_trading_allowed()["trading_allowed"] and
                   self.account_equity > 0)
        except Exception:
            return False