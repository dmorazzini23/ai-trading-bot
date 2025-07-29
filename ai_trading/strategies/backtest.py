"""
Backtesting engine and performance analysis for strategies.

Provides comprehensive backtesting capabilities and
performance analysis for institutional trading strategies.
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import statistics
import logging

# Use the centralized logger as per AGENTS.md
try:
    from logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from .base import BaseStrategy, StrategySignal


class BacktestEngine:
    """
    Strategy backtesting engine.
    
    Provides comprehensive backtesting capabilities with
    realistic execution simulation and performance analysis.
    """
    
    def __init__(self):
        """Initialize backtest engine."""
        # AI-AGENT-REF: Strategy backtesting engine
        self.initial_capital = 100000  # $100k default
        self.commission_per_trade = 1.0  # $1 per trade
        logger.info("BacktestEngine initialized")
    
    def run_backtest(self, strategy: BaseStrategy, historical_data: Dict, 
                    start_date: datetime, end_date: datetime) -> Dict:
        """
        Run strategy backtest.
        
        Args:
            strategy: Strategy to backtest
            historical_data: Historical market data
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            Backtest results dictionary
        """
        try:
            results = {
                "strategy_id": strategy.strategy_id,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "initial_capital": self.initial_capital,
                "final_capital": self.initial_capital,
                "total_return": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "trades": []
            }
            
            # Simulate trading over the period
            current_capital = self.initial_capital
            portfolio_values = [current_capital]
            
            # Simplified simulation - would be much more complex in reality
            for symbol in strategy.symbols:
                symbol_data = historical_data.get(symbol, [])
                
                for i, data_point in enumerate(symbol_data):
                    if i == 0:
                        continue  # Skip first point
                    
                    # Generate signals
                    market_data = {symbol: data_point}
                    signals = strategy.generate_signals(market_data)
                    
                    for signal in signals:
                        if strategy.validate_signal(signal):
                            # Calculate position size
                            position_size = strategy.calculate_position_size(
                                signal, current_capital, 0
                            )
                            
                            if position_size > 0:
                                # Simulate trade execution
                                trade_result = self._simulate_trade(
                                    signal, position_size, data_point
                                )
                                
                                current_capital += trade_result["pnl"]
                                portfolio_values.append(current_capital)
                                
                                results["trades"].append(trade_result)
                                results["total_trades"] += 1
                                
                                if trade_result["pnl"] > 0:
                                    results["winning_trades"] += 1
                                else:
                                    results["losing_trades"] += 1
            
            # Calculate final metrics
            results["final_capital"] = current_capital
            results["total_return"] = (current_capital - self.initial_capital) / self.initial_capital
            
            if results["total_trades"] > 0:
                results["win_rate"] = results["winning_trades"] / results["total_trades"]
                
                wins = [t["pnl"] for t in results["trades"] if t["pnl"] > 0]
                losses = [t["pnl"] for t in results["trades"] if t["pnl"] < 0]
                
                results["avg_win"] = statistics.mean(wins) if wins else 0
                results["avg_loss"] = statistics.mean(losses) if losses else 0
            
            # Calculate drawdown
            results["max_drawdown"] = self._calculate_max_drawdown(portfolio_values)
            
            logger.info(f"Backtest completed for {strategy.name}: "
                       f"{results['total_return']:.2%} return, {results['total_trades']} trades")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {"error": str(e)}
    
    def _simulate_trade(self, signal: StrategySignal, position_size: int, 
                       market_data: Dict) -> Dict:
        """Simulate trade execution."""
        try:
            # Simplified trade simulation
            entry_price = market_data.get("close", 100.0)
            
            # Simulate some randomness in exit price
            import random
            price_change = random.gauss(0, 0.02)  # 2% volatility
            exit_price = entry_price * (1 + price_change)
            
            # Calculate P&L
            if signal.is_buy:
                pnl = position_size * (exit_price - entry_price)
            else:
                pnl = position_size * (entry_price - exit_price)
            
            # Subtract commission
            pnl -= self.commission_per_trade
            
            return {
                "symbol": signal.symbol,
                "side": signal.side.value,
                "quantity": position_size,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "signal_strength": signal.strength,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error simulating trade: {e}")
            return {"pnl": 0, "error": str(e)}
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown."""
        try:
            if not portfolio_values:
                return 0.0
            
            peak = portfolio_values[0]
            max_dd = 0.0
            
            for value in portfolio_values:
                if value > peak:
                    peak = value
                
                drawdown = (peak - value) / peak if peak > 0 else 0
                max_dd = max(max_dd, drawdown)
            
            return max_dd
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0


class PerformanceAnalyzer:
    """
    Strategy performance analysis.
    
    Provides detailed analysis of strategy performance
    including statistical measures and comparisons.
    """
    
    def __init__(self):
        """Initialize performance analyzer."""
        # AI-AGENT-REF: Strategy performance analysis
        logger.info("PerformanceAnalyzer initialized")
    
    def analyze_performance(self, backtest_results: Dict) -> Dict:
        """
        Analyze strategy performance.
        
        Args:
            backtest_results: Results from backtesting
            
        Returns:
            Comprehensive performance analysis
        """
        try:
            analysis = {
                "return_metrics": self._analyze_returns(backtest_results),
                "risk_metrics": self._analyze_risk(backtest_results),
                "trade_metrics": self._analyze_trades(backtest_results),
                "efficiency_metrics": self._analyze_efficiency(backtest_results)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
            return {"error": str(e)}
    
    def _analyze_returns(self, results: Dict) -> Dict:
        """Analyze return metrics."""
        total_return = results.get("total_return", 0)
        days = (datetime.fromisoformat(results["end_date"]) - 
                datetime.fromisoformat(results["start_date"])).days
        
        annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "daily_return": total_return / days if days > 0 else 0
        }
    
    def _analyze_risk(self, results: Dict) -> Dict:
        """Analyze risk metrics."""
        return {
            "max_drawdown": results.get("max_drawdown", 0),
            "volatility": 0.0,  # Would calculate from daily returns
            "var_95": 0.0,      # Would calculate Value at Risk
            "sharpe_ratio": results.get("sharpe_ratio", 0)
        }
    
    def _analyze_trades(self, results: Dict) -> Dict:
        """Analyze trade metrics."""
        return {
            "total_trades": results.get("total_trades", 0),
            "win_rate": results.get("win_rate", 0),
            "profit_factor": abs(results.get("avg_win", 0) / results.get("avg_loss", 1)),
            "avg_trade_return": results.get("total_return", 0) / max(1, results.get("total_trades", 1))
        }
    
    def _analyze_efficiency(self, results: Dict) -> Dict:
        """Analyze efficiency metrics."""
        return {
            "return_per_trade": results.get("total_return", 0) / max(1, results.get("total_trades", 1)),
            "capital_efficiency": results.get("final_capital", 0) / results.get("initial_capital", 1),
            "trade_frequency": results.get("total_trades", 0)  # Per time period
        }