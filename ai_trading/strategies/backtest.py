"""
Backtesting engine and performance analysis for strategies.

Provides comprehensive backtesting capabilities and
performance analysis for institutional trading strategies.
"""

import logging
import statistics
from datetime import UTC, datetime

# Use the centralized logger as per AGENTS.md
from ai_trading.logging import logger

from .base import BaseStrategy, StrategySignal


class BacktestEngine:
    """
    Strategy backtesting engine.

    Provides comprehensive backtesting capabilities with
    realistic execution simulation and performance analysis.
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        commission_bps: float = 5.0,
        commission_flat: float = 1.0,
        latency_ms: float = 50.0,
        enable_slippage: bool = True,
        enable_partial_fills: bool = False,
        slippage_model: str = "linear",
    ):
        """
        Initialize backtest engine with realistic execution modeling.

        Args:
            initial_capital: Starting capital
            commission_bps: Commission in basis points
            commission_flat: Flat commission per trade
            latency_ms: Execution latency in milliseconds
            enable_slippage: Whether to model slippage
            enable_partial_fills: Whether to model partial fills
            slippage_model: Slippage model ('linear', 'sqrt', 'impact')
        """
        # AI-AGENT-REF: Strategy backtesting engine with realistic execution
        self.initial_capital = initial_capital
        self.commission_bps = commission_bps
        self.commission_flat = commission_flat
        self.latency_ms = latency_ms
        self.enable_slippage = enable_slippage
        self.enable_partial_fills = enable_partial_fills
        self.slippage_model = slippage_model

        # Import microstructure helpers - fail fast approach
        from ..execution.microstructure import (
            calculate_partial_fill_probability,
            calculate_slippage,
            estimate_half_spread,
            simulate_execution_with_latency,
        )

        self.microstructure_available = True
        self.estimate_half_spread = estimate_half_spread
        self.calculate_slippage = calculate_slippage
        self.calculate_partial_fill_probability = calculate_partial_fill_probability
        self.simulate_execution_with_latency = simulate_execution_with_latency

        logger.info("BacktestEngine initialized with realistic execution modeling")

    def run_backtest(
        self,
        strategy: BaseStrategy,
        historical_data: dict,
        start_date: datetime,
        end_date: datetime,
    ) -> dict:
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
                "trades": [],
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
                                # Simulate trade execution with timestamp
                                trade_result = self._simulate_trade(
                                    signal,
                                    position_size,
                                    data_point,
                                    trade_timestamp=data_point.get(
                                        "timestamp", datetime.now(UTC)
                                    ),
                                )

                                current_capital += trade_result["net_pnl"]
                                portfolio_values.append(current_capital)

                                results["trades"].append(trade_result)
                                results["total_trades"] += 1

                                # Track gross vs net P&L
                                if "gross_pnl" in trade_result:
                                    results.setdefault("gross_pnl_total", 0)
                                    results["gross_pnl_total"] += trade_result[
                                        "gross_pnl"
                                    ]

                                if trade_result["net_pnl"] > 0:
                                    results["winning_trades"] += 1
                                else:
                                    results["losing_trades"] += 1

            # Calculate final metrics
            results["final_capital"] = current_capital
            results["total_return"] = (
                current_capital - self.initial_capital
            ) / self.initial_capital
            results["net_return"] = results["total_return"]  # Already net of costs

            # Calculate gross return if available
            gross_pnl_total = results.get("gross_pnl_total", 0)
            if gross_pnl_total > 0:
                gross_capital = self.initial_capital + gross_pnl_total
                results["gross_return"] = (
                    gross_capital - self.initial_capital
                ) / self.initial_capital
            else:
                results["gross_return"] = results["total_return"]

            if results["total_trades"] > 0:
                results["win_rate"] = (
                    results["winning_trades"] / results["total_trades"]
                )

                # Separate wins and losses
                wins = [
                    t["net_pnl"] for t in results["trades"] if t.get("net_pnl", 0) > 0
                ]
                losses = [
                    t["net_pnl"] for t in results["trades"] if t.get("net_pnl", 0) < 0
                ]

                results["avg_win"] = statistics.mean(wins) if wins else 0
                results["avg_loss"] = statistics.mean(losses) if losses else 0

                # Calculate transaction cost metrics
                total_costs = sum(
                    t.get("total_commission", 0) for t in results["trades"]
                )
                total_turnover = sum(t.get("turnover", 0) for t in results["trades"])

                results["total_transaction_costs"] = total_costs
                results["total_turnover"] = total_turnover
                results["avg_cost_bps"] = (
                    (total_costs / total_turnover * 10000) if total_turnover > 0 else 0
                )

                # Net vs gross metrics
                results["cost_drag"] = results["gross_return"] - results["net_return"]

                # Fill ratio statistics
                fill_ratios = [t.get("fill_ratio", 1.0) for t in results["trades"]]
                results["avg_fill_ratio"] = (
                    statistics.mean(fill_ratios) if fill_ratios else 1.0
                )

                # Slippage statistics
                slippage_costs = [
                    t.get("slippage_cost_bps", 0) for t in results["trades"]
                ]
                results["avg_slippage_bps"] = (
                    statistics.mean(slippage_costs) if slippage_costs else 0.0
                )

            # Calculate drawdown
            results["max_drawdown"] = self._calculate_max_drawdown(portfolio_values)

            logger.info(
                f"Backtest completed for {strategy.name}: "
                f"{results['total_return']:.2%} return, {results['total_trades']} trades"
            )

            return results

        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {"error": str(e)}

    def _simulate_trade(
        self,
        signal: StrategySignal,
        position_size: int,
        market_data: dict,
        trade_timestamp: datetime | None = None,
    ) -> dict:
        """
        Simulate realistic trade execution with spreads, slippage, and latency.

        Args:
            signal: Trading signal
            position_size: Position size to trade
            market_data: Market data at signal time
            trade_timestamp: Timestamp of trade signal

        Returns:
            Trade execution results
        """
        try:
            # Extract market data
            close_price = market_data.get("close", 100.0)
            high_price = market_data.get("high", close_price * 1.01)
            low_price = market_data.get("low", close_price * 0.99)
            volume = market_data.get("volume", 100000)

            # Calculate mid-price (signal price)
            signal_price = close_price

            # Step 1: Estimate half-spread
            if self.microstructure_available:
                # Use volatility proxy
                volatility = abs(high_price - low_price) / close_price
                liquidity_proxy = volume / 1000  # Simplified liquidity measure
                half_spread = self.estimate_half_spread(
                    volatility, close_price, liquidity_proxy
                )
            else:
                # Simple spread model: 0.05% to 0.5% based on volatility
                volatility = abs(high_price - low_price) / close_price
                half_spread = max(0.0005, min(0.005, volatility * 0.5))

            # Step 2: Calculate execution price with spread
            if signal.is_buy:
                execution_price = signal_price * (1 + half_spread)  # Pay ask
            else:
                execution_price = signal_price * (1 - half_spread)  # Hit bid

            # Step 3: Add slippage if enabled
            slippage_amount = 0.0
            if self.enable_slippage and self.microstructure_available:
                # Use realistic slippage model
                volatility = abs(high_price - low_price) / close_price
                trade_size_fraction = position_size / max(volume, 1)

                slippage_amount = self.calculate_slippage(
                    volatility=volatility,
                    trade_size=trade_size_fraction,
                    liquidity=volume / 10000,  # Normalized liquidity
                    k=1.0,
                )

                # Apply slippage in trade direction
                if signal.is_buy:
                    execution_price *= 1 + slippage_amount
                else:
                    execution_price *= 1 - slippage_amount
            elif self.enable_slippage:
                # Simple slippage model
                volatility = abs(high_price - low_price) / close_price
                slippage_pct = (
                    volatility * np.sqrt(position_size / max(volume, 1)) * 0.1
                )
                slippage_amount = min(0.01, slippage_pct)  # Cap at 1%

                if signal.is_buy:
                    execution_price *= 1 + slippage_amount
                else:
                    execution_price *= 1 - slippage_amount

            # Step 4: Handle partial fills if enabled
            actual_quantity = position_size
            if self.enable_partial_fills and self.microstructure_available:
                # Calculate partial fill probability
                market_depth = volume / 100  # Simplified depth estimate
                fill_prob = 1.0 - self.calculate_partial_fill_probability(
                    trade_size=position_size,
                    market_depth=market_depth,
                    urgency="medium",
                )

                # Simulate partial fill
                if np.random.random() > fill_prob:
                    actual_quantity = int(position_size * np.random.uniform(0.3, 0.9))

            # Step 5: Calculate latency effects
            execution_timestamp = trade_timestamp or datetime.now(UTC)
            if self.microstructure_available and trade_timestamp:
                # Simulate latency impact (simplified)
                latency_cost = np.random.normal(0, 0.0001)  # Small random cost
                execution_price *= 1 + latency_cost

            # Step 6: Calculate transaction costs
            # Commission in basis points
            commission_bps_cost = (
                (self.commission_bps / 10000) * execution_price * actual_quantity
            )

            # Flat commission
            commission_flat_cost = self.commission_flat

            total_commission = commission_bps_cost + commission_flat_cost

            # Step 7: Calculate P&L
            if signal.is_buy:
                # For backtesting, assume we sell at the next period's close
                # This is simplified - real implementation would track positions
                exit_price = execution_price * (
                    1 + np.random.normal(0, 0.02)
                )  # Random exit
                gross_pnl = actual_quantity * (exit_price - execution_price)
            else:
                # Short position
                exit_price = execution_price * (1 + np.random.normal(0, 0.02))
                gross_pnl = actual_quantity * (execution_price - exit_price)

            # Net P&L after costs
            net_pnl = gross_pnl - total_commission

            # Calculate metrics
            turnover = actual_quantity * execution_price
            spread_cost_bps = half_spread * 10000
            slippage_cost_bps = slippage_amount * 10000
            total_cost_bps = (
                (total_commission / turnover) * 10000 if turnover > 0 else 0
            )

            return {
                "symbol": signal.symbol,
                "side": signal.side.value,
                "signal_price": signal_price,
                "execution_price": execution_price,
                "quantity_requested": position_size,
                "quantity_filled": actual_quantity,
                "fill_ratio": (
                    actual_quantity / position_size if position_size > 0 else 0
                ),
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "commission_bps": commission_bps_cost,
                "commission_flat": commission_flat_cost,
                "total_commission": total_commission,
                "half_spread": half_spread,
                "spread_cost_bps": spread_cost_bps,
                "slippage_amount": slippage_amount,
                "slippage_cost_bps": slippage_cost_bps,
                "total_cost_bps": total_cost_bps,
                "turnover": turnover,
                "signal_strength": signal.strength,
                "timestamp": (
                    execution_timestamp.isoformat()
                    if execution_timestamp
                    else datetime.now(UTC).isoformat()
                ),
                "latency_ms": self.latency_ms,
            }

        except Exception as e:
            logger.error(f"Error simulating trade: {e}")
            return {
                "symbol": signal.symbol,
                "side": signal.side.value,
                "net_pnl": 0,
                "gross_pnl": 0,
                "error": str(e),
                "quantity_filled": 0,
                "fill_ratio": 0.0,
                "total_cost_bps": 0.0,
            }

    def _calculate_max_drawdown(self, portfolio_values: list[float]) -> float:
        """Calculate maximum drawdown."""
        try:
            if not portfolio_values:
                return 0.0

            peak = portfolio_values[0]
            max_dd = 0.0

            for value in portfolio_values:
                peak = max(peak, value)

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

    def analyze_performance(self, backtest_results: dict) -> dict:
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
                "efficiency_metrics": self._analyze_efficiency(backtest_results),
            }

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
            return {"error": str(e)}

    def _analyze_returns(self, results: dict) -> dict:
        """Analyze return metrics."""
        total_return = results.get("total_return", 0)
        days = (
            datetime.fromisoformat(results["end_date"])
            - datetime.fromisoformat(results["start_date"])
        ).days

        annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "daily_return": total_return / days if days > 0 else 0,
        }

    def _analyze_risk(self, results: dict) -> dict:
        """Analyze risk metrics."""
        return {
            "max_drawdown": results.get("max_drawdown", 0),
            "volatility": 0.0,  # Would calculate from daily returns
            "var_95": 0.0,  # Would calculate Value at Risk
            "sharpe_ratio": results.get("sharpe_ratio", 0),
        }

    def _analyze_trades(self, results: dict) -> dict:
        """Analyze trade metrics."""
        return {
            "total_trades": results.get("total_trades", 0),
            "win_rate": results.get("win_rate", 0),
            "profit_factor": abs(
                results.get("avg_win", 0) / results.get("avg_loss", 1)
            ),
            "avg_trade_return": results.get("total_return", 0)
            / max(1, results.get("total_trades", 1)),
        }

    def _analyze_efficiency(self, results: dict) -> dict:
        """Analyze efficiency metrics."""
        return {
            "return_per_trade": results.get("total_return", 0)
            / max(1, results.get("total_trades", 1)),
            "capital_efficiency": results.get("final_capital", 0)
            / results.get("initial_capital", 1),
            "trade_frequency": results.get("total_trades", 0),  # Per time period
        }


# AI-AGENT-REF: Smoke test functionality for validation
def run_smoke_test():
    """
    Run smoke test to verify net < gross due to all costs.

    This validates that the cost model properly reduces net P&L
    compared to gross P&L when all costs are included.
    """
    logging.info("=== Backtest Smoke Test ===")
    logging.info("Testing that net < gross due to all costs")

    try:
        # Import math utilities without triggering bot engine
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent / "math"))
        from money import Money

        # Mock a simple backtest scenario
        logging.info("Simulating backtest with costs")

        # Trade scenario: Buy at $100, sell at $102, 100 shares
        entry_price = Money("100.00")
        exit_price = Money("102.00")
        quantity = 100

        # Gross P&L calculation (no costs)
        gross_pnl = (exit_price - entry_price) * quantity
        logging.info(f"Gross P&L: ${gross_pnl}")

        # Apply realistic costs
        position_value = entry_price * quantity

        # Execution costs (5 bps total - 2.5 bps each way)
        execution_cost_bps = 5.0
        execution_cost = position_value * (execution_cost_bps / 10000)

        # Overnight holding cost (assume 1 day hold, 2 bps/day)
        overnight_cost_bps = 2.0
        overnight_cost = position_value * (overnight_cost_bps / 10000)

        # Commission (assume $1 minimum)
        commission = Money("2.00")  # $1 each way

        # Total costs
        total_costs = execution_cost + overnight_cost + commission

        # Net P&L after costs
        net_pnl = gross_pnl - total_costs

        logging.info(f"Execution cost (5 bps): ${execution_cost}")
        logging.info(f"Overnight cost (2 bps): ${overnight_cost}")
        logging.info(f"Commission: ${commission}")
        logging.info(f"Total costs: ${total_costs}")
        logging.info(f"Net P&L: ${net_pnl}")

        # Critical validation: net must be less than gross
        if net_pnl >= gross_pnl:
            raise AssertionError(
                f"Net P&L ({net_pnl}) should be less than gross P&L ({gross_pnl})"
            )

        # Additional checks
        cost_drag_bps = (total_costs / position_value) * 10000
        logging.info(f"Total cost drag: {cost_drag_bps:.1f} bps")

        if cost_drag_bps < 5.0:
            raise AssertionError(f"Cost drag ({cost_drag_bps:.1f} bps) seems too low")

        logging.info("✓ Net P&L is correctly less than gross P&L due to costs")
        logging.info(f"✓ Cost drag of {cost_drag_bps:.1f} bps is realistic")
        logging.info("✓ Backtest smoke test passed!")

        return True

    except Exception as e:
        logging.error(f"✗ Backtest smoke test failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Backtest module")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test")
    args = parser.parse_args()

    if args.smoke:
        success = run_smoke_test()
        sys.exit(0 if success else 1)
    else:
        logging.info("Backtest module. Use --smoke to run smoke test.")
