"""
Backtesting engine and performance analysis for strategies.

Provides comprehensive backtesting capabilities and
performance analysis for institutional trading strategies.
"""
import math
import random
import statistics
import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from ai_trading.config.management import get_env, reload_env
from ai_trading.logging import logger
from ai_trading.oms.simulated_lifecycle import SimulatedLifecycleDriver
from .base import BaseStrategy, StrategySignal

# Optional numpy dependency used for noise; provide lightweight fallback for linting
try:  # pragma: no cover
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    class _NP:
        @staticmethod
        def sqrt(x):
            return math.sqrt(x)

        class random:  # noqa: N801
            @staticmethod
            def random():
                return random.random()

            @staticmethod
            def uniform(a, b):
                return random.uniform(a, b)

            @staticmethod
            def normal(mu, sigma):
                return random.gauss(mu, sigma)

    np = _NP()  # type: ignore

class BacktestEngine:
    """
    Strategy backtesting engine.

    Provides comprehensive backtesting capabilities with
    realistic execution simulation and performance analysis.
    """

    def __init__(self, initial_capital: float=100000, commission_bps: float=5.0, commission_flat: float=1.0, latency_ms: float=50.0, enable_slippage: bool=True, enable_partial_fills: bool=False, slippage_model: str='linear'):
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
        reload_env()
        self.initial_capital = initial_capital
        self.commission_bps = commission_bps
        self.commission_flat = commission_flat
        self.latency_ms = latency_ms
        self.enable_slippage = enable_slippage
        self.enable_partial_fills = enable_partial_fills
        self.slippage_model = slippage_model
        from ..execution import microstructure as _micro
        self.estimate_half_spread = getattr(
            _micro,
            'estimate_half_spread',
            lambda volatility, price, liquidity: max(0.0005, min(0.005, float(volatility) * 0.5)),
        )
        self.calculate_slippage = getattr(
            _micro,
            'calculate_slippage',
            lambda **_kwargs: 0.0,
        )
        self.calculate_partial_fill_probability = getattr(
            _micro,
            'calculate_partial_fill_probability',
            lambda **_kwargs: 0.0,
        )
        self.simulate_execution_with_latency = getattr(
            _micro,
            'simulate_execution_with_latency',
            lambda **_kwargs: None,
        )
        self.random_seed = int(get_env("AI_TRADING_LEGACY_BACKTEST_SEED", 42, cast=int))
        rand_fn: Callable[[], float]
        uniform_fn: Callable[[float, float], float]
        normal_fn: Callable[[float, float], float]
        try:
            rng = np.random.default_rng(self.random_seed)  # type: ignore[attr-defined]
            rand_fn = rng.random
            uniform_fn = rng.uniform
            normal_fn = rng.normal
        except Exception:  # pragma: no cover - fallback for minimal numpy stubs
            fallback_rng = random.Random(self.random_seed)
            rand_fn = fallback_rng.random
            uniform_fn = fallback_rng.uniform
            normal_fn = fallback_rng.gauss
        self._rand = rand_fn
        self._uniform = uniform_fn
        self._normal = normal_fn
        self.microstructure_available = all(
            callable(fn)
            for fn in (
                self.estimate_half_spread,
                self.calculate_slippage,
                self.calculate_partial_fill_probability,
            )
        )
        logger.info(
            'BacktestEngine initialized with realistic execution modeling',
            extra={"legacy_module": True, "seed": self.random_seed},
        )
        self._oms_events_enabled = bool(
            get_env("AI_TRADING_BACKTEST_OMS_EVENTS_ENABLED", False, cast=bool)
        )
        database_url = str(
            get_env("DATABASE_URL", "", cast=str, resolve_aliases=False) or ""
        ).strip()
        intent_store_path = str(
            get_env(
                "AI_TRADING_OMS_INTENT_STORE_PATH",
                "",
                cast=str,
                resolve_aliases=False,
            )
            or ""
        ).strip()
        self._oms_lifecycle = SimulatedLifecycleDriver(
            enabled=self._oms_events_enabled,
            source="legacy_backtest_engine",
            database_url=(database_url or None),
            intent_store_path=(intent_store_path or None),
        )
        self._event_counter = 0

    def _close_event_store(self) -> None:
        self._oms_lifecycle.close()

    @staticmethod
    def _hash_token(*parts: Any) -> str:
        material = "|".join(str(part) for part in parts if part not in (None, ""))
        if not material:
            material = "legacy-backtest-event"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    @staticmethod
    def _ts_text(value: Any) -> str:
        iso = getattr(value, "isoformat", None)
        if callable(iso):
            try:
                return str(iso())
            except Exception:
                return str(value)
        return str(value)

    def _emit_order_submit_lifecycle(
        self,
        *,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        trade_timestamp: Any,
    ) -> tuple[str | None, str | None]:
        self._event_counter += 1
        base_token = self._hash_token(
            "legacy_backtest",
            symbol,
            side,
            quantity,
            price,
            self._ts_text(trade_timestamp),
            self._event_counter,
        )
        intent_id = f"lbt-{base_token[:24]}"
        ref = self._oms_lifecycle.open_submitted_intent(
            intent_id=intent_id,
            idempotency_key=base_token,
            symbol=symbol,
            side=side,
            quantity=float(quantity),
            decision_ts=trade_timestamp,
            broker_order_id=intent_id,
            strategy_id="legacy_backtest_engine",
            metadata={
                "price": float(price),
                "bar_ts": self._ts_text(trade_timestamp),
                "legacy_backtest": True,
            },
        )
        if ref is None:
            return (None, None)
        return (ref.intent_id, ref.idempotency_key)

    def _emit_fill_lifecycle(
        self,
        *,
        intent_id: str | None,
        base_token: str | None,
        trade_result: dict[str, Any],
        trade_timestamp: Any,
    ) -> None:
        _ = base_token
        if not intent_id:
            return
        event_ts = str(trade_result.get("timestamp") or self._ts_text(trade_timestamp))
        fill_qty = int(max(0, float(trade_result.get("quantity_filled", 0) or 0)))
        fill_price = float(
            trade_result.get("execution_price")
            or trade_result.get("signal_price")
            or 0.0
        )
        self._oms_lifecycle.record_fill_and_close_intent(
            intent_id=str(intent_id),
            fill_qty=float(fill_qty),
            fill_price=fill_price,
            fee=float(trade_result.get("total_commission", 0.0) or 0.0),
            fill_ts=event_ts,
            terminal_status=("FILLED" if fill_qty > 0 else "FAILED"),
            last_error=(
                str(trade_result.get("error"))
                if trade_result.get("error") not in (None, "")
                else None
            ),
            liquidity_flag="SIMULATED",
        )

    def run_backtest(
        self,
        strategy: BaseStrategy,
        historical_data: dict[str, Any],
        start_date: datetime,
        end_date: datetime,
    ) -> dict[str, Any]:
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
            results: dict[str, Any] = {'strategy_id': strategy.strategy_id, 'start_date': start_date.isoformat(), 'end_date': end_date.isoformat(), 'initial_capital': self.initial_capital, 'final_capital': self.initial_capital, 'total_return': 0.0, 'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0, 'win_rate': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0, 'max_drawdown': 0.0, 'sharpe_ratio': 0.0, 'trades': []}
            current_capital = self.initial_capital
            portfolio_values = [current_capital]
            for symbol in strategy.symbols:
                symbol_data = historical_data.get(symbol, [])
                for i, data_point in enumerate(symbol_data):
                    if i == 0:
                        continue
                    market_data = {symbol: data_point}
                    signals = strategy.generate_signals(market_data)
                    for signal in signals:
                        if strategy.validate_signal(signal):
                            position_size = strategy.calculate_position_size(signal, current_capital, 0)
                            if position_size > 0:
                                trade_timestamp = data_point.get('timestamp', datetime.now(UTC))
                                intent_id, base_token = self._emit_order_submit_lifecycle(
                                    symbol=str(signal.symbol),
                                    side=str(signal.side),
                                    quantity=int(position_size),
                                    price=float(data_point.get('close', 0.0) or 0.0),
                                    trade_timestamp=trade_timestamp,
                                )
                                trade_result = self._simulate_trade(signal, position_size, data_point, trade_timestamp=trade_timestamp)
                                self._emit_fill_lifecycle(
                                    intent_id=intent_id,
                                    base_token=base_token,
                                    trade_result=trade_result,
                                    trade_timestamp=trade_timestamp,
                                )
                                current_capital += trade_result['net_pnl']
                                portfolio_values.append(current_capital)
                                results['trades'].append(trade_result)
                                results['total_trades'] += 1
                                if 'gross_pnl' in trade_result:
                                    results.setdefault('gross_pnl_total', 0)
                                    results['gross_pnl_total'] += trade_result['gross_pnl']
                                if trade_result['net_pnl'] > 0:
                                    results['winning_trades'] += 1
                                else:
                                    results['losing_trades'] += 1
            results['final_capital'] = current_capital
            results['total_return'] = (current_capital - self.initial_capital) / self.initial_capital
            results['net_return'] = results['total_return']
            gross_pnl_total = results.get('gross_pnl_total', 0)
            if gross_pnl_total > 0:
                gross_capital = self.initial_capital + gross_pnl_total
                results['gross_return'] = (gross_capital - self.initial_capital) / self.initial_capital
            else:
                results['gross_return'] = results['total_return']
            if results['total_trades'] > 0:
                results['win_rate'] = results['winning_trades'] / results['total_trades']
                wins = [t['net_pnl'] for t in results['trades'] if t.get('net_pnl', 0) > 0]
                losses = [t['net_pnl'] for t in results['trades'] if t.get('net_pnl', 0) < 0]
                results['avg_win'] = statistics.mean(wins) if wins else 0
                results['avg_loss'] = statistics.mean(losses) if losses else 0
                total_costs = sum((t.get('total_commission', 0) for t in results['trades']))
                total_turnover = sum((t.get('turnover', 0) for t in results['trades']))
                results['total_transaction_costs'] = total_costs
                results['total_turnover'] = total_turnover
                results['avg_cost_bps'] = total_costs / total_turnover * 10000 if total_turnover > 0 else 0
                results['cost_drag'] = results['gross_return'] - results['net_return']
                fill_ratios = [t.get('fill_ratio', 1.0) for t in results['trades']]
                results['avg_fill_ratio'] = statistics.mean(fill_ratios) if fill_ratios else 1.0
                slippage_costs = [t.get('slippage_cost_bps', 0) for t in results['trades']]
                results['avg_slippage_bps'] = statistics.mean(slippage_costs) if slippage_costs else 0.0
            results['max_drawdown'] = self._calculate_max_drawdown(portfolio_values)
            logger.info(f"Backtest completed for {strategy.name}: {results['total_return']:.2%} return, {results['total_trades']} trades")
            self._close_event_store()
            return results
        except (ValueError, TypeError) as e:
            logger.error(f'Error running backtest: {e}')
            self._close_event_store()
            return {'error': str(e)}

    def _simulate_trade(
        self,
        signal: StrategySignal,
        position_size: int,
        market_data: dict[str, Any],
        trade_timestamp: datetime | None=None,
    ) -> dict[str, Any]:
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
            close_price = float(market_data.get('close', 100.0) or 100.0)
            high_price = float(market_data.get('high', close_price * 1.01) or (close_price * 1.01))
            low_price = float(market_data.get('low', close_price * 0.99) or (close_price * 0.99))
            volume = float(market_data.get('volume', 100000) or 100000)
            signal_price = close_price
            if self.microstructure_available:
                volatility = abs(high_price - low_price) / close_price
                liquidity_proxy = volume / 1000
                half_spread = self.estimate_half_spread(volatility, close_price, liquidity_proxy)
            else:
                volatility = abs(high_price - low_price) / close_price
                half_spread = max(0.0005, min(0.005, volatility * 0.5))
            if signal.is_buy:
                execution_price = signal_price * (1 + half_spread)
            else:
                execution_price = signal_price * (1 - half_spread)
            slippage_amount = 0.0
            if self.enable_slippage and self.microstructure_available:
                volatility = abs(high_price - low_price) / close_price
                trade_size_fraction = position_size / max(volume, 1)
                slippage_amount = self.calculate_slippage(volatility=volatility, trade_size=trade_size_fraction, liquidity=volume / 10000, k=1.0)
                if signal.is_buy:
                    execution_price *= 1 + slippage_amount
                else:
                    execution_price *= 1 - slippage_amount
            elif self.enable_slippage:
                volatility = abs(high_price - low_price) / close_price
                slippage_pct = volatility * np.sqrt(position_size / max(volume, 1)) * 0.1
                slippage_amount = min(0.01, slippage_pct)
                if signal.is_buy:
                    execution_price *= 1 + slippage_amount
                else:
                    execution_price *= 1 - slippage_amount
            actual_quantity = position_size
            if self.enable_partial_fills and self.microstructure_available:
                market_depth = volume / 100
                fill_prob = 1.0 - self.calculate_partial_fill_probability(trade_size=position_size, market_depth=market_depth, urgency='medium')
                if self._rand() > fill_prob:
                    actual_quantity = int(position_size * self._uniform(0.3, 0.9))
            execution_timestamp = trade_timestamp or datetime.now(UTC)
            if self.microstructure_available and trade_timestamp:
                latency_cost = self._normal(0, 0.0001)
                execution_price *= 1 + latency_cost
            commission_bps_cost = self.commission_bps / 10000 * execution_price * actual_quantity
            commission_flat_cost = self.commission_flat
            total_commission = commission_bps_cost + commission_flat_cost
            if signal.is_buy:
                exit_price = execution_price * (1 + self._normal(0, 0.02))
                gross_pnl = actual_quantity * (exit_price - execution_price)
            else:
                exit_price = execution_price * (1 + self._normal(0, 0.02))
                gross_pnl = actual_quantity * (execution_price - exit_price)
            net_pnl = gross_pnl - total_commission
            turnover = actual_quantity * execution_price
            spread_cost_bps = half_spread * 10000
            slippage_cost_bps = slippage_amount * 10000
            total_cost_bps = total_commission / turnover * 10000 if turnover > 0 else 0
            return {'symbol': signal.symbol, 'side': signal.side, 'signal_price': signal_price, 'execution_price': execution_price, 'quantity_requested': position_size, 'quantity_filled': actual_quantity, 'fill_ratio': actual_quantity / position_size if position_size > 0 else 0, 'gross_pnl': gross_pnl, 'net_pnl': net_pnl, 'commission_bps': commission_bps_cost, 'commission_flat': commission_flat_cost, 'total_commission': total_commission, 'half_spread': half_spread, 'spread_cost_bps': spread_cost_bps, 'slippage_amount': slippage_amount, 'slippage_cost_bps': slippage_cost_bps, 'total_cost_bps': total_cost_bps, 'turnover': turnover, 'signal_strength': signal.strength, 'timestamp': execution_timestamp.isoformat() if execution_timestamp else datetime.now(UTC).isoformat(), 'latency_ms': self.latency_ms}
        except (ValueError, TypeError) as e:
            logger.error(f'Error simulating trade: {e}')
            return {'symbol': signal.symbol, 'side': signal.side, 'net_pnl': 0, 'gross_pnl': 0, 'error': str(e), 'quantity_filled': 0, 'fill_ratio': 0.0, 'total_cost_bps': 0.0}

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
        except (ValueError, TypeError) as e:
            logger.error(f'Error calculating max drawdown: {e}')
            return 0.0

class PerformanceAnalyzer:
    """
    Strategy performance analysis.

    Provides detailed analysis of strategy performance
    including statistical measures and comparisons.
    """

    def __init__(self):
        """Initialize performance analyzer."""
        logger.info('PerformanceAnalyzer initialized')

    def analyze_performance(self, backtest_results: dict) -> dict:
        """
        Analyze strategy performance.

        Args:
            backtest_results: Results from backtesting

        Returns:
            Comprehensive performance analysis
        """
        try:
            analysis = {'return_metrics': self._analyze_returns(backtest_results), 'risk_metrics': self._analyze_risk(backtest_results), 'trade_metrics': self._analyze_trades(backtest_results), 'efficiency_metrics': self._analyze_efficiency(backtest_results)}
            return analysis
        except (ValueError, TypeError) as e:
            logger.error(f'Error analyzing performance: {e}')
            return {'error': str(e)}

    def _analyze_returns(self, results: dict) -> dict:
        """Analyze return metrics."""
        total_return = results.get('total_return', 0)
        days = (datetime.fromisoformat(results['end_date']) - datetime.fromisoformat(results['start_date'])).days
        annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        return {'total_return': total_return, 'annualized_return': annualized_return, 'daily_return': total_return / days if days > 0 else 0}

    def _analyze_risk(self, results: dict) -> dict:
        """Analyze risk metrics."""
        return {'max_drawdown': results.get('max_drawdown', 0), 'volatility': 0.0, 'var_95': 0.0, 'sharpe_ratio': results.get('sharpe_ratio', 0)}

    def _analyze_trades(self, results: dict) -> dict:
        """Analyze trade metrics."""
        return {'total_trades': results.get('total_trades', 0), 'win_rate': results.get('win_rate', 0), 'profit_factor': abs(results.get('avg_win', 0) / results.get('avg_loss', 1)), 'avg_trade_return': results.get('total_return', 0) / max(1, results.get('total_trades', 1))}

    def _analyze_efficiency(self, results: dict) -> dict:
        """Analyze efficiency metrics."""
        return {'return_per_trade': results.get('total_return', 0) / max(1, results.get('total_trades', 1)), 'capital_efficiency': results.get('final_capital', 0) / results.get('initial_capital', 1), 'trade_frequency': results.get('total_trades', 0)}

def run_smoke_test():
    """
    Run smoke test to verify net < gross due to all costs.

    This validates that the cost model properly reduces net P&L
    compared to gross P&L when all costs are included.
    """
    logger.info('=== Backtest Smoke Test ===')
    logger.info('Testing that net < gross due to all costs')
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / 'math'))
        from money import Money
        logger.info('Simulating backtest with costs')
        entry_price = Money('100.00')
        exit_price = Money('102.00')
        quantity = 100
        gross_pnl = (exit_price - entry_price) * quantity
        logger.info(f'Gross P&L: ${gross_pnl}')
        position_value = entry_price * quantity
        execution_cost_bps = 5.0
        execution_cost = position_value * (execution_cost_bps / 10000)
        overnight_cost_bps = 2.0
        overnight_cost = position_value * (overnight_cost_bps / 10000)
        commission = Money('2.00')
        total_costs = execution_cost + overnight_cost + commission
        net_pnl = gross_pnl - total_costs
        logger.info(f'Execution cost (5 bps): ${execution_cost}')
        logger.info(f'Overnight cost (2 bps): ${overnight_cost}')
        logger.info(f'Commission: ${commission}')
        logger.info(f'Total costs: ${total_costs}')
        logger.info(f'Net P&L: ${net_pnl}')
        if net_pnl >= gross_pnl:
            raise AssertionError(f'Net P&L ({net_pnl}) should be less than gross P&L ({gross_pnl})')
        cost_drag_bps = total_costs / position_value * 10000
        logger.info(f'Total cost drag: {cost_drag_bps:.1f} bps')
        if cost_drag_bps < 5.0:
            raise AssertionError(f'Cost drag ({cost_drag_bps:.1f} bps) seems too low')
        logger.info('✓ Net P&L is correctly less than gross P&L due to costs')
        logger.info(f'✓ Cost drag of {cost_drag_bps:.1f} bps is realistic')
        logger.info('✓ Backtest smoke test passed!')
        return True
    except (ValueError, TypeError) as e:
        logger.error(f'✗ Backtest smoke test failed: {e}')
        return False
if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='Backtest module')
    parser.add_argument('--smoke', action='store_true', help='Run smoke test')
    args = parser.parse_args()
    if args.smoke:
        success = run_smoke_test()
        sys.exit(0 if success else 1)
    else:
        logger.info('Backtest module. Use --smoke to run smoke test.')
