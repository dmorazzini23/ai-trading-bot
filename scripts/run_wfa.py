"""
Walk-forward validation runner script.

Runs walk-forward analysis on the active trading universe using
the enhanced cost-aware strategy logic to validate changes before live deployment.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    class _PD:  # minimal placeholder
        @staticmethod
        def concat(objs):
            return list(objs)
    pd = _PD()  # type: ignore
try:
    from ai_trading.config.management import TradingConfig
    from ai_trading.data_fetcher import DataFetcher
    from ai_trading.evaluation.walkforward import WalkForwardEvaluator
    from ai_trading.logging import logger
    from ai_trading.signals import SignalDecisionPipeline, generate_cost_aware_signals
except ImportError:
    sys.exit(1)

def create_cost_aware_strategy(config: TradingConfig):
    """Create a strategy function that uses the new cost-aware signal logic."""

    def strategy_func(train_data: dict, test_data: dict) -> list:
        """
        Cost-aware strategy implementation for walk-forward validation.
        
        Args:
            train_data: Dictionary of {symbol: pd.DataFrame} for training period
            test_data: Dictionary of {symbol: pd.DataFrame} for testing period
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        pipeline_config = {'min_edge_threshold': 0.002, 'transaction_cost_buffer': 0.001, 'ensemble_min_agree': 2, 'ensemble_total': 3, 'atr_stop_multiplier': 2.0, 'atr_target_multiplier': 3.0, 'regime_volatility_threshold': 0.025}
        decision_pipeline = SignalDecisionPipeline(pipeline_config)
        for symbol, test_df in test_data.items():
            if symbol not in train_data:
                continue
            train_df = train_data[symbol]
            for i in range(1, len(test_df)):
                try:
                    historical_data = pd.concat([train_df.tail(100), test_df.iloc[:i]])
                    if len(historical_data) < 50:
                        continue
                    try:
                        returns = historical_data['close'].pct_change().dropna()
                        momentum = returns.tail(5).mean()
                        volatility = returns.tail(20).std()
                        predicted_edge = momentum * 0.5
                        if volatility > 0:
                            predicted_edge = predicted_edge / volatility
                        predicted_edge = max(-0.05, min(0.05, predicted_edge))
                    except (ValueError, TypeError):
                        predicted_edge = 0.0
                    decision = decision_pipeline.evaluate_signal_with_costs(symbol, historical_data, predicted_edge, quantity=1000)
                    if decision.get('decision') == 'ACCEPT':
                        entry_price = test_df['close'].iloc[i]
                        exit_idx = min(i + 5, len(test_df) - 1)
                        exit_price = test_df['close'].iloc[exit_idx]
                        stop_loss = decision.get('stop_loss', entry_price * 0.98)
                        take_profit = decision.get('take_profit', entry_price * 1.02)
                        for j in range(i + 1, exit_idx + 1):
                            day_low = test_df['low'].iloc[j] if 'low' in test_df.columns else test_df['close'].iloc[j]
                            day_high = test_df['high'].iloc[j] if 'high' in test_df.columns else test_df['close'].iloc[j]
                            if day_low <= stop_loss:
                                exit_price = stop_loss
                                break
                            elif day_high >= take_profit:
                                exit_price = take_profit
                                break
                        signal = 1 if predicted_edge > 0 else -1
                        predictions.append({'symbol': symbol, 'signal': signal, 'entry_price': entry_price, 'exit_price': exit_price, 'predicted_edge': predicted_edge, 'decision_reason': decision.get('reason', 'UNKNOWN'), 'timestamp': test_df.index[i] if hasattr(test_df.index[i], 'strftime') else str(test_df.index[i])})
                except (ValueError, TypeError) as e:
                    logger.debug('Signal generation failed for %s at %d: %s', symbol, i, e)
                    continue
        return predictions
    return strategy_func

def run_walkforward_validation(symbols: list, config: TradingConfig) -> dict:
    """Run walk-forward validation for the given symbols."""
    try:
        logger.info('Starting walk-forward validation for %d symbols', len(symbols))
        data_fetcher = DataFetcher()

        def data_provider(symbol: str, start_date: datetime, end_date: datetime):
            try:
                return data_fetcher.get_historical_data(symbol, start_date, end_date)
            except (ValueError, TypeError) as e:
                logger.warning('Failed to get data for %s: %s', symbol, e)
                return None
        strategy_func = create_cost_aware_strategy(config)
        wf_config = {'mode': 'rolling', 'train_span': 252, 'test_span': 63, 'step_size': 21, 'embargo_pct': 0.01, 'artifacts_dir': 'artifacts/wfa', 'enable_plots': False}
        evaluator = WalkForwardEvaluator(**wf_config)
        results = evaluator.run_walkforward(symbols=symbols, strategy_func=strategy_func, data_provider=data_provider)
        return results
    except (ValueError, TypeError) as e:
        logger.error('Walk-forward validation failed: %s', e)
        raise

def main():
    """Main entry point for WFA runner."""
    parser = argparse.ArgumentParser(description='Run walk-forward validation')
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols (default: from config)')
    parser.add_argument('--universe-file', type=str, help='Path to file containing symbols list')
    parser.add_argument('--dry-run', action='store_true', help='Validate setup without running')
    args = parser.parse_args()
    try:
        config = TradingConfig.from_env()
        if args.symbols:
            symbols = [s.strip() for s in args.symbols.split(',')]
        elif args.universe_file:
            universe_file = Path(args.universe_file)
            if not universe_file.exists():
                logger.error('Universe file not found: %s', universe_file)
                sys.exit(1)
            with open(universe_file) as f:
                symbols = [line.strip() for line in f if line.strip() and (not line.startswith('#'))]
        else:
            symbols = getattr(config, 'default_universe', ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 'NFLX', 'DIS'])
        if not symbols:
            logger.error('No symbols specified for validation')
            sys.exit(1)
        logger.info('Walk-forward validation universe: %s', symbols)
        if args.dry_run:
            logger.info('Dry run mode - validation setup looks good')
            return
        results = run_walkforward_validation(symbols, config)
        if results and 'performance_summary' in results:
            perf = results['performance_summary']
            if 'sharpe_ratio' in perf:
                pass
            if 'hit_rate' in perf:
                pass
            if 'max_drawdown' in perf:
                pass
            results.get('performance_grade', 'N/A')
            results.get('validation_summary', {})
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info('Walk-forward validation interrupted by user')
        sys.exit(0)
    except (ValueError, TypeError) as e:
        logger.error('Walk-forward validation error: %s', e, exc_info=True)
        sys.exit(1)
if __name__ == '__main__':
    main()