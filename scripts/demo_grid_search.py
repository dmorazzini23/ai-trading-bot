#!/usr/bin/env python3
import logging

"""
Example usage of the grid runner for backtesting and risk parameter optimization.
This demonstrates how to use the new parallel grid search functionality.
"""

import tempfile
from ai_trading.backtesting.grid_runner import grid_search, persist_artifacts


def evaluator(params):
    """
    Example evaluator function for backtesting.
    In a real implementation, this would run your backtest here and return metrics dict.
    
    Args:
        params: Dictionary of parameters to test
        
    Returns:
        Dictionary of performance metrics
    """
    # Mock calculation based on parameters
    kelly = params.get('kelly', 0.5)
    atr_mult = params.get('atr_mult', 2.0)
    lookback = params.get('lookback', 100)
    
    # Simulate different performance based on parameters
    base_sharpe = 1.2
    kelly_impact = (kelly - 0.5) * 0.8  # Kelly closer to 0.5 is better
    atr_impact = (2.0 - atr_mult) * 0.2  # ATR multiplier closer to 2.0 is better
    lookback_impact = (100 - lookback) * 0.001  # Lookback closer to 100 is better
    
    sharpe = base_sharpe + kelly_impact + atr_impact + lookback_impact
    
    return {
        "sharpe": round(sharpe, 3),
        "calmar": round(sharpe * 0.7, 3),
        "mdd": round(0.15 - sharpe * 0.02, 3),
        "total_return": round(sharpe * 0.12, 3),
        "volatility": round(0.18 - sharpe * 0.01, 3)
    }


def main():
    """Run example grid search for parameter optimization."""
    logging.info("=== AI Trading Bot - Grid Search Example ===\n")
    
    # Define parameter grid for optimization
    grid = {
        "kelly": [0.3, 0.6, 0.9],
        "atr_mult": [1.5, 2.0, 2.5],
        "lookback": [50, 100, 150],
    }
    
    logging.info(str(f"Testing {len(grid['kelly']) * len(grid['atr_mult']) * len(grid['lookback'])} parameter combinations..."))
    logging.info(f"Grid: {grid}\n")
    
    # Run parallel grid search
    logging.info("Running grid search with parallel processing...")
    run = grid_search(evaluator, grid, n_jobs=-1)
    logging.info(str(f"Completed {run['count']} backtests\n"))
    
    # Find best parameters
    results = run['results']
    best_result = max(results, key=lambda x: x['metrics']['sharpe'])
    worst_result = min(results, key=lambda x: x['metrics']['sharpe'])
    
    logging.info("=== Results Summary ===")
    logging.info(str(f"Best Parameters: {best_result['params']}"))
    logging.info(str(f"Best Metrics: {best_result['metrics']}"))
    logging.info(str(f"Best Sharpe: {best_result['metrics']['sharpe']}"))
    print()
    logging.info(str(f"Worst Parameters: {worst_result['params']}"))
    logging.info(str(f"Worst Sharpe: {worst_result['metrics']['sharpe']}"))
    print()
    
    # Show parameter impact analysis
    logging.info("=== Parameter Impact Analysis ===")
    kelly_impact = {}
    for result in results:
        kelly = result['params']['kelly']
        if kelly not in kelly_impact:
            kelly_impact[kelly] = []
        kelly_impact[kelly].append(result['metrics']['sharpe'])
    
    for kelly, sharpes in kelly_impact.items():
        avg_sharpe = sum(sharpes) / len(sharpes)
        logging.info(f"Kelly {kelly}: Average Sharpe = {avg_sharpe:.3f}")
    
    # Persist artifacts
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = persist_artifacts(run, tmp_dir)
        logging.info(f"\nArtifacts saved to: {out_dir}")
        logging.info("(Note: Using temporary directory for demo - in production, use persistent storage)")
    
    logging.info("\n=== Grid Search Complete ===")


if __name__ == "__main__":
    main()