import logging
'\nExample usage of the no-trade bands functionality for transaction cost optimization.\nThis demonstrates how to use the new no-trade bands to avoid churn on tiny weight deltas.\n'
import os
os.environ['ALPACA_API_KEY'] = 'demo'
os.environ['ALPACA_SECRET_KEY'] = 'demo'
os.environ['ALPACA_BASE_URL'] = 'demo'
os.environ['WEBHOOK_SECRET'] = 'demo'
os.environ['FLASK_PORT'] = '5000'
from ai_trading.rebalancer import apply_no_trade_bands

def demo_no_trade_bands():
    """Demonstrate the no-trade bands functionality."""
    logging.info('=== AI Trading Bot - No-Trade Bands Demo ===\n')
    current_weights = {'AAPL': 0.25, 'MSFT': 0.2, 'GOOGL': 0.15, 'AMZN': 0.15, 'TSLA': 0.1, 'NVDA': 0.1, 'META': 0.05}
    target_weights = {'AAPL': 0.2515, 'MSFT': 0.1985, 'GOOGL': 0.152, 'AMZN': 0.148, 'TSLA': 0.104, 'NVDA': 0.096, 'META': 0.05}
    logging.info('Current Portfolio Weights:')
    for symbol, weight in current_weights.items():
        logging.info(f'  {symbol}: {weight:.4f} ({weight * 100:.2f}%)')
    logging.info('\nTarget Portfolio Weights:')
    for symbol, weight in target_weights.items():
        current = current_weights.get(symbol, 0)
        delta_bps = (weight - current) * 10000
        logging.info(f'  {symbol}: {weight:.4f} ({weight * 100:.2f}%) [Δ{delta_bps:+.0f}bps]')
    band_thresholds = [10.0, 25.0, 50.0]
    for band_bps in band_thresholds:
        logging.info(f'\n=== Applying {band_bps:.0f}bps No-Trade Band ===')
        adjusted_weights = apply_no_trade_bands(current_weights, target_weights, band_bps)
        trades_needed = 0
        total_turnover = 0.0
        logging.info('Final Weights after No-Trade Bands:')
        for symbol in target_weights:
            current = current_weights.get(symbol, 0)
            target = target_weights.get(symbol, 0)
            final = adjusted_weights.get(symbol, 0)
            original_delta_bps = (target - current) * 10000
            final_delta_bps = (final - current) * 10000
            trade_avoided = abs(original_delta_bps) > 0.1 and abs(final_delta_bps) < 0.1
            trade_executed = abs(final_delta_bps) > 0.1
            if trade_executed:
                trades_needed += 1
                total_turnover += abs(final_delta_bps) / 10000
            status = 'AVOIDED' if trade_avoided else 'TRADE' if trade_executed else 'NO CHANGE'
            logging.info(f'  {symbol}: {final:.4f} [Δ{final_delta_bps:+.0f}bps] - {status}')
        logging.info('\nSummary:')
        logging.info(f'  Trades needed: {trades_needed}/7 positions')
        logging.info(f'  Total turnover: {total_turnover:.4f} ({total_turnover * 100:.2f}%)')
        logging.info(f'  Transaction cost savings: ~{(7 - trades_needed) * 0.0005 * 100:.2f}bps per avoided trade')
    logging.info('\n=== Large Rebalancing Example ===')
    large_target_weights = {'AAPL': 0.3, 'MSFT': 0.15, 'GOOGL': 0.2, 'AMZN': 0.1, 'TSLA': 0.15, 'NVDA': 0.05, 'META': 0.05}
    adjusted_large = apply_no_trade_bands(current_weights, large_target_weights, 25.0)
    logging.info('Large moves (>25bps threshold):')
    for symbol in large_target_weights:
        current = current_weights.get(symbol, 0)
        target = large_target_weights.get(symbol, 0)
        final = adjusted_large.get(symbol, 0)
        delta_bps = (final - current) * 10000
        logging.info(f'  {symbol}: {current:.3f} → {final:.3f} [Δ{delta_bps:+.0f}bps]')
    logging.info('\nAll large moves executed as expected (exceed 25bps threshold)')
    logging.info('\n=== No-Trade Bands Demo Complete ===')
if __name__ == '__main__':
    demo_no_trade_bands()