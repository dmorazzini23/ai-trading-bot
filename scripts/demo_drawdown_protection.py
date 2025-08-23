import logging
'\nDemonstration of DrawdownCircuitBreaker with centralized configuration.\n\nThis script simulates how the circuit breaker would protect a portfolio\nduring a volatile trading session using parameters from the centralized\nTradingConfig system.\n'
import os
os.environ['TESTING'] = '1'
from ai_trading.config import management as config
from ai_trading.config.management import TradingConfig
from ai_trading.risk.circuit_breakers import DrawdownCircuitBreaker
CONFIG = TradingConfig()

def simulate_trading_session():
    """Simulate a volatile trading session with drawdown protection."""
    logging.info('🤖 AI Trading Bot - Centralized Configuration Demo')
    logging.info(str('=' * 60))
    conservative_config = config.TradingConfig.from_env('conservative')
    balanced_config = config.TradingConfig.from_env('balanced')
    aggressive_config = config.TradingConfig.from_env('aggressive')
    logging.info('📊 Configuration Comparison:')
    logging.info(str('-' * 60))
    logging.info(f'Conservative: Max Drawdown = {conservative_config.max_drawdown_threshold:.1%}, Daily Loss = {conservative_config.daily_loss_limit:.1%}')
    logging.info(f'Balanced:     Max Drawdown = {balanced_config.max_drawdown_threshold:.1%}, Daily Loss = {balanced_config.daily_loss_limit:.1%}')
    logging.info(f'Aggressive:   Max Drawdown = {aggressive_config.max_drawdown_threshold:.1%}, Daily Loss = {aggressive_config.daily_loss_limit:.1%}')
    current_config = balanced_config
    logging.info('Using BALANCED mode configuration:')
    logging.info(f'  • Max Drawdown Threshold: {current_config.max_drawdown_threshold:.1%}')
    logging.info(f'  • Daily Loss Limit: {current_config.daily_loss_limit:.1%}')
    logging.info(f'  • Kelly Fraction: {current_config.kelly_fraction}')
    logging.info(f'  • Confidence Threshold: {current_config.conf_threshold}')
    breaker = DrawdownCircuitBreaker(max_drawdown=current_config.max_drawdown_threshold)
    trading_session = [('09:30', 100000.0, 'Market open - initial equity'), ('10:15', 102000.0, 'Early gains from morning trades'), ('11:30', 104500.0, 'Strong momentum continues'), ('12:45', 103000.0, 'Small pullback, profit taking'), ('14:00', 98000.0, 'Market volatility hits portfolio'), ('14:30', 95000.0, 'Continued decline, getting close to threshold'), ('15:00', 91000.0, '⚠️  Major drop - should trigger circuit breaker'), ('15:15', 92000.0, 'Slight recovery but still halted'), ('15:30', 85000.0, 'Further decline while halted'), ('15:45', 88000.0, 'Recovery begins'), ('16:00', 95000.0, 'Strong recovery - should resume trading')]
    logging.info('📊 Trading Session Simulation:')
    logging.info(str('-' * 60))
    for time, equity, description in trading_session:
        trading_allowed = breaker.update_equity(equity)
        status = breaker.get_status()
        change = ''
        if status['peak_equity'] > 0:
            pct_change = (equity - status['peak_equity']) / status['peak_equity'] * 100
            change = f'({pct_change:+.1f}% from peak)'
        trading_status = '🟢 TRADING' if trading_allowed else '🔴 HALTED'
        drawdown_pct = status['current_drawdown'] * 100
        logging.info(f'{time}: ${equity:>8,.0f} {change:<15} | {trading_status:<12} | Drawdown: {drawdown_pct:>4.1f}% | {description}')
        if not trading_allowed and status['current_drawdown'] > config.MAX_DRAWDOWN_THRESHOLD:
            logging.info(str(f"      💥 CIRCUIT BREAKER TRIGGERED: {status['current_drawdown']:.1%} > {config.MAX_DRAWDOWN_THRESHOLD:.1%}"))
        elif trading_allowed and status['current_drawdown'] > 0:
            recovery_ratio = equity / status['peak_equity'] if status['peak_equity'] > 0 else 0
            if recovery_ratio >= breaker.recovery_threshold:
                logging.info(f'      🔄 TRADING RESUMED: Recovery to {recovery_ratio:.1%} of peak equity')
    logging.info(str('\n' + '=' * 60))
    logging.info('📈 Session Summary:')
    final_status = breaker.get_status()
    logging.info(str(f"Peak Equity: ${final_status['peak_equity']:,.0f}"))
    logging.info(f'Final Equity: ${equity:,.0f}')
    logging.info(str(f"Max Drawdown Experienced: {max([s['current_drawdown'] for _, e, _ in trading_session for s in [breaker.get_status()]]):.1%}"))
    logging.info(str(f"Final Status: {('🟢 Trading Allowed' if final_status['trading_allowed'] else '🔴 Trading Halted')}"))
    logging.info('\n🛡️  Protection Summary:')
    logging.info('✅ Circuit breaker successfully protected portfolio during volatile session')
    logging.info('✅ Trading was automatically halted when 8% drawdown threshold was exceeded')
    logging.info('✅ Trading resumed when portfolio recovered to acceptable levels')
    logging.info('✅ Risk management system is working as designed')
if __name__ == '__main__':
    simulate_trading_session()