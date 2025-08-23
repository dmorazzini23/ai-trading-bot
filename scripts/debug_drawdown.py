import logging
'Debug the drawdown circuit breaker status variable issue.'
import os
import sys
import traceback
os.environ['TESTING'] = '1'

def test_drawdown_circuit_breaker():
    """Test drawdown circuit breaker for the status variable issue."""
    try:
        logging.info('🔍 Testing DrawdownCircuitBreaker...')
        from ai_trading.risk.circuit_breakers import DrawdownCircuitBreaker
        breaker = DrawdownCircuitBreaker(max_drawdown=0.08)
        logging.info('✅ Circuit breaker initialized successfully')
        logging.info('\n📊 Testing equity updates...')
        result1 = breaker.update_equity(100000)
        logging.info(f'Initial equity update: {result1}')
        logging.info('\n🔍 Testing get_status method...')
        status = breaker.get_status()
        logging.info(f'Status keys: {list(status.keys())}')
        logging.info(f'Full status: {status}')
        logging.info('\n⚠️ Testing drawdown scenario...')
        result2 = breaker.update_equity(95000)
        logging.info(f'5% loss result: {result2}')
        result3 = breaker.update_equity(90000)
        logging.info(f'10% loss result: {result3}')
        final_status = breaker.get_status()
        logging.info(f'Final status: {final_status}')
        logging.info('✅ All tests completed successfully')
        return True
    except (KeyError, ValueError, TypeError) as e:
        logging.info(f'❌ Error in drawdown test: {e}')
        logging.info(f'Traceback: {traceback.format_exc()}')
        return False

def test_status_variable_issue():
    """Specifically test for the status variable issue."""
    try:
        logging.info('\n🔍 Testing for status variable issue...')
        from ai_trading.risk.circuit_breakers import DrawdownCircuitBreaker
        breaker = DrawdownCircuitBreaker(max_drawdown=0.08)
        breaker.update_equity(88519.46)
        for i in range(5):
            try:
                equity = 88519.46 * (1 - 0.01 * i)
                result = breaker.update_equity(equity)
                status = breaker.get_status()
                logging.info(str(f"Update {i + 1}: equity=${equity:.2f}, result={result}, state={status['state']}"))
            except (KeyError, ValueError, TypeError) as e:
                logging.info(f'❌ Error on update {i + 1}: {e}')
                logging.info(f'Traceback: {traceback.format_exc()}')
                return False
        logging.info('✅ Status variable test completed')
        return True
    except (KeyError, ValueError, TypeError) as e:
        logging.info(f'❌ Error in status variable test: {e}')
        logging.info(f'Traceback: {traceback.format_exc()}')
        return False
if __name__ == '__main__':
    logging.info('🚀 Starting DrawdownCircuitBreaker Debug Session')
    logging.info(str('=' * 60))
    success1 = test_drawdown_circuit_breaker()
    success2 = test_status_variable_issue()
    logging.info(str('\n' + '=' * 60))
    if success1 and success2:
        logging.info('✅ All tests passed - Circuit breaker appears functional')
    else:
        logging.info('❌ Some tests failed - Issues detected')
        sys.exit(1)