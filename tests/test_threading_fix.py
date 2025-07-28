"""
Test for the threading fix in timeout_protection and market_is_open functions.

This test ensures that the signal-based timeout works in the main thread
and gracefully falls back to no timeout in worker threads.
"""
import threading
import time
import os
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
import bot_engine


class TestThreadingFix:
    """Test suite for the threading fix in timeout_protection"""

    def test_timeout_protection_main_thread(self):
        """Test that timeout_protection works in main thread"""
        # Should work normally in main thread
        with bot_engine.timeout_protection(2):
            time.sleep(0.1)  # Short operation should complete

    def test_timeout_protection_main_thread_timeout(self):
        """Test that timeout_protection times out in main thread"""
        # Should timeout in main thread
        with pytest.raises(TimeoutError):
            with bot_engine.timeout_protection(1):
                time.sleep(2)

    def test_timeout_protection_worker_thread(self):
        """Test that timeout_protection works in worker thread without signal errors"""
        def worker_function():
            # Should not raise "signal only works in main thread" error
            with bot_engine.timeout_protection(1):
                time.sleep(0.1)  # Short operation
            return True

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(worker_function)
            result = future.result()
            assert result is True

    def test_market_is_open_from_worker_thread(self):
        """Test that market_is_open can be called from worker threads"""
        def worker_market_check():
            # This was the function that was failing before the fix
            return bot_engine.market_is_open()

        # Use FORCE_MARKET_OPEN to ensure consistent results
        with patch.dict(os.environ, {'FORCE_MARKET_OPEN': 'true'}):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(worker_market_check)
                result = future.result()
                assert result is True

    def test_market_is_open_main_thread(self):
        """Test that market_is_open still works in main thread"""
        with patch.dict(os.environ, {'FORCE_MARKET_OPEN': 'true'}):
            result = bot_engine.market_is_open()
            assert result is True

    def test_process_symbol_simulation(self):
        """Simulate the exact scenario that was causing the threading issue"""
        def mock_process_symbol(symbol):
            """Simulate the process_symbol function that calls is_market_open"""
            # This is the line that was causing the threading issue
            if not bot_engine.is_market_open():
                return f"MARKET_CLOSED_SKIP_SYMBOL | symbol={symbol}"
            return f"PROCESSED | symbol={symbol}"

        symbols = ["AAPL", "GOOGL", "MSFT"]
        
        # Test that all symbols can be processed in worker threads
        with patch.dict(os.environ, {'FORCE_MARKET_OPEN': 'true'}):
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(mock_process_symbol, sym) for sym in symbols]
                results = [f.result() for f in futures]
                
                # All should be processed successfully (no signal errors)
                for result in results:
                    assert "PROCESSED" in result
                    assert "MARKET_CLOSED_SKIP_SYMBOL" not in result

    def test_thread_detection(self):
        """Test that our thread detection logic works correctly"""
        # In main thread
        assert threading.current_thread() is threading.main_thread()
        
        # In worker thread
        def worker_check():
            return threading.current_thread() is threading.main_thread()
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(worker_check)
            is_main_in_worker = future.result()
            assert is_main_in_worker is False

    def test_timeout_protection_preserves_exceptions(self):
        """Test that timeout_protection doesn't swallow other exceptions"""
        class CustomError(Exception):
            pass

        # In main thread
        with pytest.raises(CustomError):
            with bot_engine.timeout_protection(5):
                raise CustomError("test error")

        # In worker thread
        def worker_with_error():
            with bot_engine.timeout_protection(5):
                raise CustomError("worker error")

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(worker_with_error)
            with pytest.raises(CustomError):
                future.result()

    def test_force_market_open_override_in_threads(self):
        """Test that FORCE_MARKET_OPEN override works in all threads"""
        def check_market_from_worker():
            return bot_engine.market_is_open()

        # Test with FORCE_MARKET_OPEN=true
        with patch.dict(os.environ, {'FORCE_MARKET_OPEN': 'true'}):
            # Main thread
            assert bot_engine.market_is_open() is True
            
            # Worker thread
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(check_market_from_worker)
                assert future.result() is True

        # Test with FORCE_MARKET_OPEN=false (default behavior)
        with patch.dict(os.environ, {'FORCE_MARKET_OPEN': 'false'}, clear=False):
            # Should work in both threads (may return True or False based on actual market hours)
            main_result = bot_engine.market_is_open()
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(check_market_from_worker)
                worker_result = future.result()
                
            # Both should complete without errors (results may vary based on actual time)
            assert isinstance(main_result, bool)
            assert isinstance(worker_result, bool)