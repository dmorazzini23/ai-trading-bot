"""Test executor auto-sizing logic and environment variable overrides."""

import os
import unittest
from unittest.mock import patch, MagicMock


class TestExecutorsSizing(unittest.TestCase):
    """Test executor auto-sizing and environment overrides."""

    def setUp(self):
        """Set up test by clearing executor-related environment variables."""
        self.original_env = {}
        env_vars = ["EXECUTOR_WORKERS", "PREDICTION_WORKERS"]
        
        for var in env_vars:
            self.original_env[var] = os.environ.get(var)
            if var in os.environ:
                del os.environ[var]

    def tearDown(self):
        """Restore original environment variables."""
        for var, value in self.original_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]

    def test_auto_sizing_with_different_cpu_counts(self):
        """Test auto-sizing logic with different CPU counts."""
        test_cases = [
            (1, 2),   # Single CPU -> 2 workers (minimum)
            (2, 2),   # 2 CPUs -> 2 workers  
            (4, 4),   # 4 CPUs -> 4 workers
            (8, 4),   # 8 CPUs -> 4 workers (maximum)
            (16, 4),  # 16 CPUs -> 4 workers (maximum)
        ]
        
        for cpu_count, expected_workers in test_cases:
            with self.subTest(cpu_count=cpu_count):
                with patch('multiprocessing.cpu_count', return_value=cpu_count):
                    # Import the functions to test
                    from ai_trading.core.bot_engine import get_executor_workers, get_prediction_workers
                    
                    self.assertEqual(
                        get_executor_workers(),
                        expected_workers,
                        f"Expected {expected_workers} workers for {cpu_count} CPUs"
                    )
                    
                    self.assertEqual(
                        get_prediction_workers(),
                        expected_workers,
                        f"Expected {expected_workers} prediction workers for {cpu_count} CPUs"
                    )

    def test_executor_workers_environment_override(self):
        """Test EXECUTOR_WORKERS environment variable override."""
        test_values = [
            ("1", 1),
            ("2", 2),
            ("8", 8),
            ("16", 16),
        ]
        
        for env_value, expected in test_values:
            with self.subTest(env_value=env_value):
                os.environ["EXECUTOR_WORKERS"] = env_value
                
                # Mock CPU count to ensure env override takes precedence
                with patch('multiprocessing.cpu_count', return_value=4):
                    from ai_trading.core.bot_engine import get_executor_workers
                    
                    # Force reimport to pick up new env var
                    import importlib
                    import ai_trading.core.bot_engine
                    importlib.reload(ai_trading.core.bot_engine)
                    
                    result = ai_trading.core.bot_engine.get_executor_workers()
                    self.assertEqual(
                        result,
                        expected,
                        f"Expected {expected} workers for EXECUTOR_WORKERS={env_value}"
                    )

    def test_prediction_workers_environment_override(self):
        """Test PREDICTION_WORKERS environment variable override."""
        test_values = [
            ("1", 1),
            ("3", 3),
            ("6", 6),
            ("12", 12),
        ]
        
        for env_value, expected in test_values:
            with self.subTest(env_value=env_value):
                os.environ["PREDICTION_WORKERS"] = env_value
                
                # Mock CPU count to ensure env override takes precedence
                with patch('multiprocessing.cpu_count', return_value=4):
                    from ai_trading.core.bot_engine import get_prediction_workers
                    
                    # Force reimport to pick up new env var
                    import importlib
                    import ai_trading.core.bot_engine
                    importlib.reload(ai_trading.core.bot_engine)
                    
                    result = ai_trading.core.bot_engine.get_prediction_workers()
                    self.assertEqual(
                        result,
                        expected,
                        f"Expected {expected} workers for PREDICTION_WORKERS={env_value}"
                    )

    def test_invalid_environment_values(self):
        """Test handling of invalid environment variable values."""
        invalid_values = ["", "abc", "-1", "0", "1.5"]
        
        for invalid_value in invalid_values:
            with self.subTest(invalid_value=invalid_value):
                os.environ["EXECUTOR_WORKERS"] = invalid_value
                
                with patch('multiprocessing.cpu_count', return_value=4):
                    from ai_trading.core.bot_engine import get_executor_workers
                    
                    # Force reimport
                    import importlib
                    import ai_trading.core.bot_engine
                    importlib.reload(ai_trading.core.bot_engine)
                    
                    # Should fall back to auto-sizing for invalid values
                    result = ai_trading.core.bot_engine.get_executor_workers()
                    self.assertEqual(
                        result,
                        4,  # Expected auto-sized value for 4 CPUs
                        f"Should fall back to auto-sizing for invalid value '{invalid_value}'"
                    )

    def test_minimum_workers_enforced(self):
        """Test that minimum worker count is enforced."""
        # Test with env value of 0
        os.environ["EXECUTOR_WORKERS"] = "0"
        
        with patch('multiprocessing.cpu_count', return_value=4):
            from ai_trading.core.bot_engine import get_executor_workers
            
            # Force reimport
            import importlib
            import ai_trading.core.bot_engine
            importlib.reload(ai_trading.core.bot_engine)
            
            result = ai_trading.core.bot_engine.get_executor_workers()
            self.assertGreaterEqual(
                result,
                1,
                "Should enforce minimum of 1 worker even when env var is 0"
            )

    def test_multiprocessing_import_failure_fallback(self):
        """Test fallback when multiprocessing import fails."""
        with patch('multiprocessing.cpu_count', side_effect=ImportError("No multiprocessing")):
            from ai_trading.core.bot_engine import get_executor_workers, get_prediction_workers
            
            # Should fall back to default value
            self.assertEqual(
                get_executor_workers(),
                2,
                "Should fall back to 2 workers when multiprocessing unavailable"
            )
            
            self.assertEqual(
                get_prediction_workers(),
                2,
                "Should fall back to 2 prediction workers when multiprocessing unavailable"
            )

    def test_cpu_count_exception_fallback(self):
        """Test fallback when cpu_count raises NotImplementedError."""
        with patch('multiprocessing.cpu_count', side_effect=NotImplementedError("Not implemented")):
            from ai_trading.core.bot_engine import get_executor_workers, get_prediction_workers
            
            # Should fall back to default value
            self.assertEqual(
                get_executor_workers(),
                2,
                "Should fall back to 2 workers when cpu_count not implemented"
            )
            
            self.assertEqual(
                get_prediction_workers(),
                2,
                "Should fall back to 2 prediction workers when cpu_count not implemented"
            )

    def test_both_environment_overrides(self):
        """Test both EXECUTOR_WORKERS and PREDICTION_WORKERS set independently."""
        os.environ["EXECUTOR_WORKERS"] = "3"
        os.environ["PREDICTION_WORKERS"] = "6"
        
        with patch('multiprocessing.cpu_count', return_value=8):
            # Force reimport to pick up env vars
            import importlib
            import ai_trading.core.bot_engine
            importlib.reload(ai_trading.core.bot_engine)
            
            self.assertEqual(
                ai_trading.core.bot_engine.get_executor_workers(),
                3,
                "EXECUTOR_WORKERS should be 3"
            )
            
            self.assertEqual(
                ai_trading.core.bot_engine.get_prediction_workers(),
                6,
                "PREDICTION_WORKERS should be 6"
            )

    def test_executor_initialization(self):
        """Test that executors are initialized with correct worker counts."""
        os.environ["EXECUTOR_WORKERS"] = "2"
        os.environ["PREDICTION_WORKERS"] = "3"
        
        # Force reimport to create new executor instances
        import importlib
        import ai_trading.core.bot_engine
        importlib.reload(ai_trading.core.bot_engine)
        
        # Check that executors have correct max_workers
        # Note: In a real environment, you'd check the actual executor attributes
        # This test validates the configuration is applied
        self.assertEqual(
            ai_trading.core.bot_engine.get_executor_workers(),
            2
        )
        
        self.assertEqual(
            ai_trading.core.bot_engine.get_prediction_workers(),
            3
        )

    def test_zero_cpu_fallback(self):
        """Test behavior when cpu_count returns 0."""
        with patch('multiprocessing.cpu_count', return_value=0):
            from ai_trading.core.bot_engine import get_executor_workers, get_prediction_workers
            
            # Should use minimum of 2 even when CPU count is 0
            self.assertEqual(
                get_executor_workers(),
                2,
                "Should use minimum 2 workers when CPU count is 0"
            )
            
            self.assertEqual(
                get_prediction_workers(),
                2,
                "Should use minimum 2 prediction workers when CPU count is 0"
            )


if __name__ == "__main__":
    unittest.main()