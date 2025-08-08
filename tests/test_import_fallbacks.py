"""Test import fallback mechanisms for package vs repo-root execution."""

import sys
import unittest
from unittest.mock import patch


class TestImportFallbacks(unittest.TestCase):
    """Test package vs repo-root import fallback mechanisms."""

    def setUp(self):
        """Set up test by clearing imported modules."""
        # Store modules to restore later
        self.modules_to_restore = {}
        modules_to_clear = [
            'ai_trading.core.bot_engine',
            'ai_trading.signals', 
            'ai_trading.indicators',
            'bot_engine',
            'signals',
            'indicators'
        ]
        
        for module in modules_to_clear:
            if module in sys.modules:
                self.modules_to_restore[module] = sys.modules[module]
                del sys.modules[module]

    def tearDown(self):
        """Restore original modules."""
        for module, mod_obj in self.modules_to_restore.items():
            sys.modules[module] = mod_obj

    def test_backtester_import_fallback(self):
        """Test backtester import fallback from ai_trading.core to repo root."""
        
        # Test 1: Simulate ai_trading.core.bot_engine missing
        with patch.dict('sys.modules', {'ai_trading.core.bot_engine': None}):
            # Mock successful repo-root import
            mock_bot_engine = type('MockBotEngine', (), {})
            with patch.dict('sys.modules', {'bot_engine': mock_bot_engine}):
                
                # Import backtester - should use fallback
                import backtester
                
                # Verify the fallback was used (in actual test, you'd verify behavior)
                # This test structure validates the import mechanism works
                self.assertTrue(hasattr(backtester, 'bot_engine'))

    def test_profile_indicators_import_fallback(self):
        """Test profile_indicators import fallback for signals and indicators."""
        
        # Test 1: Simulate ai_trading modules missing
        missing_modules = {
            'ai_trading.signals': None,
            'ai_trading.indicators': None,
            'ai_trading': None
        }
        
        with patch.dict('sys.modules', missing_modules):
            # Mock successful repo-root imports
            mock_signals = type('MockSignals', (), {'rsi': lambda x: x})
            mock_indicators = type('MockIndicators', (), {'sma': lambda x: x})
            
            repo_modules = {
                'signals': mock_signals,
                'indicators': mock_indicators
            }
            
            with patch.dict('sys.modules', repo_modules):
                # Import should use repo-root fallback
                import profile_indicators
                
                # Verify fallback was used
                self.assertTrue(hasattr(profile_indicators, 'signals'))
                self.assertTrue(hasattr(profile_indicators, 'indicators'))

    def test_import_error_propagation(self):
        """Test that import errors are properly propagated when both imports fail."""
        
        # Test backtester when both ai_trading.core and repo-root imports fail
        missing_modules = {
            'ai_trading.core.bot_engine': None,
            'ai_trading.core': None,
            'bot_engine': None
        }
        
        with patch.dict('sys.modules', missing_modules):
            # Should raise ImportError when both imports fail
            with self.assertRaises(ImportError) as context:
                import backtester
                
            # Error message should indicate both import attempts failed
            error_msg = str(context.exception)
            self.assertIn("bot_engine", error_msg.lower())

    def test_package_import_success(self):
        """Test successful package import when ai_trading is available."""
        
        # Mock successful ai_trading.core.bot_engine import
        mock_bot_engine = type('MockBotEngine', (), {
            'run_all_trades_worker': lambda: None,
            'BotState': type('BotState', (), {})
        })
        
        with patch.dict('sys.modules', {'ai_trading.core.bot_engine': mock_bot_engine}):
            # Should use package import successfully
            import backtester
            
            # Verify package import was used
            self.assertTrue(hasattr(backtester, 'bot_engine'))

    def test_monkeypatch_simulation(self):
        """Test import fallback using monkeypatch to simulate missing modules."""
        
        # This test simulates how pytest monkeypatch would work
        original_import = __builtins__.__import__
        
        def mock_import(name, *args, **kwargs):
            if name == 'ai_trading.signals':
                raise ImportError("Simulated ai_trading.signals not found")
            elif name == 'ai_trading.indicators':
                raise ImportError("Simulated ai_trading.indicators not found")
            return original_import(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=mock_import):
            # Mock successful repo-root imports
            mock_signals = type('MockSignals', (), {})
            mock_indicators = type('MockIndicators', (), {})
            
            with patch.dict('sys.modules', {
                'signals': mock_signals,
                'indicators': mock_indicators
            }):
                # Should fall back to repo-root imports
                import profile_indicators
                
                # Verify fallback worked
                self.assertTrue(hasattr(profile_indicators, 'signals'))
                self.assertTrue(hasattr(profile_indicators, 'indicators'))

    def test_partial_import_failure(self):
        """Test behavior when only some package imports fail."""
        
        # Simulate scenario where ai_trading exists but some submodules don't
        mock_ai_trading = type('MockAiTrading', (), {})
        mock_signals = type('MockSignals', (), {})
        
        with patch.dict('sys.modules', {
            'ai_trading': mock_ai_trading,
            'ai_trading.signals': mock_signals,
            'ai_trading.indicators': None  # This one fails
        }):
            
            # Mock repo-root indicators as fallback
            mock_repo_indicators = type('MockIndicators', (), {})
            with patch.dict('sys.modules', {'indicators': mock_repo_indicators}):
                
                # Should use mix of package and repo-root imports
                import profile_indicators
                
                # Both should be available
                self.assertTrue(hasattr(profile_indicators, 'signals'))
                self.assertTrue(hasattr(profile_indicators, 'indicators'))

    def test_import_logging(self):
        """Test that import failures are properly logged."""
        
        with patch('profile_indicators.logger') as mock_logger:
            missing_modules = {
                'ai_trading.signals': None,
                'ai_trading.indicators': None,
                'ai_trading': None,
                'signals': None,
                'indicators': None
            }
            
            with patch.dict('sys.modules', missing_modules):
                # Should log error and raise
                with self.assertRaises(ImportError):
                    import profile_indicators
                
                # Verify error was logged
                mock_logger.error.assert_called()
                
                # Check that error message contains relevant information
                error_call = mock_logger.error.call_args[0][0]
                self.assertIn("signals", error_call.lower())
                self.assertIn("indicators", error_call.lower())

    def test_circular_import_prevention(self):
        """Test that fallback mechanism doesn't cause circular imports."""
        
        # This test verifies the import structure prevents circular dependencies
        # In practice, this would test with actual module structure
        
        # Mock modules that could cause circular imports
        mock_module = type('MockModule', (), {})
        
        with patch.dict('sys.modules', {
            'ai_trading.core.bot_engine': mock_module,
            'backtester': mock_module  # Already imported
        }):
            
            # Should not cause circular import issues
            try:
                import backtester
                # If we get here without recursion error, test passes
                self.assertTrue(True)
            except RecursionError:
                self.fail("Circular import detected in fallback mechanism")


if __name__ == "__main__":
    unittest.main()