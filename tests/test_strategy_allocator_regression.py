"""
Regression tests for StrategyAllocator signal confirmation issues.

This test module contains specific tests to prevent regression of the signal
confirmation bug that was causing test_allocator to fail with empty results
on the second call when min_confidence=0.0.
"""

from strategies import TradeSignal
import strategy_allocator


class TestStrategyAllocatorRegression:
    """Regression tests for previously fixed signal confirmation issues."""
    
    def test_signal_confirmation_with_zero_min_confidence(self):
        """
        Regression test for the original failing scenario.
        
        The original issue was that with min_confidence=0.0, the second call
        to allocate() was returning an empty list instead of confirmed signals.
        """
        alloc = strategy_allocator.StrategyAllocator()
        
        # Set exact configuration from original failing test
        alloc.config.delta_threshold = 0.0
        alloc.config.signal_confirmation_bars = 2
        alloc.config.min_confidence = 0.0
        
        sig = TradeSignal(symbol="AAPL", side="buy", confidence=1.0, strategy="s1")
        
        # First call: Build signal history (should return empty list)
        out1 = alloc.allocate({"s1": [sig]})
        assert out1 == [], "First call should return empty list (unconfirmed signals)"
        
        # Second call: Confirm signals (should return confirmed signal)
        out2 = alloc.allocate({"s1": [sig]})
        assert out2 and out2[0].symbol == "AAPL", "Second call should return confirmed AAPL signal"
        assert len(out2) == 1, "Should return exactly one signal"
        assert out2[0].confidence == 1.0, "Signal confidence should be preserved"
    
    def test_config_missing_min_confidence_attribute(self):
        """
        Regression test for missing min_confidence attribute in config.
        
        Previously, if min_confidence was missing from config, it could cause
        AttributeError or incorrect behavior.
        """
        alloc = strategy_allocator.StrategyAllocator()
        
        # Remove min_confidence attribute to simulate missing config
        if hasattr(alloc.config, 'min_confidence'):
            delattr(alloc.config, 'min_confidence')
        
        alloc.config.delta_threshold = 0.0
        alloc.config.signal_confirmation_bars = 2
        
        sig = TradeSignal(symbol="AAPL", side="buy", confidence=1.0, strategy="s1")
        
        # Should not raise exception and should use default threshold
        out1 = alloc.allocate({"s1": [sig]})
        out2 = alloc.allocate({"s1": [sig]})
        
        # With default min_confidence (0.6) and signal confidence (1.0), should confirm
        assert len(out2) == 1, "Should confirm signal with default threshold"
        assert out2[0].symbol == "AAPL", "Should return AAPL signal"
    
    def test_config_none_min_confidence(self):
        """
        Regression test for None min_confidence value.
        
        Previously, if min_confidence was set to None, it could cause
        comparison errors or unexpected behavior.
        """
        alloc = strategy_allocator.StrategyAllocator()
        
        alloc.config.min_confidence = None
        alloc.config.delta_threshold = 0.0
        alloc.config.signal_confirmation_bars = 2
        
        sig = TradeSignal(symbol="AAPL", side="buy", confidence=1.0, strategy="s1")
        
        # Should not raise exception and should use default threshold
        out1 = alloc.allocate({"s1": [sig]})
        out2 = alloc.allocate({"s1": [sig]})
        
        # With default min_confidence (0.6) and signal confidence (1.0), should confirm
        assert len(out2) == 1, "Should confirm signal with default threshold"
        assert out2[0].symbol == "AAPL", "Should return AAPL signal"
    
    def test_signal_confirmation_boundary_conditions(self):
        """
        Test signal confirmation at various boundary conditions.
        
        Ensures that the confirmation logic works correctly at edge cases
        that could have caused the original failure.
        """
        test_cases = [
            # (min_confidence, signal_confidence, should_confirm)
            (0.0, 0.0, True),    # Zero threshold, zero confidence
            (0.0, 1.0, True),    # Zero threshold, high confidence
            (0.6, 0.6, True),    # Exact threshold match
            (0.8, 0.5, False),   # Below threshold
            (0.5, 0.8, True),    # Above threshold
        ]
        
        for min_conf, sig_conf, should_confirm in test_cases:
            alloc = strategy_allocator.StrategyAllocator()
            alloc.config.delta_threshold = 0.0
            alloc.config.signal_confirmation_bars = 2
            alloc.config.min_confidence = min_conf
            
            sig = TradeSignal(symbol="AAPL", side="buy", confidence=sig_conf, strategy="s1")
            
            out1 = alloc.allocate({"s1": [sig]})
            out2 = alloc.allocate({"s1": [sig]})
            
            if should_confirm:
                assert len(out2) == 1, f"Should confirm with min_conf={min_conf}, sig_conf={sig_conf}"
                assert out2[0].symbol == "AAPL", "Should return AAPL signal"
            else:
                assert len(out2) == 0, f"Should NOT confirm with min_conf={min_conf}, sig_conf={sig_conf}"
    
    def test_invalid_signal_confidence_handling(self):
        """
        Test handling of invalid signal confidence values.
        
        Ensures that out-of-range confidence values are properly normalized
        and don't cause the confirmation logic to fail.
        """
        alloc = strategy_allocator.StrategyAllocator()
        alloc.config.delta_threshold = 0.0
        alloc.config.signal_confirmation_bars = 2
        alloc.config.min_confidence = 0.0
        
        # Test high confidence (> 1.0)
        sig_high = TradeSignal(symbol="AAPL", side="buy", confidence=2.0, strategy="s1")
        
        out1 = alloc.allocate({"s1": [sig_high]})
        out2 = alloc.allocate({"s1": [sig_high]})
        
        # Should handle gracefully and still confirm
        assert len(out2) == 1, "Should handle high confidence gracefully"
        assert out2[0].symbol == "AAPL", "Should return AAPL signal"
        
        # Test negative confidence
        alloc_fresh = strategy_allocator.StrategyAllocator()
        alloc_fresh.config.delta_threshold = 0.0
        alloc_fresh.config.signal_confirmation_bars = 2
        alloc_fresh.config.min_confidence = 0.0
        
        sig_neg = TradeSignal(symbol="AAPL", side="buy", confidence=-0.5, strategy="s1")
        
        out1 = alloc_fresh.allocate({"s1": [sig_neg]})
        out2 = alloc_fresh.allocate({"s1": [sig_neg]})
        
        # Should handle gracefully (confidence normalized to 0, still >= 0.0 threshold)
        assert len(out2) == 1, "Should handle negative confidence gracefully"
        assert out2[0].symbol == "AAPL", "Should return AAPL signal"
    
    def test_multiple_instances_no_shared_state(self):
        """
        Test that multiple allocator instances don't share state.
        
        Ensures that the signal confirmation works consistently across
        different allocator instances.
        """
        for i in range(3):
            alloc = strategy_allocator.StrategyAllocator()
            alloc.config.delta_threshold = 0.0
            alloc.config.signal_confirmation_bars = 2
            alloc.config.min_confidence = 0.0
            
            sig = TradeSignal(symbol="AAPL", side="buy", confidence=1.0, strategy="s1")
            
            out1 = alloc.allocate({"s1": [sig]})
            out2 = alloc.allocate({"s1": [sig]})
            
            assert out1 == [], f"Instance {i}: First call should return empty list"
            assert out2 and out2[0].symbol == "AAPL", f"Instance {i}: Second call should return AAPL signal"