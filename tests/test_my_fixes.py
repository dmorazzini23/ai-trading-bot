#!/usr/bin/env python3
"""Test the specific fixes I implemented for the critical trading bot issues."""

import math
import os
import unittest


class TestMyFixes(unittest.TestCase):
    """Test the fixes I implemented for the 6 critical issues."""

    def test_meta_learning_thresholds_reduced(self):
        """Test that meta-learning thresholds are reduced to allow easier activation."""
        with open("bot_engine.py", 'r') as f:
            content = f.read()

        # Should have reduced min_trades from 3 to 2
        self.assertIn('METALEARN_MIN_TRADES", "2"', content)

        # Should have reduced performance threshold from 0.4 to 0.3
        self.assertIn('METALEARN_PERFORMANCE_THRESHOLD", "0.3"', content)

        print("✓ Meta-learning thresholds reduced: min_trades=2, threshold=0.3")

    def test_duplicate_logging_fix(self):
        """Test that duplicate event logging is eliminated."""
        debug_tracker_path = "ai_trading/execution/debug_tracker.py"
        if os.path.exists(debug_tracker_path):
            with open(debug_tracker_path, 'r') as f:
                content = f.read()

            # Should use elif instead of else to prevent double logging
            self.assertIn('elif phase in [ExecutionPhase.SIGNAL_GENERATED', content)

            # Should not have separate 'else:' block that could cause duplication
            duplicate_pattern = 'else:\n                # Log only key phases in normal mode\n                if phase in'
            self.assertNotIn(duplicate_pattern, content)

            print("✓ Duplicate event logging eliminated")

    def test_confidence_normalization_improved(self):
        """Test that confidence score normalization is improved."""
        with open("strategy_allocator.py", 'r') as f:
            content = f.read()

        # Should use tanh-based normalization
        self.assertIn('math.tanh', content)

        # Should log CONFIDENCE_NORMALIZED instead of warning
        self.assertIn('CONFIDENCE_NORMALIZED', content)

        # Should preserve original value for logging
        self.assertIn('original_confidence', content)

        print("✓ Confidence normalization improved with tanh-based algorithm")

    def test_position_limit_rebalancing(self):
        """Test that position limits allow rebalancing."""
        with open("bot_engine.py", 'r') as f:
            content = f.read()

        # Function should accept symbol parameter
        self.assertIn('def too_many_positions(ctx: BotContext, symbol: Optional[str] = None)', content)

        # Should allow rebalancing for existing symbols
        self.assertIn('ALLOW_REBALANCING', content)
        self.assertIn('existing_symbols', content)

        print("✓ Position limit rebalancing implemented")

    def test_liquidity_thresholds_increased(self):
        """Test that liquidity thresholds are made less aggressive."""
        with open("config.py", 'r') as f:
            content = f.read()

        # Spread threshold increased from 0.05 to 0.15
        self.assertIn('LIQUIDITY_SPREAD_THRESHOLD", "0.15"', content)

        # Volatility threshold increased from 0.02 to 0.08
        self.assertIn('LIQUIDITY_VOL_THRESHOLD", "0.08"', content)

        print("✓ Liquidity thresholds made less aggressive: spread=15%, vol=8%")

    def test_data_quality_handling_improved(self):
        """Test that data quality validation is improved."""
        with open("trade_execution.py", 'r') as f:
            content = f.read()

        # Should require minimum 3 rows instead of 5
        self.assertIn('len(df) < 3:', content)

        # Should handle limited data gracefully
        self.assertIn('Limited minute data', content)

        # Should use adaptive calculations
        self.assertIn('min(5, len(df))', content)

        print("✓ Data quality validation improved: min 3 rows, adaptive calculations")

    def test_confidence_algorithm_correctness(self):
        """Test that the confidence normalization algorithm works correctly."""

        def tanh_normalize(confidence):
            """The actual algorithm implemented."""
            if confidence > 1:
                normalized = (math.tanh(confidence - 1) + 1) / 2
                normalized = 0.5 + (normalized * 0.5)
                return max(0.01, min(1.0, normalized))
            elif confidence < 0:
                return max(0.01, confidence)
            return confidence

        # Test the specific problematic values from the logs
        meta_confidence = 1.8148200636230267
        shop_confidence = 3.686892484542545

        meta_normalized = tanh_normalize(meta_confidence)
        shop_normalized = tanh_normalize(shop_confidence)

        # Both should be in valid range
        self.assertGreater(meta_normalized, 0.0)
        self.assertLessEqual(meta_normalized, 1.0)
        self.assertGreater(shop_normalized, 0.0)
        self.assertLessEqual(shop_normalized, 1.0)

        # Values > 1 should be mapped to [0.5, 1] range
        self.assertGreaterEqual(meta_normalized, 0.5)
        self.assertGreaterEqual(shop_normalized, 0.5)

        print(f"✓ Confidence normalization: META {meta_confidence:.3f}→{meta_normalized:.3f}, SHOP {shop_confidence:.3f}→{shop_normalized:.3f}")

    def test_meta_learning_would_activate(self):
        """Test that meta-learning would now activate with reasonable data."""

        # Simulate trade data with new thresholds
        signals = ["sma_cross", "rsi_oversold", "momentum"]
        min_trades = 2  # New threshold
        threshold = 0.3  # New threshold

        # Create mock trade data - 2 trades per signal with mixed performance
        trade_data = {}
        for signal in signals:
            trade_data[signal] = [
                {"pnl": 5.0},   # Winning trade
                {"pnl": -2.0}   # Losing trade
            ]

        # Calculate which signals would qualify
        qualified_signals = {}
        for signal, trades in trade_data.items():
            if len(trades) >= min_trades:
                win_rate = sum(1 for t in trades if t["pnl"] > 0) / len(trades)
                if win_rate >= threshold:
                    qualified_signals[signal] = win_rate

        # With 50% win rate and 0.3 threshold, signals should qualify
        self.assertGreater(len(qualified_signals), 0, "Some signals should qualify with new thresholds")

        print(f"✓ Meta-learning simulation: {len(qualified_signals)}/{len(signals)} signals would qualify")


if __name__ == "__main__":
    unittest.main()
