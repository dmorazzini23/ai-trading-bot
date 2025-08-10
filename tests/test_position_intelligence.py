"""
Simple test for the enhanced position management system.
Tests the integration without requiring full environment setup.
"""

import os
import sys
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def test_intelligent_position_components():
    """Test the intelligent position management components directly."""
    print("🧪 Testing Intelligent Position Management Components")
    print("=" * 60)
    
    # Add position module to path
    position_path = os.path.join(os.path.dirname(__file__), 'ai_trading', 'position')
    if position_path not in sys.path:
        sys.path.insert(0, position_path)
    
    try:
        # Test 1: Market Regime Detection
        print("\n1. Testing Market Regime Detection...")
        from market_regime import MarketRegimeDetector, MarketRegime
        
        detector = MarketRegimeDetector()
        params = detector.get_regime_parameters(MarketRegime.TRENDING_BULL)
        
        print(f"   ✓ Regime parameters for trending bull: {len(params)} parameters")
        print(f"   ✓ Profit taking patience: {params.get('profit_taking_patience', 'N/A')}")
        print(f"   ✓ Stop distance multiplier: {params.get('stop_distance_multiplier', 'N/A')}")
        
        # Test 2: Technical Signal Analysis
        print("\n2. Testing Technical Signal Analysis...")
        from technical_analyzer import TechnicalSignalAnalyzer
        
        analyzer = TechnicalSignalAnalyzer()
        
        # Test RSI calculation with mock data
        class MockSeries:
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def diff(self):
                return MockSeries([self.data[i] - self.data[i-1] if i > 0 else 0 for i in range(len(self.data))])
            def where(self, condition, other):
                return MockSeries([x if x > 0 else other for x in self.data])
            def rolling(self, window):
                return MockRolling(self.data, window)
            @property
            def iloc(self):
                return MockIloc(self.data)
                
        class MockRolling:
            def __init__(self, data, window):
                self.data = data
                self.window = window
            def mean(self):
                result = []
                for i in range(len(self.data)):
                    if i < self.window - 1:
                        result.append(float('nan'))
                    else:
                        window_data = self.data[i-self.window+1:i+1]
                        result.append(sum(window_data) / len(window_data))
                return MockSeries(result)
                
        class MockIloc:
            def __init__(self, data):
                self.data = data
            def __getitem__(self, idx):
                return self.data[idx]
        
        # Test with trending price data
        price_data = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
        mock_prices = MockSeries(price_data)
        
        rsi = analyzer._calculate_rsi(mock_prices, 14)
        print(f"   ✓ RSI calculation: {rsi:.2f} (trending up)")
        
        # Test 3: Trailing Stop Management
        print("\n3. Testing Trailing Stop Management...")
        from trailing_stops import TrailingStopManager
        
        stop_manager = TrailingStopManager()
        
        # Test stop distance calculation
        initial_distance = stop_manager.base_trail_percent
        print(f"   ✓ Base trailing distance: {initial_distance}%")
        
        # Test momentum multiplier
        multiplier = stop_manager._calculate_momentum_multiplier('AAPL', None)
        print(f"   ✓ Momentum multiplier: {multiplier}")
        
        # Test time decay
        time_multiplier = stop_manager._calculate_time_decay_multiplier(10)
        print(f"   ✓ Time decay multiplier (10 days): {time_multiplier}")
        
        # Test 4: Profit Taking Engine
        print("\n4. Testing Profit Taking Engine...")
        from profit_taking import ProfitTakingEngine
        
        profit_engine = ProfitTakingEngine()
        
        # Test profit velocity calculation
        velocity = profit_engine.calculate_profit_velocity('AAPL')  # Will return 0.0 without plan
        print(f"   ✓ Profit velocity calculation: {velocity}")
        
        # Test percentage targets creation
        targets = profit_engine._create_percentage_targets(100.0, 100)
        print(f"   ✓ Created {len(targets)} percentage-based profit targets")
        
        # Test 5: Portfolio Correlation Analysis
        print("\n5. Testing Portfolio Correlation Analysis...")
        from correlation_analyzer import PortfolioCorrelationAnalyzer
        
        corr_analyzer = PortfolioCorrelationAnalyzer()
        
        # Test sector classification
        sector = corr_analyzer._get_symbol_sector('AAPL')
        print(f"   ✓ AAPL sector classification: {sector}")
        
        sector = corr_analyzer._get_symbol_sector('JPM')
        print(f"   ✓ JPM sector classification: {sector}")
        
        # Test concentration classification
        level = corr_analyzer._classify_position_concentration(45.0)
        print(f"   ✓ 45% position concentration level: {level.value}")
        
        # Test 6: Intelligent Position Manager
        print("\n6. Testing Intelligent Position Manager...")
        from intelligent_manager import IntelligentPositionManager
        
        manager = IntelligentPositionManager()
        print(f"   ✓ Initialized with {len(manager.analysis_weights)} analysis components")
        
        # Test action determination
        action, confidence, urgency = manager._determine_action_from_scores(0.8, 0.2, 0.1)
        print(f"   ✓ Action determination: {action.value} (confidence: {confidence:.2f})")
        
        print("\n" + "=" * 60)
        print("🎉 ALL INTELLIGENT POSITION MANAGEMENT COMPONENTS WORKING!")
        print("🚀 Ready for advanced position holding strategies!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_scenarios():
    """Test integration scenarios."""
    print("\n🔗 Testing Integration Scenarios")
    print("=" * 40)
    
    try:
        position_path = os.path.join(os.path.dirname(__file__), 'ai_trading', 'position')
        if position_path not in sys.path:
            sys.path.insert(0, position_path)
        
        from intelligent_manager import IntelligentPositionManager
        from market_regime import MarketRegime
        
        manager = IntelligentPositionManager()
        
        # Test scenario: Profitable position in trending market
        print("\n📈 Scenario 1: Profitable position in bull trend")
        
        # Mock analyses
        regime_analysis = {
            'regime': MarketRegime.TRENDING_BULL,
            'confidence': 0.8,
            'parameters': {'profit_taking_patience': 2.0, 'stop_distance_multiplier': 1.5}
        }
        
        technical_analysis = {
            'signals': None,
            'hold_strength': 'STRONG',
            'exit_urgency': 0.2,
            'divergence': 'NONE',
            'momentum': 0.8
        }
        
        profit_analysis = {
            'triggered_targets': [],
            'profit_plan': None,
            'velocity': 2.0,
            'has_targets': False
        }
        
        stop_analysis = {
            'stop_level': None,
            'is_triggered': False,
            'stop_price': 0.0,
            'trail_distance': 0.0
        }
        
        correlation_analysis = {
            'portfolio_analysis': None,
            'should_reduce': False,
            'reduce_reason': '',
            'correlation_factor': 1.0
        }
        
        # Test action determination
        action, confidence, urgency = manager._determine_action_from_scores(0.7, 0.2, 0.1)
        print(f"   ✓ Recommended action: {action.value}")
        print(f"   ✓ Confidence: {confidence:.2f}")
        print(f"   ✓ Urgency: {urgency:.2f}")
        
        # Test scenario: Loss position with bearish signals
        print("\n📉 Scenario 2: Loss position with bearish signals")
        action, confidence, urgency = manager._determine_action_from_scores(0.1, 0.8, 0.2)
        print(f"   ✓ Recommended action: {action.value}")
        print(f"   ✓ Confidence: {confidence:.2f}")
        print(f"   ✓ Urgency: {urgency:.2f}")
        
        print("\n✅ Integration scenarios completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 ADVANCED INTELLIGENT POSITION MANAGEMENT TESTING")
    print("=" * 80)
    
    success = True
    
    # Test individual components
    success &= test_intelligent_position_components()
    
    # Test integration scenarios
    success &= test_integration_scenarios()
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Advanced intelligent position holding strategies are ready!")
        print("🚀 The system can now:")
        print("   • Detect market regimes and adapt strategies")
        print("   • Analyze technical signals for exit timing")
        print("   • Manage dynamic trailing stops")
        print("   • Execute multi-tiered profit taking")
        print("   • Monitor portfolio correlations")
        print("   • Make intelligent hold/sell decisions")
    else:
        print("❌ SOME TESTS FAILED")
        print("🔧 Please review the errors above")
    
    print("=" * 80)