"""
Simple momentum strategy for testing enhanced strategy discovery.
This demonstrates that the framework can discover strategies in any submodule.
"""

from typing import List

from ai_trading.strategies.base import BaseStrategy, StrategySignal
from ai_trading.core.enums import OrderSide, RiskLevel


class MomentumStrategy(BaseStrategy):
    """
    Simple momentum-based trading strategy for testing.
    
    This is a minimal strategy that demonstrates the strategy discovery framework
    can find concrete strategies defined in any submodule under ai_trading/strategies/.
    """
    
    def __init__(
        self,
        strategy_id: str = "momentum",
        name: str = "Simple Momentum Strategy", 
        risk_level: RiskLevel = RiskLevel.MODERATE
    ):
        """Initialize momentum strategy."""
        super().__init__(strategy_id, name, risk_level)
        
        # Strategy parameters
        self.parameters = {
            "lookback_period": 20,  # Days to look back for momentum calculation
            "momentum_threshold": 0.02,  # 2% threshold for momentum signal
            "max_position_size": 0.05,  # 5% max position size
        }
        
    def generate_signals(self, market_data: dict) -> List[StrategySignal]:
        """
        Generate trading signals based on simple momentum.
        
        Args:
            market_data: Current market data and indicators
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # Simple momentum logic for demonstration
        # In a real strategy, this would use actual market data
        for symbol in market_data.get('symbols', []):
            # Mock momentum calculation
            momentum = 0.03  # Mock 3% momentum
            
            if momentum > self.parameters['momentum_threshold']:
                signal = StrategySignal(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    strength=min(momentum / 0.05, 1.0),  # Scale to [0,1]
                    confidence=0.7,
                    strategy_id=self.strategy_id,
                    signal_type="momentum"
                )
                signals.append(signal)
                
        return signals

    # Back-compat: engine may call `generate(ctx)`
    def generate(self, ctx) -> List[StrategySignal]:
        # AI-AGENT-REF: ensure MomentumStrategy exposes legacy generate()
        return super().generate(ctx)
    
    def calculate_position_size(
        self,
        signal: StrategySignal,
        portfolio_value: float,
        current_position: float = 0
    ) -> int:
        """
        Calculate optimal position size for signal.
        
        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value  
            current_position: Current position size
            
        Returns:
            Recommended position size
        """
        # Simple position sizing based on signal strength and max position size
        max_dollar_amount = portfolio_value * self.parameters['max_position_size']
        
        # Assume $100 per share for simplicity (in real strategy, would use actual price)
        assumed_price = 100.0
        max_shares = int(max_dollar_amount / assumed_price)
        
        # Scale by signal strength
        position_size = int(max_shares * signal.strength)
        
        return max(1, position_size)  # At least 1 share
