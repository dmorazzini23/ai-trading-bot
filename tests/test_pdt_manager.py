"""Tests for PDT Manager."""

from datetime import datetime
import pytest
from types import SimpleNamespace
from ai_trading.execution.pdt_manager import PDTManager, PDTStatus


class TestPDTManager:
    """Test PDT manager functionality."""
    
    def test_non_pdt_account(self):
        """Test that non-PDT accounts can trade freely."""
        
        manager = PDTManager()
        account = SimpleNamespace(
            pattern_day_trader=False,
            daytrade_count=0,
            daytrade_limit=3
        )
        
        status = manager.get_pdt_status(account)
        
        assert status.is_pattern_day_trader is False
        assert status.can_daytrade is True
        assert status.strategy_recommendation == "normal"
    
    def test_pdt_under_limit(self):
        """Test PDT account under the day trade limit."""
        
        manager = PDTManager()
        account = SimpleNamespace(
            pattern_day_trader=True,
            daytrade_count=1,
            daytrade_limit=3
        )
        
        status = manager.get_pdt_status(account)
        
        assert status.is_pattern_day_trader is True
        assert status.daytrade_count == 1
        assert status.daytrade_limit == 3
        assert status.can_daytrade is True
        assert status.remaining_daytrades == 2
        assert status.strategy_recommendation == "normal"
    
    def test_pdt_at_limit(self):
        """Test PDT account at the day trade limit."""
        
        manager = PDTManager()
        account = SimpleNamespace(
            pattern_day_trader=True,
            daytrade_count=3,
            daytrade_limit=3
        )
        
        status = manager.get_pdt_status(account)
        
        assert status.is_pattern_day_trader is True
        assert status.daytrade_count == 3
        assert status.can_daytrade is False
        assert status.remaining_daytrades == 0
        assert status.strategy_recommendation == "swing_only"
    
    def test_pdt_over_limit(self):
        """Test PDT account over the day trade limit (like in the logs)."""
        
        manager = PDTManager()
        account = SimpleNamespace(
            pattern_day_trader=True,
            daytrade_count=6,
            daytrade_limit=3
        )
        
        status = manager.get_pdt_status(account)
        
        assert status.is_pattern_day_trader is True
        assert status.daytrade_count == 6
        assert status.can_daytrade is False
        assert status.remaining_daytrades == 0
        assert status.strategy_recommendation == "swing_only"
    
    def test_pdt_conservative_mode(self):
        """Test conservative mode when one trade left."""
        
        manager = PDTManager()
        account = SimpleNamespace(
            pattern_day_trader=True,
            daytrade_count=2,
            daytrade_limit=3
        )
        
        status = manager.get_pdt_status(account)
        
        assert status.remaining_daytrades == 1
        assert status.strategy_recommendation == "conservative"

    def test_pdt_high_equity_exempts_daytrade_limit(self):
        """Pattern-day-trader flag should not constrain accounts above $25k equity."""

        manager = PDTManager()
        account = SimpleNamespace(
            pattern_day_trader=True,
            daytrade_count=2,
            daytrade_limit=3,
            equity=30000,
        )

        status = manager.get_pdt_status(account)

        assert status.is_pattern_day_trader is True
        assert status.pdt_limit_applicable is False
        assert status.can_daytrade is True
        assert status.strategy_recommendation == "normal"
    
    def test_should_allow_order_non_pdt(self):
        """Test order allowance for non-PDT account."""
        
        manager = PDTManager()
        account = SimpleNamespace(
            pattern_day_trader=False,
            daytrade_count=0,
            daytrade_limit=3
        )
        
        allow, reason, context = manager.should_allow_order(
            account, "AAPL", "buy", current_position=0
        )
        
        assert allow is True
        assert reason == "not_pdt"
    
    def test_should_allow_order_pdt_limit_reached_allows_swing_entry(self):
        """When PDT capacity is zero, entries must be allowed as swing mode entries."""
        
        manager = PDTManager()
        account = SimpleNamespace(
            pattern_day_trader=True,
            daytrade_count=6,
            daytrade_limit=3
        )
        
        allow, reason, context = manager.should_allow_order(
            account, "AAPL", "buy", current_position=0
        )

        assert allow is True
        assert reason == "swing_mode_entry"
        assert context["daytrade_count"] == 6
        assert context["daytrade_limit"] == 3

    def test_should_allow_order_pdt_limit_reached_existing_position(self):
        """Adding to an existing position when PDT is exhausted remains allowed with a warning."""

        manager = PDTManager()
        account = SimpleNamespace(
            pattern_day_trader=True,
            daytrade_count=5,
            daytrade_limit=3
        )

        allow, reason, context = manager.should_allow_order(
            account, "AAPL", "buy", current_position=10
        )

        assert allow is True
        assert reason == "pdt_conservative"
        assert context["daytrade_count"] == 5

    def test_should_allow_order_high_equity_returns_exempt_reason(self):
        manager = PDTManager()
        account = SimpleNamespace(
            pattern_day_trader=True,
            daytrade_count=2,
            daytrade_limit=3,
            equity=35000,
        )

        allow, reason, context = manager.should_allow_order(
            account, "AAPL", "buy", current_position=0
        )

        assert allow is True
        assert reason == "pdt_equity_exempt"
        assert context["pdt_limit_applicable"] is False

    def test_reset_cache_clears_cached_status(self):
        """Resetting the cache removes the cached status and timestamp."""

        manager = PDTManager()
        manager.cached_status = PDTStatus(
            is_pattern_day_trader=True,
            daytrade_count=1,
            daytrade_limit=3,
            can_daytrade=True,
            remaining_daytrades=2,
            strategy_recommendation="normal",
        )
        manager.last_check_time = datetime.now()

        manager.reset_cache()

        assert manager.cached_status is None
        assert manager.last_check_time is None
    
    def test_should_allow_closing_position(self):
        """Test that closing positions is always allowed."""
        
        manager = PDTManager()
        account = SimpleNamespace(
            pattern_day_trader=True,
            daytrade_count=6,
            daytrade_limit=3
        )
        
        # Closing long position
        allow, reason, context = manager.should_allow_order(
            account, "AAPL", "sell", current_position=100
        )
        
        assert allow is True
        assert reason == "closing_position"
        
        # Covering short position
        allow, reason, context = manager.should_allow_order(
            account, "AAPL", "buy", current_position=-100
        )
        
        assert allow is True
        assert reason == "closing_position"
    
    def test_swing_mode_entry(self):
        """Test swing mode allows entry when no position."""
        
        manager = PDTManager()
        account = SimpleNamespace(
            pattern_day_trader=True,
            daytrade_count=6,
            daytrade_limit=3
        )
        
        allow, reason, context = manager.should_allow_order(
            account, "AAPL", "buy", current_position=0, force_swing_mode=True
        )
        
        assert allow is True
        assert reason == "swing_mode_entry"
    
    def test_swing_mode_blocks_when_has_position(self):
        """Test swing mode blocks new orders when position exists."""
        
        manager = PDTManager()
        account = SimpleNamespace(
            pattern_day_trader=True,
            daytrade_count=6,
            daytrade_limit=3
        )
        
        allow, reason, context = manager.should_allow_order(
            account, "AAPL", "buy", current_position=100, force_swing_mode=True
        )
        
        assert allow is False
        assert reason == "swing_mode_has_position"
    
    def test_get_recommended_strategy(self):
        """Test strategy recommendations."""
        
        manager = PDTManager()
        
        # Non-PDT account
        account1 = SimpleNamespace(
            pattern_day_trader=False,
            daytrade_count=0,
            daytrade_limit=3
        )
        assert manager.get_recommended_strategy(account1) == "day_trading"

        manager.reset_cache()

        # PDT at limit
        account2 = SimpleNamespace(
            pattern_day_trader=True,
            daytrade_count=3,
            daytrade_limit=3
        )
        assert manager.get_recommended_strategy(account2) == "swing_trading"

        manager.reset_cache()

        # PDT with 1 trade left
        account3 = SimpleNamespace(
            pattern_day_trader=True,
            daytrade_count=2,
            daytrade_limit=3
        )
        assert manager.get_recommended_strategy(account3) == "conservative_swing"

        manager.reset_cache()

        # PDT with room
        account4 = SimpleNamespace(
            pattern_day_trader=True,
            daytrade_count=0,
            daytrade_limit=3
        )
        assert manager.get_recommended_strategy(account4) == "mixed"
    
    def test_cache_functionality(self):
        """Test that status is cached properly."""
        
        manager = PDTManager()
        account = SimpleNamespace(
            pattern_day_trader=True,
            daytrade_count=1,
            daytrade_limit=3
        )
        
        # First call
        status1 = manager.get_pdt_status(account)
        
        # Modify account (simulating change)
        account.daytrade_count = 2
        
        # Second call should return cached value
        status2 = manager.get_pdt_status(account)
        
        assert status1.daytrade_count == status2.daytrade_count == 1
        
        # After cache expires, should get new value
        manager.cache_ttl_seconds = 0
        status3 = manager.get_pdt_status(account)
        
        assert status3.daytrade_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
