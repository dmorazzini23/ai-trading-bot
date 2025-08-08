"""
Single Settings singleton with aliases for Alpaca credentials.

Provides centralized configuration management with environment variable
aliases and singleton pattern to ensure consistent config across modules.
"""

from functools import lru_cache
from typing import Tuple, Optional
import logging

# AI-AGENT-REF: graceful fallback for missing pydantic-settings
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    # Fallback for testing environments
    class BaseSettings:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    def Field(*args, **kwargs):
        return kwargs.get('default')
    
    PYDANTIC_AVAILABLE = False

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Unified settings with Alpaca credential aliases and singleton pattern.
    
    Supports both ALPACA_* and APCA_* environment variable formats for
    maximum compatibility with different deployment environments.
    """
    
    # Alpaca credentials with aliases (accept both formats)
    alpaca_api_key: Optional[str] = Field(
        default=None, 
        validation_alias='ALPACA_API_KEY',
        description="Alpaca API key (ALPACA_API_KEY or APCA_API_KEY_ID)"
    )
    alpaca_api_key_alt: Optional[str] = Field(
        default=None, 
        validation_alias='APCA_API_KEY_ID',
        description="Alternative Alpaca API key format"
    )
    alpaca_secret_key: Optional[str] = Field(
        default=None, 
        validation_alias='ALPACA_SECRET_KEY',
        description="Alpaca secret key (ALPACA_SECRET_KEY or APCA_API_SECRET_KEY)"
    )
    alpaca_secret_key_alt: Optional[str] = Field(
        default=None, 
        validation_alias='APCA_API_SECRET_KEY',
        description="Alternative Alpaca secret key format"
    )
    alpaca_base_url: Optional[str] = Field(
        default="https://paper-api.alpaca.markets", 
        validation_alias='ALPACA_BASE_URL',
        description="Alpaca API base URL"
    )
    
    # Trading configuration
    trading_mode: str = Field(
        default="paper", 
        validation_alias='TRADING_MODE',
        description="Trading mode (paper/live)"
    )
    bot_mode: str = Field(
        default="balanced", 
        validation_alias='BOT_MODE',
        description="Bot trading mode"
    )
    
    # Operational flags
    dry_run: bool = Field(
        default=False, 
        validation_alias='DRY_RUN',
        description="Simulate trades without execution"
    )
    shadow_mode: bool = Field(
        default=False, 
        validation_alias='SHADOW_MODE',
        description="Log trades without execution"
    )
    force_trades: bool = Field(
        default=False, 
        validation_alias='FORCE_TRADES',
        description="Override safety checks (DANGEROUS)"
    )
    
    # API Keys (optional)
    finnhub_api_key: Optional[str] = Field(
        default=None, 
        validation_alias='FINNHUB_API_KEY',
        description="Finnhub API key"
    )
    news_api_key: Optional[str] = Field(
        default=None, 
        validation_alias='NEWS_API_KEY',
        description="News API key"
    )
    
    # File paths
    model_path: str = Field(
        default="trained_model.pkl", 
        validation_alias='MODEL_PATH',
        description="ML model file path"
    )
    trade_log_file: str = Field(
        default="data/trades.csv", 
        validation_alias='TRADE_LOG_FILE',
        description="Trade log file path"
    )
    
    # Risk management
    max_portfolio_positions: int = Field(
        default=20, 
        validation_alias='MAX_PORTFOLIO_POSITIONS',
        description="Maximum portfolio positions"
    )
    max_open_positions: int = Field(
        default=10, 
        validation_alias='MAX_OPEN_POSITIONS',
        description="Maximum concurrent positions"
    )
    buy_threshold: float = Field(
        default=0.5, 
        validation_alias='BUY_THRESHOLD',
        description="Signal strength threshold for buys"
    )
    
    # Rate limiting
    rate_limit_budget: int = Field(
        default=190, 
        validation_alias='RATE_LIMIT_BUDGET',
        description="API rate limit budget"
    )

    if PYDANTIC_AVAILABLE:
        model_config = {
            'env_file': '.env',
            'env_file_encoding': 'utf-8',
            'case_sensitive': True,
            'extra': 'ignore'  # Allow unknown env vars
        }

    def get_alpaca_keys(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Get Alpaca API credentials using alias fallback.
        
        Returns:
            Tuple of (api_key, secret_key) or (None, None) if not found
        """
        # Try primary format first, then alternative format
        api_key = self.alpaca_api_key or self.alpaca_api_key_alt
        secret_key = self.alpaca_secret_key or self.alpaca_secret_key_alt
        
        return api_key, secret_key
    
    def has_alpaca_credentials(self) -> bool:
        """Check if valid Alpaca credentials are available."""
        api_key, secret_key = self.get_alpaca_keys()
        return bool(api_key and secret_key)
    
    def get_alpaca_config(self) -> dict:
        """Get Alpaca client configuration dictionary."""
        api_key, secret_key = self.get_alpaca_keys()
        return {
            'api_key': api_key,
            'secret_key': secret_key,
            'base_url': self.alpaca_base_url,
            'paper': self.trading_mode == 'paper'
        }


# AI-AGENT-REF: Fallback settings class for environments without pydantic
if not PYDANTIC_AVAILABLE:
    import os
    
    class FallbackSettings:
        """Fallback settings when pydantic-settings is not available."""
        
        def __init__(self):
            # Alpaca credentials with alias support
            self.alpaca_api_key = os.getenv("ALPACA_API_KEY")
            self.alpaca_api_key_alt = os.getenv("APCA_API_KEY_ID")
            self.alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY")
            self.alpaca_secret_key_alt = os.getenv("APCA_API_SECRET_KEY")
            self.alpaca_base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
            
            # Trading configuration
            self.trading_mode = os.getenv("TRADING_MODE", "paper")
            self.bot_mode = os.getenv("BOT_MODE", "balanced")
            
            # Operational flags
            self.dry_run = os.getenv("DRY_RUN", "false").lower() in ("true", "1", "yes")
            self.shadow_mode = os.getenv("SHADOW_MODE", "false").lower() in ("true", "1", "yes")
            self.force_trades = os.getenv("FORCE_TRADES", "false").lower() in ("true", "1", "yes")
            
            # API Keys
            self.finnhub_api_key = os.getenv("FINNHUB_API_KEY")
            self.news_api_key = os.getenv("NEWS_API_KEY")
            
            # File paths
            self.model_path = os.getenv("MODEL_PATH", "trained_model.pkl")
            self.trade_log_file = os.getenv("TRADE_LOG_FILE", "data/trades.csv")
            
            # Risk management
            self.max_portfolio_positions = int(os.getenv("MAX_PORTFOLIO_POSITIONS", "20"))
            self.max_open_positions = int(os.getenv("MAX_OPEN_POSITIONS", "10"))
            self.buy_threshold = float(os.getenv("BUY_THRESHOLD", "0.5"))
            
            # Rate limiting
            self.rate_limit_budget = int(os.getenv("RATE_LIMIT_BUDGET", "190"))
        
        def get_alpaca_keys(self) -> Tuple[Optional[str], Optional[str]]:
            """Get Alpaca API credentials using alias fallback."""
            api_key = self.alpaca_api_key or self.alpaca_api_key_alt
            secret_key = self.alpaca_secret_key or self.alpaca_secret_key_alt
            return api_key, secret_key
        
        def has_alpaca_credentials(self) -> bool:
            """Check if valid Alpaca credentials are available."""
            api_key, secret_key = self.get_alpaca_keys()
            return bool(api_key and secret_key)
        
        def get_alpaca_config(self) -> dict:
            """Get Alpaca client configuration dictionary."""
            api_key, secret_key = self.get_alpaca_keys()
            return {
                'api_key': api_key,
                'secret_key': secret_key,
                'base_url': self.alpaca_base_url,
                'paper': self.trading_mode == 'paper'
            }
    
    # Replace the Settings class for fallback mode
    Settings = FallbackSettings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get or create singleton Settings instance.
    
    Uses LRU cache with maxsize=1 to implement singleton pattern.
    Thread-safe and ensures only one Settings instance exists.
    
    Returns:
        Settings singleton instance
    """
    try:
        if PYDANTIC_AVAILABLE:
            settings = Settings()
        else:
            settings = Settings()
            logger.warning("Using fallback settings - pydantic-settings not available")
        
        logger.debug("Settings singleton created/retrieved")
        return settings
        
    except Exception as e:
        logger.error(f"Failed to create settings: {e}")
        # Return fallback settings in case of error
        return FallbackSettings() if not PYDANTIC_AVAILABLE else Settings()


# Convenience functions for backward compatibility
def validate_alpaca_credentials() -> bool:
    """Validate that Alpaca credentials are present and non-empty."""
    settings = get_settings()
    return settings.has_alpaca_credentials()


def get_masked_config() -> dict:
    """Get configuration with sensitive values masked for logging."""
    settings = get_settings()
    api_key, secret_key = settings.get_alpaca_keys()
    
    def mask_value(value: Optional[str]) -> str:
        """Mask sensitive value for logging."""
        if not value:
            return "NOT_SET"
        if len(value) <= 8:
            return "***MASKED***"
        return f"{value[:4]}...{value[-4:]}"
    
    return {
        'alpaca_api_key': mask_value(api_key),
        'alpaca_secret_key': mask_value(secret_key),
        'alpaca_base_url': settings.alpaca_base_url,
        'trading_mode': settings.trading_mode,
        'bot_mode': settings.bot_mode,
        'dry_run': settings.dry_run,
        'shadow_mode': settings.shadow_mode,
        'has_credentials': settings.has_alpaca_credentials(),
        'finnhub_api_key': mask_value(settings.finnhub_api_key),
        'news_api_key': mask_value(settings.news_api_key),
    }


if __name__ == "__main__":
    # Test the settings singleton
    print("Testing Settings Singleton...")
    
    # Test singleton behavior
    settings1 = get_settings()
    settings2 = get_settings()
    print(f"Singleton test: {settings1 is settings2}")
    
    # Test credential handling
    api_key, secret_key = settings1.get_alpaca_keys()
    print(f"Has credentials: {settings1.has_alpaca_credentials()}")
    
    # Test masked config
    masked = get_masked_config()
    print(f"Masked config: {masked}")
    
    print("Settings singleton tests completed!")