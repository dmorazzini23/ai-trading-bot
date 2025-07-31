"""Environment validation using pydantic-settings with enhanced security checks."""

import logging
import re
import os
from typing import Optional, List, Dict, Any
from datetime import datetime

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import validator, Field

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Application settings loaded from environment with comprehensive validation.
    
    This class defines all configuration parameters for the AI trading bot with
    built-in validation, type checking, and security measures. It ensures that
    critical settings are properly configured before the bot starts trading.
    """
    
    # Server Configuration
    FLASK_PORT: int = Field(default=9000, ge=1024, le=65535, description="Flask web server port")
    HEALTHCHECK_PORT: int = Field(default=8081, ge=1024, le=65535, description="Health check service port")
    RUN_HEALTHCHECK: str = Field(default="0", description="Enable health check endpoint")
    WEBHOOK_PORT: int = Field(default=9000, ge=1024, le=65535, description="Webhook listener port")
    
    # Critical API Keys (Required)
    ALPACA_API_KEY: str = Field(..., min_length=1, description="Alpaca API key for trading")
    ALPACA_SECRET_KEY: str = Field(..., min_length=1, description="Alpaca secret key for trading")
    
    # AI-AGENT-REF: Support for live vs paper trading configuration
    TRADING_MODE: str = Field(default="paper", description="Trading mode (paper/live)")
    ALPACA_BASE_URL: str = Field(default="https://paper-api.alpaca.markets", description="Alpaca API base URL")
    ALPACA_DATA_FEED: str = Field(default="iex", description="Alpaca data feed source")
    
    # Optional API Keys
    FINNHUB_API_KEY: Optional[str] = Field(default=None, description="Finnhub API key for market data")
    FUNDAMENTAL_API_KEY: Optional[str] = Field(default=None, description="Fundamental data API key")
    NEWS_API_KEY: Optional[str] = Field(default=None, description="News API key")
    IEX_API_TOKEN: Optional[str] = Field(default=None, description="IEX Cloud API token")
    
    # Trading Configuration
    BOT_MODE: str = Field(default="balanced", description="Trading mode (conservative/balanced/aggressive)")
    MODEL_PATH: str = Field(default="trained_model.pkl", description="Path to ML model file")
    HALT_FLAG_PATH: str = Field(default="halt.flag", description="Emergency halt flag file")
    MAX_PORTFOLIO_POSITIONS: int = Field(default=20, ge=1, le=100, description="Maximum portfolio positions")
    MAX_OPEN_POSITIONS: int = Field(default=10, ge=1, le=50, description="Maximum concurrent positions")
    
    # Risk Management
    LIMIT_ORDER_SLIPPAGE: float = Field(default=0.005, ge=0.0, le=0.1, description="Maximum slippage for limit orders")
    SLIPPAGE_THRESHOLD: float = Field(default=0.003, ge=0.0, le=0.05, description="Slippage warning threshold")
    DISASTER_DD_LIMIT: float = Field(default=0.2, ge=0.05, le=0.5, description="Emergency drawdown limit")
    WEEKLY_DRAWDOWN_LIMIT: float = Field(default=0.15, ge=0.01, le=0.5, description="Weekly drawdown limit")
    DOLLAR_RISK_LIMIT: float = Field(default=0.02, ge=0.001, le=0.1, description="Dollar risk per trade")
    
    # Signal and Entry Thresholds
    BUY_THRESHOLD: float = Field(default=0.5, ge=0.0, le=1.0, description="Signal strength threshold for buys")
    VOLUME_THRESHOLD: int = Field(default=50000, ge=1000, description="Minimum daily volume requirement")
    
    # Security and Compliance
    WEBHOOK_SECRET: str = Field(default="", description="Webhook authentication secret")
    
    # Operation Modes
    SHADOW_MODE: bool = Field(default=False, description="Log trades without execution")
    DRY_RUN: bool = Field(default=False, description="Simulate trading without real orders")
    DISABLE_DAILY_RETRAIN: bool = Field(default=False, description="Skip daily model retraining")
    FORCE_TRADES: bool = Field(default=False, description="Override safety checks (DANGEROUS)")
    
    # File Paths
    TRADE_LOG_FILE: str = Field(default="data/trades.csv", description="Trade history log file")
    MODEL_RF_PATH: str = Field(default="model_rf.pkl", description="Random Forest model path")
    MODEL_XGB_PATH: str = Field(default="model_xgb.pkl", description="XGBoost model path")
    MODEL_LGB_PATH: str = Field(default="model_lgb.pkl", description="LightGBM model path")
    RL_MODEL_PATH: str = Field(default="rl_agent.zip", description="Reinforcement learning model path")
    
    # Advanced Features
    USE_RL_AGENT: bool = Field(default=False, description="Enable reinforcement learning agent")
    SECTOR_EXPOSURE_CAP: float = Field(default=0.4, ge=0.1, le=1.0, description="Maximum sector exposure")
    
    # Performance and Rate Limiting
    REBALANCE_INTERVAL_MIN: int = Field(default=1440, ge=1, description="Portfolio rebalance interval (minutes)")
    FINNHUB_RPM: int = Field(default=60, ge=1, le=300, description="Finnhub requests per minute limit")
    
    # Additional fields from main branch
    MINUTE_CACHE_TTL: int = Field(default=60, description="Cache TTL for minute data")
    EQUITY_EXPOSURE_CAP: float = Field(default=2.5, description="Maximum equity exposure")
    PORTFOLIO_EXPOSURE_CAP: float = Field(default=2.5, description="Maximum portfolio exposure")
    SEED: int = Field(default=42, description="Random seed for reproducibility")
    RATE_LIMIT_BUDGET: int = Field(default=190, description="API rate limit budget")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"  # allow unknown env vars (e.g. SLACK_WEBHOOK)
    )
    
    @validator('ALPACA_API_KEY')
    def validate_alpaca_api_key(cls, v):
        """Validate Alpaca API key format and security."""
        if not v:
            raise ValueError("ALPACA_API_KEY is required")
        
        # Basic format validation (Alpaca keys are typically alphanumeric)
        if not re.match(r'^[A-Z0-9]+$', v):
            logger.warning("ALPACA_API_KEY format may be invalid")
        
        # Check for common placeholder values
        placeholder_values = ['your_api_key', 'placeholder', 'changeme', 'test']
        if v.lower() in placeholder_values:
            raise ValueError("ALPACA_API_KEY appears to be a placeholder value")
        
        # Minimum length check
        if len(v) < 10:
            raise ValueError("ALPACA_API_KEY appears too short to be valid")
            
        return v
    
    @validator('ALPACA_SECRET_KEY')
    def validate_alpaca_secret_key(cls, v):
        """Validate Alpaca secret key format and security."""
        if not v:
            raise ValueError("ALPACA_SECRET_KEY is required")
        
        # Check for common placeholder values
        placeholder_values = ['your_secret_key', 'placeholder', 'changeme', 'test']
        if v.lower() in placeholder_values:
            raise ValueError("ALPACA_SECRET_KEY appears to be a placeholder value")
        
        # Minimum length check
        if len(v) < 20:
            raise ValueError("ALPACA_SECRET_KEY appears too short to be valid")
            
        return v
    
    @validator('ALPACA_BASE_URL')
    def validate_alpaca_base_url(cls, v):
        """Validate Alpaca base URL format."""
        valid_urls = [
            'https://api.alpaca.markets',      # Live trading
            'https://paper-api.alpaca.markets'  # Paper trading
        ]
        
        if v not in valid_urls:
            logger.warning(f"ALPACA_BASE_URL '{v}' is not a standard Alpaca URL")
        
        if not v.startswith('https://'):
            raise ValueError("ALPACA_BASE_URL must use HTTPS")
            
        return v
    
    @validator('BOT_MODE')
    def validate_bot_mode(cls, v):
        """Validate trading bot mode."""
        valid_modes = ['conservative', 'balanced', 'aggressive', 'paper', 'testing']
        if v.lower() not in valid_modes:
            raise ValueError(f"BOT_MODE must be one of: {valid_modes}")
        return v.lower()
    
    @validator('TRADING_MODE')
    def validate_trading_mode(cls, v):
        """Validate trading mode."""
        if v not in ["paper", "live"]:
            raise ValueError(f"Invalid TRADING_MODE: {v}. Must be 'paper' or 'live'")
        return v
    
    @validator('FORCE_TRADES')
    def validate_force_trades(cls, v):
        """Warn about dangerous FORCE_TRADES setting."""
        if v:
            logger.warning("⚠️  FORCE_TRADES is enabled - this bypasses safety checks!")
        return v


# Create global settings instance
settings = Settings()


def validate_trading_mode() -> None:
    """Validate trading mode configuration and set appropriate URLs."""
    # AI-AGENT-REF: Trading mode validation and URL configuration
    if settings.TRADING_MODE not in ["paper", "live"]:
        raise ValueError(f"Invalid TRADING_MODE: {settings.TRADING_MODE}. Must be 'paper' or 'live'")
    
    # Auto-configure base URL based on trading mode if not explicitly set
    if settings.TRADING_MODE == "live":
        if settings.ALPACA_BASE_URL == "https://paper-api.alpaca.markets":
            logger.warning("TRADING_MODE is 'live' but ALPACA_BASE_URL is paper trading URL")
            logger.warning("Auto-configuring for live trading. Set ALPACA_BASE_URL explicitly to override.")
            settings.ALPACA_BASE_URL = "https://api.alpaca.markets"
        logger.critical("LIVE TRADING MODE ENABLED - Real money at risk!")
    else:
        if settings.ALPACA_BASE_URL != "https://paper-api.alpaca.markets":
            logger.info("TRADING_MODE is 'paper', ensuring paper trading URL")
            settings.ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
        logger.info("Paper trading mode enabled - Safe for testing")


def get_alpaca_config() -> dict:
    """Get Alpaca client configuration based on current settings."""
    validate_trading_mode()
    return {
        "api_key": settings.ALPACA_API_KEY,
        "secret_key": settings.ALPACA_SECRET_KEY,
        "base_url": settings.ALPACA_BASE_URL,
        "paper": settings.TRADING_MODE == "paper"
    }


def validate_environment() -> tuple[bool, List[str], Settings]:
    """
    Comprehensive environment validation with security checks.
    
    Returns
    -------
    tuple[bool, List[str], Settings]
        - is_valid: Whether all validation checks passed
        - errors: List of validation error messages
        - settings: Validated settings object (None if validation failed)
    """
    errors = []
    
    try:
        # Additional security checks
        security_errors = _perform_security_checks(settings)
        errors.extend(security_errors)
        
        # File system checks
        filesystem_errors = _validate_file_paths(settings)
        errors.extend(filesystem_errors)
        
        # API connectivity checks (optional)
        api_errors = _validate_api_connectivity(settings)
        errors.extend(api_errors)
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info("✅ Environment validation passed")
        else:
            logger.error("❌ Environment validation failed with %d errors", len(errors))
            for error in errors:
                logger.error("  - %s", error)
        
        return is_valid, errors, settings
        
    except Exception as e:
        error_msg = f"Critical validation error: {e}"
        logger.error(error_msg)
        return False, [error_msg], None


def _perform_security_checks(settings: Settings) -> List[str]:
    """Perform additional security validation checks."""
    errors = []
    
    # Check for development/testing indicators in production
    if 'paper' not in settings.ALPACA_BASE_URL.lower():
        if settings.BOT_MODE in ['testing', 'development']:
            errors.append("Production API URL detected with test/dev mode")
        
        if settings.DRY_RUN:
            logger.warning("DRY_RUN enabled with production API - trades will not execute")
    
    # Validate webhook security
    if settings.WEBHOOK_SECRET and len(settings.WEBHOOK_SECRET) < 32:
        errors.append("WEBHOOK_SECRET should be at least 32 characters for security")
    
    # Check for insecure configurations
    if settings.FORCE_TRADES and 'paper' not in settings.ALPACA_BASE_URL.lower():
        errors.append("FORCE_TRADES should never be enabled in production")
    
    return errors


def _validate_file_paths(settings: Settings) -> List[str]:
    """Validate file paths and permissions."""
    errors = []
    
    # Check if log directory is writable
    try:
        log_dir = os.path.dirname(settings.TRADE_LOG_FILE)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Test write permission
        test_file = os.path.join(log_dir or '.', '.write_test')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        
    except (OSError, IOError) as e:
        errors.append(f"Cannot write to log directory: {e}")
    
    # Check model file existence (if required)
    if not settings.DRY_RUN and not settings.SHADOW_MODE:
        if not os.path.exists(settings.MODEL_PATH):
            logger.warning(f"Model file not found: {settings.MODEL_PATH}")
    
    return errors


def _validate_api_connectivity(settings: Settings) -> List[str]:
    """Optional API connectivity validation."""
    errors = []
    
    # This is optional and can be skipped if APIs are temporarily unavailable
    try:
        # Basic URL validation
        import requests
        
        # Test Alpaca API reachability (without authentication)
        response = requests.head(settings.ALPACA_BASE_URL, timeout=10)
        if response.status_code >= 500:
            logger.warning(f"Alpaca API may be experiencing issues: {response.status_code}")
            
    except requests.RequestException:
        # API connectivity issues are warnings, not errors
        logger.warning("Could not verify API connectivity (this may be temporary)")
    except ImportError:
        # requests not available - skip connectivity check
        pass
    
    return errors


def validate_trading_environment() -> bool:
    """
    Quick validation for trading environment readiness.
    
    Returns
    -------
    bool
        True if environment is ready for trading, False otherwise
    """
    is_valid, errors, settings_obj = validate_environment()
    
    if not is_valid:
        logger.error("Environment validation failed - trading cannot proceed")
        return False
    
    # Additional runtime checks
    if settings.BOT_MODE == 'testing' and 'paper' not in settings.ALPACA_BASE_URL.lower():
        logger.error("Testing mode with production API is not allowed")
        return False
    
    logger.info(f"Trading environment ready - Mode: {settings.BOT_MODE}, URL: {settings.ALPACA_BASE_URL}")
    return True


def generate_schema() -> dict:
    """Return JSONSchema for the environment settings."""
    # AI-AGENT-REF: expose env schema for validation in CI
    return Settings.model_json_schema()


def _main() -> None:  # pragma: no cover - simple CLI helper
    logger.info("Environment variables successfully validated")


if __name__ == "__main__":
    _main()