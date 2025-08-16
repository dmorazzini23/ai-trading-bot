"""
Production configuration management for live trading.

This module provides secure configuration management, validation,
and runtime configuration changes with proper audit logging.
"""

import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypedDict

from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)  # AI-AGENT-REF: allow extras

logger = logging.getLogger(__name__)


# Helper functions to get settings without triggering import cascade
def _get_settings_safe():
    """Get settings instance."""
    from ai_trading.config import get_settings

    return get_settings()


def _get_secret_filter_safe():
    """Get SecretFilter."""
    from ai_trading.logging_filters import SecretFilter

    return SecretFilter()


# Ensure secrets are masked across all handlers attached to this logger.
secret_filter = _get_secret_filter_safe()
for h in logger.handlers:
    h.addFilter(secret_filter)
# Also try the root logger for broad coverage (safe no-op if none).
root = logging.getLogger()
for h in root.handlers:
    h.addFilter(secret_filter)


class LegacyParams(TypedDict):
    """Typed contract for legacy parameter export."""  # AI-AGENT-REF: typed legacy map

    KELLY_FRACTION: float
    CONF_THRESHOLD: float
    CONFIRMATION_COUNT: int
    LOOKBACK_DAYS: int
    MIN_SIGNAL_STRENGTH: float
    CAPITAL_CAP: float
    DOLLAR_RISK_LIMIT: float
    MAX_DRAWDOWN_THRESHOLD: float
    DAILY_LOSS_LIMIT: float
    REBALANCE_INTERVAL_MIN: int
    TRADE_COOLDOWN_MIN: int
    MAX_TRADES_PER_HOUR: int
    MAX_TRADES_PER_DAY: int
    TAKE_PROFIT_FACTOR: float
    STOP_LOSS: float
    TAKE_PROFIT: float
    TRAILING_FACTOR: float
    POSITION_SIZE_MIN_USD: float
    VOLUME_THRESHOLD: float
    SEED: int


class ConfigValidator:
    """
    Configuration validation and security checker.

    Validates configuration values, checks for security issues,
    and ensures configuration integrity.
    """

    def __init__(self):
        """Initialize configuration validator."""
        self.validation_rules = {
            # Trading mode validation
            "TRADING_MODE": {
                "type": str,
                "allowed_values": ["paper", "live"],
                "required": True,
                "default": "paper",
            },
            # API configuration
            "ALPACA_API_KEY": {
                "type": str,
                "required": True,
                "min_length": 10,
                "security_check": True,
            },
            "ALPACA_SECRET_KEY": {
                "type": str,
                "required": True,
                "min_length": 10,
                "security_check": True,
            },
            "ALPACA_BASE_URL": {
                "type": str,
                "required": True,
                "allowed_values": [
                    "https://api.alpaca.markets",
                    "https://paper-api.alpaca.markets",
                ],
            },
            # Risk management
            "MAX_PORTFOLIO_POSITIONS": {
                "type": int,
                "min_value": 1,
                "max_value": 100,
                "default": 20,
            },
            "LIMIT_ORDER_SLIPPAGE": {
                "type": float,
                "min_value": 0.0001,
                "max_value": 0.1,
                "default": 0.005,
            },
            "DISASTER_DD_LIMIT": {
                "type": float,
                "min_value": 0.01,
                "max_value": 0.5,
                "default": 0.2,
            },
            # Performance limits
            "VOLUME_THRESHOLD": {
                "type": int,
                "min_value": 1000,
                "max_value": 10000000,
                "default": 50000,
            },
            "DOLLAR_RISK_LIMIT": {
                "type": float,
                "min_value": 0.001,
                "max_value": 0.1,
                "default": 0.05,
            },
            # System configuration
            "FLASK_PORT": {
                "type": int,
                "min_value": 1024,
                "max_value": 65535,
                "default": 9000,
            },
            "HEALTHCHECK_PORT": {
                "type": int,
                "min_value": 1024,
                "max_value": 65535,
                "default": 8081,
            },
        }

    def validate_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Validate configuration against rules.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Validation result with errors and warnings
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "security_issues": [],
            "missing_required": [],
            "invalid_values": [],
        }

        # Check required fields
        for key, rules in self.validation_rules.items():
            if rules.get("required", False) and key not in config:
                result["missing_required"].append(key)
                result["valid"] = False

        # Validate existing configuration values
        for key, value in config.items():
            if key in self.validation_rules:
                validation_errors = self._validate_value(key, value, self.validation_rules[key])
                result["errors"].extend(validation_errors)

                if validation_errors:
                    result["valid"] = False
                    result["invalid_values"].append(
                        {"key": key, "value": str(value), "errors": validation_errors}
                    )

        # Security checks
        security_issues = self._check_security(config)
        result["security_issues"] = security_issues
        if security_issues:
            result["warnings"].extend([f"Security issue: {issue}" for issue in security_issues])

        return result

    def _validate_value(self, key: str, value: Any, rules: dict[str, Any]) -> list[str]:
        """Validate a single configuration value."""
        errors = []

        # Type validation
        expected_type = rules.get("type")
        if expected_type and not isinstance(value, expected_type):
            errors.append(f"Expected {expected_type.__name__}, got {type(value).__name__}")
            return errors  # Skip other validations if type is wrong

        # Allowed values validation
        allowed_values = rules.get("allowed_values")
        if allowed_values and value not in allowed_values:
            errors.append(f"Value must be one of: {allowed_values}")

        # Numeric range validation
        if isinstance(value, int | float):
            min_value = rules.get("min_value")
            max_value = rules.get("max_value")

            if min_value is not None and value < min_value:
                errors.append(f"Value must be >= {min_value}")

            if max_value is not None and value > max_value:
                errors.append(f"Value must be <= {max_value}")

        # String length validation
        if isinstance(value, str):
            min_length = rules.get("min_length")
            max_length = rules.get("max_length")

            if min_length is not None and len(value) < min_length:
                errors.append(f"Length must be >= {min_length}")

            if max_length is not None and len(value) > max_length:
                errors.append(f"Length must be <= {max_length}")

        return errors

    def _check_security(self, config: dict[str, Any]) -> list[str]:
        """Check for security issues in configuration."""
        issues = []

        # Check for live trading with weak credentials
        if config.get("TRADING_MODE") == "live":
            api_key = config.get("ALPACA_API_KEY", "")
            secret_key = config.get("ALPACA_SECRET_KEY", "")

            if len(api_key) < 20:
                issues.append("API key too short for live trading")

            if len(secret_key) < 20:
                issues.append("Secret key too short for live trading")

            if "test" in api_key.lower() or "demo" in api_key.lower():
                issues.append("API key appears to be test/demo key for live trading")

        # Check for default/weak passwords
        weak_indicators = ["test", "demo", "default", "admin", "password", "12345"]
        for key in ["ALPACA_API_KEY", "ALPACA_SECRET_KEY", "WEBHOOK_SECRET"]:
            value = config.get(key, "").lower()
            for indicator in weak_indicators:
                if indicator in value:
                    issues.append(f"Potentially weak credential detected in {key}")
                    break

        return issues


class ConfigManager:
    """
    Secure configuration manager for production trading.

    Manages configuration loading, validation, runtime updates,
    and secure storage with audit logging.
    """

    def __init__(self, config_dir: str = "config"):
        """Initialize configuration manager."""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

        self.validator = ConfigValidator()
        self.current_config = {}
        self.config_history = []
        self.config_hash = None

        # Configuration files
        self.main_config_file = self.config_dir / "main.json"
        self.backup_config_file = self.config_dir / "backup.json"
        self.audit_log_file = self.config_dir / "audit.log"

        logger.info("ConfigManager initialized")

    def load_config(self, config_file: str | None = None) -> dict[str, Any]:
        """
        Load configuration from file with validation.

        Args:
            config_file: Optional specific config file to load

        Returns:
            Loaded and validated configuration
        """
        try:
            # Determine config file to use
            if config_file:
                config_path = Path(config_file)
            elif self.main_config_file.exists():
                config_path = self.main_config_file
            else:
                # Create default configuration
                logger.info("No configuration file found, creating default")
                return self._create_default_config()

            # Load configuration
            with open(config_path) as f:
                config = json.load(f)

            # Merge with environment variables
            config = self._merge_with_env(config)

            # Validate configuration
            validation_result = self.validator.validate_config(config)

            if not validation_result["valid"]:
                logger.error(f"Configuration validation failed: {validation_result['errors']}")
                if validation_result["missing_required"]:
                    logger.error(
                        f"Missing required fields: {validation_result['missing_required']}"
                    )
                raise ValueError(f"Invalid configuration: {validation_result['errors']}")

            # Log warnings
            for warning in validation_result["warnings"]:
                logger.warning(f"Configuration warning: {warning}")

            # Calculate and store configuration hash
            self.config_hash = self._calculate_config_hash(config)

            # Store configuration
            self.current_config = config

            # Log configuration load
            self._log_config_change("load", f"Configuration loaded from {config_path}")

            logger.info(f"Configuration loaded successfully from {config_path}")
            return config

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def save_config(self, config: dict[str, Any], backup: bool = True) -> bool:
        """
        Save configuration to file with backup.

        Args:
            config: Configuration to save
            backup: Whether to create backup

        Returns:
            True if successful
        """
        try:
            # Validate before saving
            validation_result = self.validator.validate_config(config)
            if not validation_result["valid"]:
                logger.error(f"Cannot save invalid configuration: {validation_result['errors']}")
                return False

            # Create backup if requested
            if backup and self.main_config_file.exists():
                self.main_config_file.replace(self.backup_config_file)
                logger.info("Configuration backup created")

            # Save main configuration
            with open(self.main_config_file, "w") as f:
                json.dump(config, f, indent=2)

            # Update internal state
            old_hash = self.config_hash
            self.config_hash = self._calculate_config_hash(config)
            self.current_config = config

            # Log configuration change
            self._log_config_change(
                "save", f"Configuration saved (hash: {old_hash} -> {self.config_hash})"
            )

            logger.info("Configuration saved successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False

    def update_config(self, updates: dict[str, Any], reason: str = "Manual update") -> bool:
        """
        Update configuration at runtime.

        Args:
            updates: Configuration updates to apply
            reason: Reason for the update

        Returns:
            True if successful
        """
        try:
            # Create updated configuration
            new_config = self.current_config.copy()
            new_config.update(updates)

            # Validate updated configuration
            validation_result = self.validator.validate_config(new_config)
            if not validation_result["valid"]:
                logger.error(
                    f"Configuration update validation failed: {validation_result['errors']}"
                )
                return False

            # Check for critical changes
            critical_changes = self._detect_critical_changes(self.current_config, new_config)
            if critical_changes:
                logger.warning(f"Critical configuration changes detected: {critical_changes}")
                self._log_config_change("critical_update", f"Critical changes: {critical_changes}")

            # Apply updates
            old_config = self.current_config.copy()
            self.current_config = new_config
            self.config_hash = self._calculate_config_hash(new_config)

            # Save to file
            if not self.save_config(new_config):
                # Rollback on save failure
                self.current_config = old_config
                self.config_hash = self._calculate_config_hash(old_config)
                return False

            # Log update
            self._log_config_change("update", f"Runtime update: {updates} - Reason: {reason}")

            logger.info(f"Configuration updated successfully: {updates}")
            return True

        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return False

    def get_config(self) -> dict[str, Any]:
        """Get current configuration."""
        return self.current_config.copy()

    def get_config_summary(self) -> dict[str, Any]:
        """Get configuration summary for monitoring."""
        return {
            "hash": self.config_hash,
            "last_updated": (self.config_history[-1]["timestamp"] if self.config_history else None),
            "trading_mode": self.current_config.get("TRADING_MODE", "unknown"),
            "num_changes": len(self.config_history),
            "validation_status": (
                "valid"
                if self.validator.validate_config(self.current_config)["valid"]
                else "invalid"
            ),
        }

    def restore_backup(self) -> bool:
        """Restore configuration from backup."""
        try:
            if not self.backup_config_file.exists():
                logger.error("No backup configuration file found")
                return False

            # Load backup
            backup_config = self.load_config(str(self.backup_config_file))

            # Save as main configuration
            if self.save_config(backup_config):
                self._log_config_change("restore", "Configuration restored from backup")
                logger.info("Configuration restored from backup")
                return True
            else:
                return False

        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            return False

    def _create_default_config(self) -> dict[str, Any]:
        """Create default configuration."""
        default_config = {
            "TRADING_MODE": "paper",
            "ALPACA_BASE_URL": "https://paper-api.alpaca.markets",
            "ALPACA_DATA_FEED": "iex",
            "MAX_PORTFOLIO_POSITIONS": 20,
            "LIMIT_ORDER_SLIPPAGE": 0.005,
            "DISASTER_DD_LIMIT": 0.2,
            "VOLUME_THRESHOLD": 50000,
            "DOLLAR_RISK_LIMIT": 0.05,
            "FLASK_PORT": 9000,
            "HEALTHCHECK_PORT": 8081,
        }

        # Save default configuration
        self.save_config(default_config)
        self._log_config_change("create", "Default configuration created")

        return default_config

    def _merge_with_env(self, config: dict[str, Any]) -> dict[str, Any]:
        """Merge configuration with environment variables."""
        merged = config.copy()

        # Environment variables take precedence
        for key in self.validator.validation_rules:
            env_value = os.environ.get(key)
            if env_value is not None:
                # Convert to appropriate type
                expected_type = self.validator.validation_rules[key].get("type", str)
                try:
                    if expected_type == bool:
                        merged[key] = env_value.lower() in ("true", "1", "yes", "on")
                    elif expected_type == int:
                        merged[key] = int(env_value)
                    elif expected_type == float:
                        merged[key] = float(env_value)
                    else:
                        merged[key] = env_value
                except ValueError:
                    logger.warning(
                        f"Could not convert environment variable {key}={env_value} to {expected_type}"
                    )

        return merged

    def _calculate_config_hash(self, config: dict[str, Any]) -> str:
        """Calculate hash of configuration for change detection."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def _detect_critical_changes(
        self, old_config: dict[str, Any], new_config: dict[str, Any]
    ) -> list[str]:
        """Detect critical configuration changes."""
        critical_keys = [
            "TRADING_MODE",
            "ALPACA_BASE_URL",
            "ALPACA_API_KEY",
            "ALPACA_SECRET_KEY",
        ]

        changes = []
        for key in critical_keys:
            if old_config.get(key) != new_config.get(key):
                changes.append(key)

        return changes

    def _log_config_change(self, action: str, details: str):
        """Log configuration changes for audit."""
        timestamp = datetime.now(UTC).isoformat()

        log_entry = {
            "timestamp": timestamp,
            "action": action,
            "details": details,
            "config_hash": self.config_hash,
            "user": os.environ.get("USER", "unknown"),
        }

        self.config_history.append(log_entry)

        # Write to audit log file
        try:
            with open(self.audit_log_file, "a") as f:
                f.write(f"{json.dumps(log_entry)}\n")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")


# Convenience function for getting validated configuration
def get_production_config() -> dict[str, Any]:
    """Get validated production configuration."""
    config_manager = ConfigManager()
    return config_manager.load_config()


# Compatibility function for root config interface
def get_env(
    key: str,
    default: str | None = None,
    *,
    reload: bool = False,
    required: bool = False,
) -> str | None:
    """Return environment variable key.

    Parameters
    ----------
    key : str
        Name of the variable.
    default : str | None, optional
        Value returned if the variable is missing.
    reload : bool, optional
        Reload .env before checking when True.
    required : bool, optional
        If True and the variable is missing, raise RuntimeError.
    """
    import os
    from pathlib import Path

    if reload:
        from dotenv import load_dotenv

        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path, override=True)

    value = os.environ.get(key, default)
    if required and value is None:
        logger.error("Required environment variable '%s' is missing", key)
        raise RuntimeError(f"Required environment variable '{key}' is missing")
    return value


# Compatibility attributes for root config interface

SCHEDULER_SLEEP_SECONDS = float(os.getenv("SCHEDULER_SLEEP_SECONDS", "30"))
TESTING = os.getenv("TESTING", "false").lower() in ("true", "1", "yes")
SEED = int(os.getenv("SEED", "42"))
ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
REQUIRED_ENV_VARS = [
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "ALPACA_BASE_URL",
    "WEBHOOK_SECRET",
    "FLASK_PORT",
]

# Additional compatibility attributes
# AI-AGENT-REF: Use proper runtime paths for default file locations
from ai_trading import paths

TRADE_LOG_FILE = os.getenv("TRADE_LOG_FILE", str(paths.LOG_DIR / "trades.csv"))
MODEL_PATH = os.getenv("MODEL_PATH", str(paths.DATA_DIR / "models" / "trained_model.pkl"))
RL_MODEL_PATH = os.getenv("RL_MODEL_PATH", str(paths.DATA_DIR / "models" / "rl_model.pkl"))
HALT_FLAG_PATH = os.getenv("HALT_FLAG_PATH", str(paths.DATA_DIR / "halt.flag"))

USE_RL_AGENT = os.getenv("USE_RL_AGENT", "false").lower() in ("true", "1", "yes")
BOT_MODE = os.getenv("BOT_MODE", "balanced")
VERBOSE_LOGGING = os.getenv("VERBOSE_LOGGING", "false").lower() in ("true", "1", "yes")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
SENTIMENT_API_KEY = os.getenv("SENTIMENT_API_KEY") or os.getenv("NEWS_API_KEY")
SENTIMENT_API_URL = os.getenv("SENTIMENT_API_URL", "https://newsapi.org/v2/everything")
IEX_API_TOKEN = os.getenv("IEX_API_TOKEN")
ALPACA_PAPER = "paper" in ALPACA_BASE_URL.lower()
MIN_HEALTH_ROWS = int(os.getenv("MIN_HEALTH_ROWS", "50"))
MAX_DRAWDOWN_THRESHOLD = float(os.getenv("MAX_DRAWDOWN_THRESHOLD", "0.2"))


# TradingConfig class for compatibility
class _LegacyTradingConfig:
    def __init__(
        self,
        mode: str = "balanced",
        trailing_factor: float = 1.0,
        kelly_fraction: float = 0.6,
        max_position_size: float = 1.0,
        stop_loss: float = 0.05,
        take_profit: float = 0.10,
        take_profit_factor: float = 2.0,
        lookback_days: int = 60,
        min_signal_strength: float = 0.1,
        scaling_factor: float = 1.0,
        limit_order_slippage: float = 0.001,
        pov_slice_pct: float = 0.05,
        daily_loss_limit: float = 0.03,
        max_portfolio_risk: float = 0.10,
        entry_start_offset_min: int = 0,
        entry_end_offset_min: int = 0,
        conf_threshold: float = 0.75,
        buy_threshold: float = 0.50,
        confirmation_count: int = 2,
        # Required trading risk parameters
        capital_cap: float = 0.25,
        dollar_risk_limit: float = 0.05,
        position_size_min_usd: float = 0.0,
        # model/feature toggles
        enable_finbert: bool = False,
        enable_sklearn: bool = False,
        # strategy/allocator knobs (added)
        signal_confirmation_bars: int = 2,
        delta_threshold: float = 0.02,
        min_confidence: float = 0.60,
        # allocator-related defaults (silence warnings; explicit behavior)
        signal_confirm_bars: int = 2,
        delta_hold: float = 0.02,
        # Required new attributes for production reliability
        trading_mode: str = "paper",
        alpaca_base_url: str = "https://paper-api.alpaca.markets",
        sleep_interval: float = 1.0,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        max_backoff_interval: float = 60.0,
        pct: float = 0.05,
        MODEL_PATH: str = None,
        scheduler_iterations: int = 0,
        scheduler_sleep_seconds: int = 60,
        window: int = 20,
        enabled: bool = True,
        # Rate limiter configuration
        capacity: int = 100,
        refill_rate: float = 10.0,
        queue_timeout: float = 30.0,
        # AI-AGENT-REF: optional model configuration
        ml_model_path: str | None = None,
        ml_model_module: str | None = None,
        halt_file: str | None = None,
        intraday_lookback_minutes: int = 120,
        data_warmup_lookback_days: int = 60,
        disable_daily_retrain: bool = False,
        REGIME_MIN_ROWS: int = 200,
        signal_period: int = 20,
        **kwargs,
    ):
        self.mode = mode
        self.trailing_factor = trailing_factor
        self.kelly_fraction = kelly_fraction
        self.max_position_size = max_position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.take_profit_factor = take_profit_factor
        self.lookback_days = lookback_days
        self.min_signal_strength = min_signal_strength
        self.scaling_factor = scaling_factor
        self.limit_order_slippage = limit_order_slippage
        self.pov_slice_pct = pov_slice_pct
        self.daily_loss_limit = daily_loss_limit
        self.max_portfolio_risk = max_portfolio_risk
        self.entry_start_offset_min = entry_start_offset_min
        self.entry_end_offset_min = entry_end_offset_min
        self.conf_threshold = conf_threshold
        self.buy_threshold = buy_threshold
        self.confirmation_count = confirmation_count

        self.signal_period = signal_period

        # Required trading risk parameters
        self.capital_cap = capital_cap
        self.dollar_risk_limit = dollar_risk_limit
        self.position_size_min_usd = position_size_min_usd

        # feature toggles
        self.enable_finbert = bool(enable_finbert)
        self.enable_sklearn = bool(enable_sklearn)
        # strategy/allocator knobs
        self.signal_confirmation_bars = signal_confirmation_bars
        self.delta_threshold = delta_threshold
        self.min_confidence = min_confidence
        # allocator/strategy knobs
        self.signal_confirm_bars = signal_confirm_bars
        self.delta_hold = delta_hold

        # Required new attributes for production reliability
        self.trading_mode = trading_mode
        self.alpaca_base_url = alpaca_base_url
        self.sleep_interval = sleep_interval
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.max_backoff_interval = max_backoff_interval
        self.pct = pct
        self.MODEL_PATH = MODEL_PATH
        self.scheduler_iterations = scheduler_iterations
        self.scheduler_sleep_seconds = scheduler_sleep_seconds
        self.window = window
        self.enabled = enabled
        # Rate limiter configuration
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.queue_timeout = queue_timeout
        # AI-AGENT-REF: ML model configuration and halt file
        self.ml_model_path = ml_model_path
        self.ml_model_module = ml_model_module
        self.halt_file = halt_file
        self.intraday_lookback_minutes = intraday_lookback_minutes
        self.data_warmup_lookback_days = data_warmup_lookback_days
        self.disable_daily_retrain = bool(disable_daily_retrain)
        self.REGIME_MIN_ROWS = REGIME_MIN_ROWS

        # Basic validation for new fields
        if not (0.0 <= self.conf_threshold <= 1.0):
            raise ValueError(f"conf_threshold must be in [0,1], got {self.conf_threshold}")
        if not (0.0 <= self.buy_threshold <= 1.0):
            raise ValueError(f"buy_threshold must be in [0,1], got {self.buy_threshold}")
        if not isinstance(self.confirmation_count, int) or self.confirmation_count < 1:
            raise ValueError(f"confirmation_count must be >= 1, got {self.confirmation_count}")

        # Apply kwargs fallbacks into attributes if present
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_env(cls, mode=None, **overrides):
        """
        Build a TradingConfig from environment variables and optional overrides.
        Must be a classmethod so callers can use TradingConfig.from_env(...).
        """
        import os

        getenv = os.getenv

        mode = (mode or overrides.get("mode") or getenv("TRADING_MODE", "balanced")).lower()
        if mode not in {"conservative", "balanced", "aggressive"}:
            mode = "balanced"
        defaults = {
            "conservative": {"kelly_fraction": 0.25, "conf_threshold": 0.85},
            "balanced": {"kelly_fraction": 0.60, "conf_threshold": 0.75},
            "aggressive": {"kelly_fraction": 0.75, "conf_threshold": 0.65},
        }
        mode_defaults = defaults[mode]
        from ai_trading.config.settings import (
            get_settings as get_config_settings,
        )
        from ai_trading.settings import (
            _secret_to_str,
            _to_bool,
            _to_float,
            _to_int,
            get_max_drawdown_threshold,
            get_seed_int,
        )
        from ai_trading.settings import (
            get_settings as get_runtime_settings,
        )

        # extract values using normalization helpers
        k_env = getenv("KELLY_FRACTION")
        c_env = getenv("CONF_THRESHOLD")
        kelly_fraction = (
            float(k_env)
            if k_env is not None
            else overrides.get("kelly_fraction", mode_defaults["kelly_fraction"])
        )
        conf_threshold = (
            float(c_env)
            if c_env is not None
            else overrides.get("conf_threshold", mode_defaults["conf_threshold"])
        )
        daily_loss_limit = _to_float(
            getenv("DAILY_LOSS_LIMIT", overrides.get("daily_loss_limit", 0.03)),
            overrides.get("daily_loss_limit", 0.03),
        )
        max_portfolio_risk = _to_float(
            getenv("MAX_PORTFOLIO_RISK", overrides.get("max_portfolio_risk", 0.10)),
            overrides.get("max_portfolio_risk", 0.10),
        )
        buy_threshold = _to_float(
            getenv("BUY_THRESHOLD", overrides.get("buy_threshold", 0.60)),
            overrides.get("buy_threshold", 0.60),
        )
        confirmation_count = _to_int(
            getenv("CONFIRMATION_COUNT", overrides.get("confirmation_count", 2)), 2
        )

        # Required trading risk parameters
        capital_cap = _to_float(
            getenv("CAPITAL_CAP", overrides.get("capital_cap", 0.25)),
            overrides.get("capital_cap", 0.25),
        )
        dollar_risk_limit = _to_float(
            getenv("DOLLAR_RISK_LIMIT", overrides.get("dollar_risk_limit", 0.05)),
            overrides.get("dollar_risk_limit", 0.05),
        )
        max_position_size = _to_float(
            getenv("MAX_POSITION_SIZE", overrides.get("max_position_size", 1.0)),
            overrides.get("max_position_size", 1.0),
        )

        # feature toggles
        enable_finbert = _to_bool(
            getenv("ENABLE_FINBERT", overrides.get("enable_finbert", False)),
            overrides.get("enable_finbert", False),
        )
        enable_sklearn = _to_bool(
            getenv("ENABLE_SKLEARN", overrides.get("enable_sklearn", False)),
            overrides.get("enable_sklearn", False),
        )
        intraday_lookback_minutes = _to_int(
            getenv(
                "INTRADAY_LOOKBACK_MINUTES",
                overrides.get("intraday_lookback_minutes", 120),
            ),
            overrides.get("intraday_lookback_minutes", 120),
        )
        data_warmup_lookback_days = _to_int(
            getenv(
                "DATA_WARMUP_LOOKBACK_DAYS",
                overrides.get("data_warmup_lookback_days", 60),
            ),
            overrides.get("data_warmup_lookback_days", 60),
        )
        disable_daily_retrain = _to_bool(
            getenv(
                "DISABLE_DAILY_RETRAIN",
                overrides.get("disable_daily_retrain", False),
            ),
            overrides.get("disable_daily_retrain", False),
        )
        REGIME_MIN_ROWS = _to_int(
            getenv("REGIME_MIN_ROWS", overrides.get("REGIME_MIN_ROWS", 200)), 200
        )
        # allocator/strategy fields (use overrides > env > defaults)
        signal_confirm_bars = _to_int(
            getenv("SIGNAL_CONFIRM_BARS", overrides.get("signal_confirm_bars", 2)),
            overrides.get("signal_confirm_bars", 2),
        )
        delta_hold = _to_float(
            getenv("DELTA_HOLD", overrides.get("delta_hold", 0.02)),
            overrides.get("delta_hold", 0.02),
        )
        min_confidence = _to_float(
            getenv("MIN_CONFIDENCE", overrides.get("min_confidence", 0.60)),
            overrides.get("min_confidence", 0.60),
        )
        signal_period = _to_int(
            getenv("SIGNAL_PERIOD", overrides.get("signal_period", 20)),
            overrides.get("signal_period", 20),
        )

        # Required new attributes from environment variables
        trading_mode = getenv("TRADING_MODE", overrides.get("trading_mode", "paper"))
        alpaca_base_url = getenv(
            "ALPACA_BASE_URL",
            overrides.get("alpaca_base_url", "https://paper-api.alpaca.markets"),
        )
        sleep_interval = _to_float(
            getenv("SLEEP_INTERVAL", overrides.get("sleep_interval", 1.0)),
            overrides.get("sleep_interval", 1.0),
        )
        max_retries = _to_int(
            getenv("MAX_RETRIES", overrides.get("max_retries", 3)),
            overrides.get("max_retries", 3),
        )
        backoff_factor = _to_float(
            getenv("BACKOFF_FACTOR", overrides.get("backoff_factor", 2.0)),
            overrides.get("backoff_factor", 2.0),
        )
        max_backoff_interval = _to_float(
            getenv("MAX_BACKOFF_INTERVAL", overrides.get("max_backoff_interval", 60.0)),
            overrides.get("max_backoff_interval", 60.0),
        )
        pct = _to_float(
            getenv("PCT", overrides.get("pct", 0.05)),
            overrides.get("pct", 0.05),
        )
        MODEL_PATH = getenv("MODEL_PATH", overrides.get("MODEL_PATH", None))
        scheduler_iterations = _to_int(
            getenv("SCHEDULER_ITERATIONS", overrides.get("scheduler_iterations", 0)),
            overrides.get("scheduler_iterations", 0),
        )
        scheduler_sleep_seconds = _to_int(
            getenv(
                "SCHEDULER_SLEEP_SECONDS",
                overrides.get("scheduler_sleep_seconds", 60),
            ),
            overrides.get("scheduler_sleep_seconds", 60),
        )
        window = _to_int(
            getenv("WINDOW", overrides.get("window", 20)),
            overrides.get("window", 20),
        )
        enabled = _to_bool(
            getenv("ENABLED", overrides.get("enabled", True)),
            overrides.get("enabled", True),
        )
        # Rate limiter configuration
        capacity = _to_int(
            getenv("CAPACITY", overrides.get("capacity", 100)),
            overrides.get("capacity", 100),
        )
        refill_rate = _to_float(
            getenv("REFILL_RATE", overrides.get("refill_rate", 10.0)),
            overrides.get("refill_rate", 10.0),
        )
        queue_timeout = _to_float(
            getenv("QUEUE_TIMEOUT", overrides.get("queue_timeout", 30.0)),
            overrides.get("queue_timeout", 30.0),
        )
        # AI-AGENT-REF: ML model configuration and halt file from env
        ml_model_path = getenv("AI_TRADER_MODEL_PATH", overrides.get("ml_model_path"))
        ml_model_module = getenv("AI_TRADER_MODEL_MODULE", overrides.get("ml_model_module"))
        halt_file = getenv("HALT_FILE", overrides.get("halt_file"))

        # include any other fields you already parse in from_env
        # and propagate remaining overrides into kwargs
        excluded_keys = {
            "conf_threshold",
            "buy_threshold",
            "confirmation_count",
            "enable_finbert",
            "capital_cap",
            "dollar_risk_limit",
            "max_position_size",
            "signal_confirm_bars",
            "delta_hold",
            "min_confidence",
            "trading_mode",
            "alpaca_base_url",
            "sleep_interval",
            "max_retries",
            "backoff_factor",
            "max_backoff_interval",
            "pct",
            "MODEL_PATH",
            "scheduler_iterations",
            "scheduler_sleep_seconds",
            "window",
            "enabled",
            "capacity",
            "refill_rate",
            "queue_timeout",
            "ml_model_path",
            "ml_model_module",
            "halt_file",
            "enable_sklearn",
            "intraday_lookback_minutes",
            "data_warmup_lookback_days",
            "disable_daily_retrain",
            "REGIME_MIN_ROWS",
            "signal_period",
        }
        cfg = cls(
            mode=mode,
            conf_threshold=conf_threshold,
            buy_threshold=buy_threshold,
            confirmation_count=confirmation_count,
            capital_cap=capital_cap,
            dollar_risk_limit=dollar_risk_limit,
            max_position_size=max_position_size,
            enable_finbert=enable_finbert,
            enable_sklearn=enable_sklearn,
            signal_confirm_bars=signal_confirm_bars,
            delta_hold=delta_hold,
            min_confidence=min_confidence,
            trading_mode=trading_mode,
            alpaca_base_url=alpaca_base_url,
            sleep_interval=sleep_interval,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            max_backoff_interval=max_backoff_interval,
            pct=pct,
            MODEL_PATH=MODEL_PATH,
            scheduler_iterations=scheduler_iterations,
            scheduler_sleep_seconds=scheduler_sleep_seconds,
            window=window,
            enabled=enabled,
            capacity=capacity,
            refill_rate=refill_rate,
            queue_timeout=queue_timeout,
            ml_model_path=ml_model_path,
            ml_model_module=ml_model_module,
            halt_file=halt_file,
            intraday_lookback_minutes=intraday_lookback_minutes,
            data_warmup_lookback_days=data_warmup_lookback_days,
            disable_daily_retrain=disable_daily_retrain,
            kelly_fraction=kelly_fraction,
            daily_loss_limit=daily_loss_limit,
            max_portfolio_risk=max_portfolio_risk,
            REGIME_MIN_ROWS=REGIME_MIN_ROWS,
            signal_period=signal_period,
            **{k: v for k, v in overrides.items() if k not in excluded_keys},
        )
        # set mode last so downstream logic sees it
        if mode is not None:
            cfg.mode = mode

        # Set legacy API keys for compatibility
        cfg.ALPACA_API_KEY = getenv("ALPACA_API_KEY", "test_key")
        cfg.ALPACA_SECRET_KEY = getenv("ALPACA_SECRET_KEY", "test_secret")

        # AI-AGENT-REF: sync with canonical settings
        s_rt = get_runtime_settings()
        cfg.ALPACA_API_KEY = getattr(s_rt, "alpaca_api_key", cfg.ALPACA_API_KEY)
        cfg.ALPACA_SECRET_KEY = (
            _secret_to_str(getattr(s_rt, "alpaca_secret_key", cfg.ALPACA_SECRET_KEY))
            or cfg.ALPACA_SECRET_KEY
        )
        cfg.ALPACA_BASE_URL = getattr(s_rt, "alpaca_base_url", cfg.alpaca_base_url)
        cfg.TRADING_MODE = getattr(s_rt, "bot_mode", cfg.trading_mode)
        cfg.trading_mode = cfg.TRADING_MODE  # AI-AGENT-REF: keep internal field synced
        cfg.TRADE_LOG_FILE = getattr(s_rt, "trade_log_file", TRADE_LOG_FILE)
        cfg.TIMEZONE = getattr(s_rt, "timezone", "UTC")
        cfg.MAX_DRAWDOWN_THRESHOLD = get_max_drawdown_threshold()
        cfg.max_drawdown_threshold = cfg.MAX_DRAWDOWN_THRESHOLD
        cfg.DAILY_LOSS_LIMIT = daily_loss_limit
        cfg.daily_loss_limit = cfg.DAILY_LOSS_LIMIT
        cfg.CAPITAL_CAP = capital_cap
        cfg.capital_cap = cfg.CAPITAL_CAP
        cfg.DOLLAR_RISK_LIMIT = dollar_risk_limit
        cfg.dollar_risk_limit = cfg.DOLLAR_RISK_LIMIT
        cfg.MAX_POSITION_SIZE = max_position_size
        cfg.max_position_size = cfg.MAX_POSITION_SIZE
        cfg.KELLY_FRACTION = cfg.kelly_fraction
        cfg.CONF_THRESHOLD = cfg.conf_threshold
        cfg.MAX_PORTFOLIO_RISK = cfg.max_portfolio_risk
        cfg.LIMIT_ORDER_SLIPPAGE = cfg.limit_order_slippage
        cfg.POV_SLICE_PCT = cfg.pov_slice_pct
        cfg.ENTRY_START_OFFSET_MIN = cfg.entry_start_offset_min
        cfg.ENTRY_END_OFFSET_MIN = cfg.entry_end_offset_min
        cfg.NEWS_API_KEY = getattr(s_rt, "news_api_key", None)
        cfg.SYSTEM_HEALTH_CHECK_INTERVAL = _to_int(
            getattr(s_rt, "system_health_check_interval", 60), 60
        )
        cfg.SYSTEM_HEALTH_ALERT_THRESHOLD = _to_int(
            getattr(s_rt, "system_health_alert_threshold", 3), 3
        )
        cfg.SYSTEM_HEALTH_EXPORT_ENABLED = _to_bool(
            getattr(s_rt, "system_health_export_enabled", False), False
        )
        cfg.ORDER_MAX_RETRY_ATTEMPTS = _to_int(getattr(s_rt, "order_max_retry_attempts", 3), 3)
        cfg.ORDER_TIMEOUT_SECONDS = _to_int(getattr(s_rt, "order_timeout_seconds", 30), 30)
        cfg.ORDER_STALE_CLEANUP_INTERVAL = _to_int(
            getattr(s_rt, "order_stale_cleanup_interval", 600), 600
        )
        cfg.ORDER_FILL_RATE_TARGET = _to_float(getattr(s_rt, "order_fill_rate_target", 0.95), 0.95)
        cfg.SENTIMENT_SUCCESS_RATE_TARGET = _to_float(
            getattr(s_rt, "sentiment_success_rate_target", 0.90), 0.90
        )

        if cfg.mode == "conservative":
            if k_env is None:
                cfg.kelly_fraction = 0.25
            if c_env is None:
                cfg.conf_threshold = 0.85
        elif cfg.mode == "balanced":
            if k_env is None:
                cfg.kelly_fraction = 0.60
            if c_env is None:
                cfg.conf_threshold = 0.75
        elif cfg.mode == "aggressive":
            if k_env is None:
                cfg.kelly_fraction = 0.75
            if c_env is None:
                cfg.conf_threshold = 0.65

        cfg.KELLY_FRACTION = cfg.kelly_fraction
        cfg.CONF_THRESHOLD = cfg.conf_threshold

        s_cfg = get_config_settings()

        # --- Optional providers / flags ---
        cfg.FINNHUB_API_KEY = getattr(s_cfg, "finnhub_api_key", None)
        cfg.DISABLE_DAILY_RETRAIN = bool(getattr(s_cfg, "disable_daily_retrain", False))

        # --- Reproducibility ---
        cfg.SEED = get_seed_int(42)

        # --- Audit / logging ---
        cfg.TRADE_AUDIT_DIR = os.getenv("TRADE_AUDIT_DIR", "logs/audit")
        cfg._CONFIG_LOGGED = False
        cfg._LOCK_TIMEOUT = 30

        # --- Liquidity controls (defaults aligned with tests) ---
        cfg.LIQUIDITY_SPREAD_THRESHOLD = float(os.getenv("LIQUIDITY_SPREAD_THRESHOLD", "0.01"))
        cfg.LIQUIDITY_VOL_THRESHOLD = float(os.getenv("LIQUIDITY_VOL_THRESHOLD", "0.0"))
        cfg.LIQUIDITY_REDUCTION_AGGRESSIVE = float(
            os.getenv("LIQUIDITY_REDUCTION_AGGRESSIVE", "0.75")
        )
        cfg.LIQUIDITY_REDUCTION_MODERATE = float(os.getenv("LIQUIDITY_REDUCTION_MODERATE", "0.90"))

        # --- Meta-learning bootstrap (tests expect presence + values) ---
        cfg.META_LEARNING_BOOTSTRAP_ENABLED = bool(
            os.getenv("META_LEARNING_BOOTSTRAP_ENABLED", "1") not in ("0", "false", "False")
        )
        cfg.META_LEARNING_MIN_TRADES_REDUCED = int(
            os.getenv("META_LEARNING_MIN_TRADES_REDUCED", "10")
        )
        cfg.META_LEARNING_BOOTSTRAP_WIN_RATE = float(
            os.getenv("META_LEARNING_BOOTSTRAP_WIN_RATE", "0.66")
        )

        return cfg

    def to_dict(self, safe=True):
        """
        Export configuration as dictionary.

        Args:
            safe: If True, redact secrets (API keys)

        Returns:
            Dict containing configuration values
        """
        config_dict = {}
        for attr in dir(self):
            if not attr.startswith("_") and not callable(getattr(self, attr)):
                value = getattr(self, attr)
                # Redact secrets if safe=True
                if safe and ("API_KEY" in attr or "SECRET" in attr):
                    config_dict[attr] = "***REDACTED***"
                else:
                    config_dict[attr] = value
        return config_dict

    def validate_environment(self) -> None:  # AI-AGENT-REF: runtime settings validation
        missing = []
        if not getattr(self, "ALPACA_API_KEY", None):
            missing.append("ALPACA_API_KEY")
        if not getattr(self, "ALPACA_SECRET_KEY", None):
            missing.append("ALPACA_SECRET_KEY")
        if getattr(self, "enable_finbert", False) and not getattr(self, "NEWS_API_KEY", None):
            missing.append("NEWS_API_KEY")
        if missing:
            raise RuntimeError(f"Missing required settings: {', '.join(missing)}")

    @classmethod
    def from_optimization(cls, params: dict[str, Any]):
        cfg = cls()
        for k, v in params.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        cfg.KELLY_FRACTION = cfg.kelly_fraction
        cfg.CONF_THRESHOLD = cfg.conf_threshold
        cfg.DAILY_LOSS_LIMIT = cfg.daily_loss_limit
        cfg.CAPITAL_CAP = cfg.capital_cap
        cfg.DOLLAR_RISK_LIMIT = cfg.dollar_risk_limit
        cfg.MAX_POSITION_SIZE = cfg.max_position_size
        cfg.LIMIT_ORDER_SLIPPAGE = cfg.limit_order_slippage
        cfg.POV_SLICE_PCT = cfg.pov_slice_pct
        cfg.ENTRY_START_OFFSET_MIN = cfg.entry_start_offset_min
        cfg.ENTRY_END_OFFSET_MIN = cfg.entry_end_offset_min
        return cfg

    def get_legacy_params(self) -> LegacyParams:
        """Return a legacy-style parameter map."""  # AI-AGENT-REF: robust legacy export
        from ai_trading.settings import (
            get_buy_threshold,
            get_capital_cap,
            get_conf_threshold,
            get_daily_loss_limit,
            get_disaster_dd_limit,
            get_dollar_risk_limit,
            get_max_drawdown_threshold,
            get_max_portfolio_positions,
            get_max_trades_per_day,
            get_max_trades_per_hour,
            get_portfolio_drift_threshold,
            get_position_size_min_usd,
            get_rebalance_interval_min,
            get_sector_exposure_cap,
            get_seed_int,
            get_trade_cooldown_min,
            get_volume_threshold,
        )

        def _get(name: str, default):
            return getattr(self, name, default)

        params: LegacyParams = {
            "KELLY_FRACTION": _get("kelly_fraction", 0.60),
            "CONF_THRESHOLD": _get("conf_threshold", get_conf_threshold()),
            "CONFIRMATION_COUNT": _get("confirmation_count", 3),
            "LOOKBACK_DAYS": _get("lookback_days", 60),
            "MIN_SIGNAL_STRENGTH": _get("min_signal_strength", 0.0),
            "STOP_LOSS": _get("stop_loss", 0.02),
            "TAKE_PROFIT": _get("take_profit", 0.04),
            "TAKE_PROFIT_FACTOR": _get("take_profit_factor", 2.0),
            "TRAILING_FACTOR": _get("trailing_factor", 1.0),
            "ENTRY_START_OFFSET_MIN": _get("entry_start_offset_min", 0),
            "ENTRY_END_OFFSET_MIN": _get("entry_end_offset_min", 390),
            "DAILY_LOSS_LIMIT": _get("daily_loss_limit", get_daily_loss_limit()),
            "MAX_DRAWDOWN_THRESHOLD": _get("max_drawdown_threshold", get_max_drawdown_threshold()),
            "PORTFOLIO_DRIFT_THRESHOLD": _get(
                "portfolio_drift_threshold", get_portfolio_drift_threshold()
            ),
            "DOLLAR_RISK_LIMIT": _get("dollar_risk_limit", get_dollar_risk_limit()),
            "CAPITAL_CAP": _get("capital_cap", get_capital_cap()),
            "SECTOR_EXPOSURE_CAP": _get("sector_exposure_cap", get_sector_exposure_cap()),
            "MAX_PORTFOLIO_POSITIONS": _get(
                "max_portfolio_positions", get_max_portfolio_positions()
            ),
            "DISASTER_DD_LIMIT": _get("disaster_dd_limit", get_disaster_dd_limit()),
            "REBALANCE_INTERVAL_MIN": _get("rebalance_interval_min", get_rebalance_interval_min()),
            "TRADE_COOLDOWN_MIN": _get("trade_cooldown_min", get_trade_cooldown_min()),
            "MAX_TRADES_PER_HOUR": _get("max_trades_per_hour", get_max_trades_per_hour()),
            "MAX_TRADES_PER_DAY": _get("max_trades_per_day", get_max_trades_per_day()),
            "BUY_THRESHOLD": _get("buy_threshold", get_buy_threshold()),
            "POSITION_SIZE_MIN_USD": _get("position_size_min_usd", get_position_size_min_usd()),
            "VOLUME_THRESHOLD": _get("volume_threshold", get_volume_threshold()),
            "SEED": _get("seed", get_seed_int()),
        }

        for optional_key in ("LIMIT_ORDER_SLIPPAGE", "SCALING_FACTOR", "POV_SLICE_PCT"):
            attr = optional_key.lower()
            if hasattr(self, attr):
                params[optional_key] = getattr(self, attr)

        return params


def build_legacy_params_from_config(cfg: "TradingConfig") -> dict[str, float | int]:
    """Pure function fallback to build legacy params."""  # AI-AGENT-REF: method lookup guard
    return cfg.get_legacy_params() if hasattr(cfg, "get_legacy_params") else {}


def validate_env_vars():
    """Validate required environment variables."""
    missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing and not TESTING:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


def _resolve_alpaca_env() -> tuple[str | None, str | None, str]:
    """
    Resolve Alpaca credentials and base URL from environment.

    Preference order:
      1) ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
      2) APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL

    Returns:
        (api_key, secret_key, base_url) where:
          - api_key, secret_key may be None if not provided
          - base_url is always set, defaulting to the paper endpoint

    This function MUST NOT raise just because credentials are absent; tests
    import without credentials and expect a clean import path.
    """
    # Primary schema
    a_key = os.getenv("ALPACA_API_KEY")
    a_secret = os.getenv("ALPACA_SECRET_KEY")
    a_url = os.getenv("ALPACA_BASE_URL")

    # Fallback schema (legacy APCA_*)
    apca_key = os.getenv("APCA_API_KEY_ID")
    apca_secret = os.getenv("APCA_API_SECRET_KEY")
    apca_url = os.getenv("APCA_API_BASE_URL")

    # Prefer ALPACA_* if both key and secret are present (even if APCA_* exist)
    if a_key and a_secret:
        base_url = a_url or "https://paper-api.alpaca.markets"
        return a_key, a_secret, base_url

    # Otherwise, if APCA_* are present, use them
    if apca_key and apca_secret:
        base_url = apca_url or "https://paper-api.alpaca.markets"
        return apca_key, apca_secret, base_url

    # Credentials absent; return None for keys but still provide a usable base_url
    base_url = a_url or apca_url or "https://paper-api.alpaca.markets"
    return None, None, base_url


def _warn_duplicate_env_keys() -> None:
    """Warn about potentially risky duplicate environment keys."""
    risky_duplicates = [
        ("ALPACA_API_KEY", "APCA_API_KEY_ID"),
        ("ALPACA_SECRET_KEY", "APCA_API_SECRET_KEY"),
        ("ALPACA_BASE_URL", "APCA_API_BASE_URL"),
    ]

    for alpaca_key, apca_key in risky_duplicates:
        alpaca_val = os.getenv(alpaca_key)
        apca_val = os.getenv(apca_key)

        if alpaca_val and apca_val and alpaca_val != apca_val:
            logger.warning(
                "Conflicting environment variables detected: %s and %s have different values. "
                "Using %s (ALPACA_* takes precedence)",
                alpaca_key,
                apca_key,
                alpaca_key,
            )


# Re-export settings components for direct import
from .settings import get_settings  # AI-AGENT-REF: avoid direct Settings import


def validate_alpaca_credentials() -> None:
    """Ensure required Alpaca credentials are present (settings-driven)."""
    if TESTING:
        return
    get_settings().require_alpaca_or_raise()


def log_config(vars_list):
    """Log configuration variables."""
    for var in vars_list:
        value = os.getenv(var, "NOT_SET")
        if "KEY" in var or "SECRET" in var:
            value = "***MASKED***" if value != "NOT_SET" else value
        logger.info(f"Config: {var}={value}")


def reload_env(env_file: str | os.PathLike[str] | None = None) -> None:
    """
    (Re)load environment variables from a .env file with override=True.

    Args:
        env_file: Optional explicit path to a .env file. If None, loads the
                  default search chain (cwd, working tree root, etc.).
    """
    # For systemd compatibility: don't override variables that have been explicitly cleared
    # Store current state of key variables that might be intentionally unset
    current_alpaca_vars = {
        key: os.getenv(key)
        for key in [
            "ALPACA_API_KEY",
            "ALPACA_SECRET_KEY",
            "ALPACA_BASE_URL",
            "APCA_API_KEY_ID",
            "APCA_API_SECRET_KEY",
            "APCA_API_BASE_URL",
        ]
    }

    if env_file:
        env_path = Path(env_file)
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        # Check if we should load default .env - don't override test values
        should_load_default = True
        for key, value in current_alpaca_vars.items():
            if value and ("test" in value.lower() or "_from_env" in value.lower()):
                should_load_default = False
                break

        if should_load_default:
            load_dotenv(override=True)

    # Restore None values for variables that were intentionally cleared
    for key, value in current_alpaca_vars.items():
        if value is None and key in os.environ:
            del os.environ[key]


class TradingConfig(BaseModel):
    """Trading parameters with mode-based defaults."""  # AI-AGENT-REF: unify config schema

    model_config = ConfigDict(extra="allow")

    # Core risk/trade fields
    daily_loss_limit: float = 0.03
    conf_threshold: float = 0.75
    kelly_fraction: float = 0.60
    slow_period: int = 21
    confirmation_count: int = 3
    capital_cap: float = Field(0.25, ge=0, le=1)  # AI-AGENT-REF: bounded risk cap
    dollar_risk_limit: float = Field(0.05, ge=0, le=1)  # AI-AGENT-REF: bounded dollar risk
    max_portfolio_risk: float = 0.10  # AI-AGENT-REF: portfolio risk cap
    max_position_size: float = 8000.0  # AI-AGENT-REF: default max position
    take_profit_factor: float = 2.0  # AI-AGENT-REF: reward multiple
    buy_threshold: float = Field(0.60, ge=0, le=1)  # AI-AGENT-REF: min buy confidence
    lookback_days: int = 60  # AI-AGENT-REF: history window
    min_confidence: float = Field(0.60, ge=0, le=1)  # AI-AGENT-REF: floor for ML signals
    signal_confirmation_bars: int = 2  # AI-AGENT-REF: bars to confirm signal
    delta_threshold: float = 0.02  # AI-AGENT-REF: price delta trigger
    max_drawdown_threshold: float = Field(0.08, ge=0, le=1)  # AI-AGENT-REF: drawdown guard
    take_profit: float = 0.04
    stop_loss: float = 0.02
    trailing_factor: float = 1.0
    limit_order_slippage: float = 0.001
    pov_slice_pct: float = 0.05
    max_slippage_bps: float = 25.0
    participation_rate: float = 0.10
    order_timeout_seconds: int = 30
    signal_period: int = 20
    fast_period: int = 12
    position_size_min_usd: float = 0.0
    portfolio_drift_threshold: float = 0.05
    sector_exposure_cap: float = 0.2
    max_portfolio_positions: int = 20
    disaster_dd_limit: float = 0.2
    rebalance_interval_min: int = 60
    trade_cooldown_min: int = 15
    max_trades_per_hour: int = 10
    max_trades_per_day: int = 100
    volume_threshold: int = 50000
    seed: int = 42
    entry_start_offset_min: int = 0
    entry_end_offset_min: int = 390

    @field_validator("kelly_fraction", "conf_threshold", mode="after")
    @classmethod
    def _clamp_unit_interval(cls, v: float):  # AI-AGENT-REF: ensure 0-1 bounds
        if not 0 <= v <= 1:
            raise ValueError("must be between 0 and 1")
        return v

    @model_validator(mode="after")
    def _risk_relations(self):  # AI-AGENT-REF: cross-field risk checks
        if self.capital_cap < self.dollar_risk_limit:
            raise ValueError("capital_cap must be >= dollar_risk_limit")
        if self.take_profit_factor < 1.0:
            raise ValueError("take_profit_factor must be >= 1.0")
        return self

    @classmethod
    def from_env(
        cls,
        mode: Literal["conservative", "balanced", "aggressive"] | None = None,
        **overrides,
    ) -> "TradingConfig":
        """Load configuration from environment with mode defaults."""  # AI-AGENT-REF
        mode = (mode or os.getenv("TRADING_MODE", "balanced")).lower()
        defaults = {"conservative": 0.85, "balanced": 0.75, "aggressive": 0.65}
        conf_threshold = defaults.get(mode, 0.75)
        conf_threshold = float(os.getenv("CONF_THRESHOLD", conf_threshold))
        if mode == "conservative":
            mode_defaults = {
                "kelly_fraction": 0.25,
                "daily_loss_limit": 0.03,
                "capital_cap": 0.20,
                "confirmation_count": 3,
                "take_profit_factor": 1.5,
                "max_position_size": 5000.0,
            }
        elif mode == "aggressive":
            mode_defaults = {
                "kelly_fraction": 0.75,
                "daily_loss_limit": 0.08,
                "capital_cap": 0.30,
                "confirmation_count": 1,
                "take_profit_factor": 2.5,
                "max_position_size": 12000.0,
            }
        else:
            mode_defaults = {
                "kelly_fraction": 0.60,
                "daily_loss_limit": 0.05,
                "capital_cap": 0.25,
                "confirmation_count": 2,
                "take_profit_factor": 1.8,
                "max_position_size": 8000.0,
            }
        base = cls().model_copy(update=mode_defaults)
        base.conf_threshold = conf_threshold  # AI-AGENT-REF: mode-specific confidence
        if env_val := os.getenv("KELLY_FRACTION"):
            base.kelly_fraction = float(env_val)
        # AI-AGENT-REF: sync additional risk/cadence knobs with env/settings
        from ai_trading.settings import (
            get_buy_threshold,
            get_capital_cap,
            get_daily_loss_limit,
            get_dollar_risk_limit,
            get_max_drawdown_threshold,
            get_max_portfolio_positions,
            get_max_trades_per_day,
            get_max_trades_per_hour,
            get_portfolio_drift_threshold,
            get_position_size_min_usd,
            get_rebalance_interval_min,
            get_sector_exposure_cap,
            get_seed_int,
            get_trade_cooldown_min,
            get_volume_threshold,
        )

        def _env_or_get(name: str, getter, cast=float):
            val = os.getenv(name)
            return cast(val) if val is not None else cast(getter())

        base.daily_loss_limit = _env_or_get("DAILY_LOSS_LIMIT", get_daily_loss_limit)
        base.capital_cap = _env_or_get("CAPITAL_CAP", get_capital_cap)
        base.dollar_risk_limit = _env_or_get("DOLLAR_RISK_LIMIT", get_dollar_risk_limit)
        base.max_drawdown_threshold = _env_or_get(
            "MAX_DRAWDOWN_THRESHOLD", get_max_drawdown_threshold
        )

        base.portfolio_drift_threshold = _env_or_get(
            "PORTFOLIO_DRIFT_THRESHOLD", get_portfolio_drift_threshold
        )
        base.sector_exposure_cap = _env_or_get("SECTOR_EXPOSURE_CAP", get_sector_exposure_cap)
        base.max_portfolio_positions = _env_or_get(
            "MAX_PORTFOLIO_POSITIONS", get_max_portfolio_positions, int
        )

        base.rebalance_interval_min = _env_or_get(
            "REBALANCE_INTERVAL_MIN", get_rebalance_interval_min, int
        )
        base.trade_cooldown_min = _env_or_get("TRADE_COOLDOWN_MIN", get_trade_cooldown_min, int)
        base.max_trades_per_hour = _env_or_get("MAX_TRADES_PER_HOUR", get_max_trades_per_hour, int)
        base.max_trades_per_day = _env_or_get("MAX_TRADES_PER_DAY", get_max_trades_per_day, int)

        base.position_size_min_usd = _env_or_get("POSITION_SIZE_MIN_USD", get_position_size_min_usd)
        base.volume_threshold = _env_or_get("VOLUME_THRESHOLD", get_volume_threshold)
        base.buy_threshold = _env_or_get("BUY_THRESHOLD", get_buy_threshold)
        base.lookback_days = int(os.getenv("LOOKBACK_DAYS") or base.lookback_days)
        base.seed = int(os.getenv("SEED") or get_seed_int())
        if overrides:
            base = base.model_copy(update=overrides)
        return base

    @classmethod
    def from_optimization(cls, params: dict[str, Any]) -> "TradingConfig":
        """Build from optimization parameters."""  # AI-AGENT-REF
        base = cls.from_env(params.get("mode", "balanced"))
        return base.model_copy(update={k: v for k, v in params.items() if k in base.model_fields})

    def to_dict(self) -> dict:
        return self.model_dump()  # AI-AGENT-REF: include extras

    def get_legacy_params(self) -> LegacyParams:
        """Return a legacy-style parameter map."""  # AI-AGENT-REF: robust legacy export
        from ai_trading.settings import (
            get_buy_threshold,
            get_capital_cap,
            get_conf_threshold,
            get_daily_loss_limit,
            get_disaster_dd_limit,
            get_dollar_risk_limit,
            get_max_drawdown_threshold,
            get_max_portfolio_positions,
            get_max_trades_per_day,
            get_max_trades_per_hour,
            get_portfolio_drift_threshold,
            get_position_size_min_usd,
            get_rebalance_interval_min,
            get_sector_exposure_cap,
            get_seed_int,
            get_trade_cooldown_min,
            get_volume_threshold,
        )

        def _get(name: str, default):
            return getattr(self, name, default)

        params: LegacyParams = {
            "KELLY_FRACTION": _get("kelly_fraction", 0.60),
            "CONF_THRESHOLD": _get("conf_threshold", get_conf_threshold()),
            "CONFIRMATION_COUNT": _get("confirmation_count", 3),
            "LOOKBACK_DAYS": _get("lookback_days", 60),
            "MIN_SIGNAL_STRENGTH": _get("min_signal_strength", 0.0),
            "STOP_LOSS": _get("stop_loss", 0.02),
            "TAKE_PROFIT": _get("take_profit", 0.04),
            "TAKE_PROFIT_FACTOR": _get("take_profit_factor", 2.0),
            "TRAILING_FACTOR": _get("trailing_factor", 1.0),
            "ENTRY_START_OFFSET_MIN": _get("entry_start_offset_min", 0),
            "ENTRY_END_OFFSET_MIN": _get("entry_end_offset_min", 390),
            "DAILY_LOSS_LIMIT": _get("daily_loss_limit", get_daily_loss_limit()),
            "MAX_DRAWDOWN_THRESHOLD": _get("max_drawdown_threshold", get_max_drawdown_threshold()),
            "PORTFOLIO_DRIFT_THRESHOLD": _get(
                "portfolio_drift_threshold", get_portfolio_drift_threshold()
            ),
            "DOLLAR_RISK_LIMIT": _get("dollar_risk_limit", get_dollar_risk_limit()),
            "CAPITAL_CAP": _get("capital_cap", get_capital_cap()),
            "SECTOR_EXPOSURE_CAP": _get("sector_exposure_cap", get_sector_exposure_cap()),
            "MAX_PORTFOLIO_POSITIONS": _get(
                "max_portfolio_positions", get_max_portfolio_positions()
            ),
            "DISASTER_DD_LIMIT": _get("disaster_dd_limit", get_disaster_dd_limit()),
            "REBALANCE_INTERVAL_MIN": _get("rebalance_interval_min", get_rebalance_interval_min()),
            "TRADE_COOLDOWN_MIN": _get("trade_cooldown_min", get_trade_cooldown_min()),
            "MAX_TRADES_PER_HOUR": _get("max_trades_per_hour", get_max_trades_per_hour()),
            "MAX_TRADES_PER_DAY": _get("max_trades_per_day", get_max_trades_per_day()),
            "BUY_THRESHOLD": _get("buy_threshold", get_buy_threshold()),
            "POSITION_SIZE_MIN_USD": _get("position_size_min_usd", get_position_size_min_usd()),
            "VOLUME_THRESHOLD": _get("volume_threshold", get_volume_threshold()),
            "SEED": _get("seed", get_seed_int()),
        }

        for optional_key in ("LIMIT_ORDER_SLIPPAGE", "SCALING_FACTOR", "POV_SLICE_PCT"):
            attr = optional_key.lower()
            if hasattr(self, attr):
                params[optional_key] = getattr(self, attr)

        return params
