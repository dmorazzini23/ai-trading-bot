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
from typing import Any

from dotenv import load_dotenv

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
                validation_errors = self._validate_value(
                    key, value, self.validation_rules[key]
                )
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
            result["warnings"].extend(
                [f"Security issue: {issue}" for issue in security_issues]
            )

        return result

    def _validate_value(self, key: str, value: Any, rules: dict[str, Any]) -> list[str]:
        """Validate a single configuration value."""
        errors = []

        # Type validation
        expected_type = rules.get("type")
        if expected_type and not isinstance(value, expected_type):
            errors.append(
                f"Expected {expected_type.__name__}, got {type(value).__name__}"
            )
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
                logger.error(
                    f"Configuration validation failed: {validation_result['errors']}"
                )
                if validation_result["missing_required"]:
                    logger.error(
                        f"Missing required fields: {validation_result['missing_required']}"
                    )
                raise ValueError(
                    f"Invalid configuration: {validation_result['errors']}"
                )

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
                logger.error(
                    f"Cannot save invalid configuration: {validation_result['errors']}"
                )
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

    def update_config(
        self, updates: dict[str, Any], reason: str = "Manual update"
    ) -> bool:
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
            critical_changes = self._detect_critical_changes(
                self.current_config, new_config
            )
            if critical_changes:
                logger.warning(
                    f"Critical configuration changes detected: {critical_changes}"
                )
                self._log_config_change(
                    "critical_update", f"Critical changes: {critical_changes}"
                )

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
            self._log_config_change(
                "update", f"Runtime update: {updates} - Reason: {reason}"
            )

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
            "last_updated": (
                self.config_history[-1]["timestamp"] if self.config_history else None
            ),
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
MODEL_PATH = os.getenv(
    "MODEL_PATH", str(paths.DATA_DIR / "models" / "trained_model.pkl")
)
RL_MODEL_PATH = os.getenv(
    "RL_MODEL_PATH", str(paths.DATA_DIR / "models" / "rl_model.pkl")
)
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
class TradingConfig:
    def __init__(self):
        # Add default trading configuration attributes
        self.trailing_factor = 0.05
        self.kelly_fraction = 0.25
        self.max_position_size = 0.1
        self.stop_loss = 0.02
        self.take_profit = 0.06
        self.take_profit_factor = 0.06
        self.lookback_days = 30
        self.min_signal_strength = 0.6
        self.scaling_factor = 1.0
        self.limit_order_slippage = 0.001
        self.pov_slice_pct = 0.1
        self.daily_loss_limit = 0.05

    @classmethod
    def from_env(cls, mode="balanced"):
        """Create a TradingConfig from environment variables."""
        instance = cls()
        instance.mode = mode
        instance.ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "test_key")
        instance.ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "test_secret")
        return instance

    def get_legacy_params(self):
        """Return legacy parameters for backward compatibility."""
        return {
            "mode": getattr(self, "mode", "balanced"),
            "ALPACA_API_KEY": getattr(self, "ALPACA_API_KEY", "test_key"),
            "ALPACA_SECRET_KEY": getattr(self, "ALPACA_SECRET_KEY", "test_secret"),
            "trailing_factor": self.trailing_factor,
            "kelly_fraction": self.kelly_fraction,
            "max_position_size": self.max_position_size,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "lookback_days": self.lookback_days,
            "min_signal_strength": self.min_signal_strength,
            "daily_loss_limit": self.daily_loss_limit,
        }


def validate_env_vars():
    """Validate required environment variables."""
    missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing and not TESTING:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}"
        )


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
from .settings import Settings, get_settings


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
