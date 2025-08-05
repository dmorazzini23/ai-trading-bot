"""
Production configuration management for live trading.

This module provides secure configuration management, validation,
and runtime configuration changes with proper audit logging.
"""

import os
import json
import hashlib
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone
from pathlib import Path
import logging

# Use the centralized logger as per AGENTS.md
try:
    from logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


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
                "default": "paper"
            },
            
            # API configuration
            "ALPACA_API_KEY": {
                "type": str,
                "required": True,
                "min_length": 10,
                "security_check": True
            },
            "ALPACA_SECRET_KEY": {
                "type": str,
                "required": True,
                "min_length": 10,
                "security_check": True
            },
            "ALPACA_BASE_URL": {
                "type": str,
                "required": True,
                "allowed_values": [
                    "https://api.alpaca.markets",
                    "https://paper-api.alpaca.markets"
                ]
            },
            
            # Risk management
            "MAX_PORTFOLIO_POSITIONS": {
                "type": int,
                "min_value": 1,
                "max_value": 100,
                "default": 20
            },
            "LIMIT_ORDER_SLIPPAGE": {
                "type": float,
                "min_value": 0.0001,
                "max_value": 0.1,
                "default": 0.005
            },
            "DISASTER_DD_LIMIT": {
                "type": float,
                "min_value": 0.01,
                "max_value": 0.5,
                "default": 0.2
            },
            
            # Performance limits
            "VOLUME_THRESHOLD": {
                "type": int,
                "min_value": 1000,
                "max_value": 10000000,
                "default": 50000
            },
            "DOLLAR_RISK_LIMIT": {
                "type": float,
                "min_value": 0.001,
                "max_value": 0.1,
                "default": 0.05
            },
            
            # System configuration
            "FLASK_PORT": {
                "type": int,
                "min_value": 1024,
                "max_value": 65535,
                "default": 9000
            },
            "HEALTHCHECK_PORT": {
                "type": int,
                "min_value": 1024,
                "max_value": 65535,
                "default": 8081
            }
        }
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
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
            "invalid_values": []
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
                    result["invalid_values"].append({"key": key, "value": str(value), "errors": validation_errors})
        
        # Security checks
        security_issues = self._check_security(config)
        result["security_issues"] = security_issues
        if security_issues:
            result["warnings"].extend([f"Security issue: {issue}" for issue in security_issues])
        
        return result
    
    def _validate_value(self, key: str, value: Any, rules: Dict[str, Any]) -> List[str]:
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
        if isinstance(value, (int, float)):
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
    
    def _check_security(self, config: Dict[str, Any]) -> List[str]:
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
    
    def load_config(self, config_file: Optional[str] = None) -> Dict[str, Any]:
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
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Merge with environment variables
            config = self._merge_with_env(config)
            
            # Validate configuration
            validation_result = self.validator.validate_config(config)
            
            if not validation_result["valid"]:
                logger.error(f"Configuration validation failed: {validation_result['errors']}")
                if validation_result["missing_required"]:
                    logger.error(f"Missing required fields: {validation_result['missing_required']}")
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
    
    def save_config(self, config: Dict[str, Any], backup: bool = True) -> bool:
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
            with open(self.main_config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Update internal state
            old_hash = self.config_hash
            self.config_hash = self._calculate_config_hash(config)
            self.current_config = config
            
            # Log configuration change
            self._log_config_change("save", f"Configuration saved (hash: {old_hash} -> {self.config_hash})")
            
            logger.info("Configuration saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def update_config(self, updates: Dict[str, Any], reason: str = "Manual update") -> bool:
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
                logger.error(f"Configuration update validation failed: {validation_result['errors']}")
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
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.current_config.copy()
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for monitoring."""
        return {
            "hash": self.config_hash,
            "last_updated": self.config_history[-1]["timestamp"] if self.config_history else None,
            "trading_mode": self.current_config.get("TRADING_MODE", "unknown"),
            "num_changes": len(self.config_history),
            "validation_status": "valid" if self.validator.validate_config(self.current_config)["valid"] else "invalid"
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
    
    def _create_default_config(self) -> Dict[str, Any]:
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
            "HEALTHCHECK_PORT": 8081
        }
        
        # Save default configuration
        self.save_config(default_config)
        self._log_config_change("create", "Default configuration created")
        
        return default_config
    
    def _merge_with_env(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration with environment variables."""
        merged = config.copy()
        
        # Environment variables take precedence
        for key in self.validator.validation_rules.keys():
            env_value = os.environ.get(key)
            if env_value is not None:
                # Convert to appropriate type
                expected_type = self.validator.validation_rules[key].get("type", str)
                try:
                    if expected_type == bool:
                        merged[key] = env_value.lower() in ('true', '1', 'yes', 'on')
                    elif expected_type == int:
                        merged[key] = int(env_value)
                    elif expected_type == float:
                        merged[key] = float(env_value)
                    else:
                        merged[key] = env_value
                except ValueError:
                    logger.warning(f"Could not convert environment variable {key}={env_value} to {expected_type}")
        
        return merged
    
    def _calculate_config_hash(self, config: Dict[str, Any]) -> str:
        """Calculate hash of configuration for change detection."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def _detect_critical_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> List[str]:
        """Detect critical configuration changes."""
        critical_keys = [
            "TRADING_MODE",
            "ALPACA_BASE_URL",
            "ALPACA_API_KEY",
            "ALPACA_SECRET_KEY"
        ]
        
        changes = []
        for key in critical_keys:
            if old_config.get(key) != new_config.get(key):
                changes.append(key)
        
        return changes
    
    def _log_config_change(self, action: str, details: str):
        """Log configuration changes for audit."""
        timestamp = datetime.now(timezone.utc).isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "action": action,
            "details": details,
            "config_hash": self.config_hash,
            "user": os.environ.get("USER", "unknown")
        }
        
        self.config_history.append(log_entry)
        
        # Write to audit log file
        try:
            with open(self.audit_log_file, 'a') as f:
                f.write(f"{json.dumps(log_entry)}\n")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")


# Convenience function for getting validated configuration
def get_production_config() -> Dict[str, Any]:
    """Get validated production configuration."""
    config_manager = ConfigManager()
    return config_manager.load_config()