"""Security enhancements for production trading platform.

Provides secure configuration with encryption for API keys,
comprehensive audit logging for compliance, and prevents
API key exposure in logs and error messages.

AI-AGENT-REF: Security enhancements for institutional-grade trading
"""
from __future__ import annotations
import base64
import json
import logging
import os
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    _CRYPTOGRAPHY_AVAILABLE = True
except (ValueError, TypeError):
    _CRYPTOGRAPHY_AVAILABLE = False

    class Fernet:

        def __init__(self, *args, **kwargs):
            pass

        def encrypt(self, data: bytes) -> bytes:
            return data

        def decrypt(self, token: bytes) -> bytes:
            return token
    hashes = PBKDF2HMAC = None
logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security event severity levels."""
    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'
    CRITICAL = 'critical'

class AuditEventType(Enum):
    """Types of audit events."""
    CONFIG_ACCESS = 'config_access'
    API_KEY_ACCESS = 'api_key_access'
    TRADE_EXECUTION = 'trade_execution'
    POSITION_CHANGE = 'position_change'
    RISK_VIOLATION = 'risk_violation'
    DATA_ACCESS = 'data_access'
    SYSTEM_CHANGE = 'system_change'
    AUTHENTICATION = 'authentication'
    ERROR_OCCURRENCE = 'error_occurrence'

@dataclass
class AuditEvent:
    """Audit event for compliance logging."""
    timestamp: datetime
    event_type: AuditEventType
    severity: SecurityLevel
    user_id: str
    action: str
    resource: str
    details: dict[str, Any]
    session_id: str | None = None
    ip_address: str | None = None
    result: str = 'success'

class SecureConfig:
    """Secure configuration management with encryption."""

    def __init__(self, master_key: str | None=None):
        self.logger = logging.getLogger(__name__)
        self._master_key = master_key or self._generate_master_key()
        self._fernet = self._init_encryption() if _CRYPTOGRAPHY_AVAILABLE else None
        self._config_cache: dict[str, Any] = {}
        self._sensitive_keys = {'ALPACA_API_KEY', 'ALPACA_SECRET_KEY', 'WEBHOOK_SECRET', 'NEWS_API_KEY', 'FINNHUB_API_KEY', 'FUNDAMENTAL_API_KEY', 'IEX_API_TOKEN', 'DATABASE_URL', 'REDIS_URL'}
        self.audit_logger = self._setup_audit_logging()
        self.logger.info('SecureConfig initialized with encryption support')

    def _generate_master_key(self) -> str:
        """Generate a secure master key for encryption."""
        env_key = os.getenv('MASTER_ENCRYPTION_KEY')
        if env_key:
            return env_key
        key = secrets.token_urlsafe(32)
        self.logger.warning('Generated new master encryption key - save MASTER_ENCRYPTION_KEY to environment')
        return key

    def _init_encryption(self) -> Fernet | None:
        """Initialize encryption cipher."""
        if not _CRYPTOGRAPHY_AVAILABLE:
            self.logger.warning('Cryptography library not available - encryption disabled')
            return None
        try:
            master_key_bytes = self._master_key.encode()
            salt = b'ai_trading_salt_2024'
            kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000)
            key = base64.urlsafe_b64encode(kdf.derive(master_key_bytes))
            return Fernet(key)
        except (ValueError, TypeError) as e:
            self.logger.error(f'Failed to initialize encryption: {e}')
            return None

    def _setup_audit_logging(self) -> logging.Logger:
        """Setup dedicated audit logging."""
        audit_logger = logging.getLogger('ai_trading.audit')
        audit_logger.setLevel(logging.INFO)
        if not audit_logger.handlers:
            log_dir = Path('logs')
            log_dir.mkdir(exist_ok=True)
            audit_file = log_dir / 'audit.log'
            handler = logging.FileHandler(audit_file)
            formatter = logging.Formatter('{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": %(message)s}')
            handler.setFormatter(formatter)
            audit_logger.addHandler(handler)
        return audit_logger

    def encrypt_value(self, value: str) -> str:
        """Encrypt a sensitive value."""
        if not self._fernet or not value:
            return value
        try:
            encrypted = self._fernet.encrypt(value.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except (ValueError, TypeError) as e:
            self.logger.error(f'Encryption failed: {e}')
            return value

    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a sensitive value."""
        if not self._fernet or not encrypted_value:
            return encrypted_value
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode())
            decrypted = self._fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
        except (ValueError, TypeError) as e:
            self.logger.error(f'Decryption failed: {e}')
            return encrypted_value

    def get_secure_config(self, key: str, default: Any=None) -> Any:
        """Get configuration value with audit logging."""
        self.log_audit_event(event_type=AuditEventType.CONFIG_ACCESS, action='get_config', resource=key, details={'key': key, 'has_default': default is not None})
        if key in self._config_cache:
            value = self._config_cache[key]
        else:
            value = os.getenv(key, default)
            self._config_cache[key] = value
        if key in self._sensitive_keys and value and isinstance(value, str):
            if self._is_encrypted(value):
                value = self.decrypt_value(value)
        return value

    def set_secure_config(self, key: str, value: Any, encrypt: bool=None) -> None:
        """Set configuration value with optional encryption."""
        should_encrypt = encrypt if encrypt is not None else key in self._sensitive_keys
        if should_encrypt and isinstance(value, str):
            encrypted_value = self.encrypt_value(value)
            self._config_cache[key] = encrypted_value
        else:
            self._config_cache[key] = value
        self.log_audit_event(event_type=AuditEventType.CONFIG_ACCESS, action='set_config', resource=key, details={'key': key, 'encrypted': should_encrypt})

    def _is_encrypted(self, value: str) -> bool:
        """Check if a value appears to be encrypted."""
        if not value or len(value) < 20:
            return False
        try:
            base64.urlsafe_b64decode(value.encode())
            return True
        except (ValueError, TypeError):
            return False

    def mask_sensitive_data(self, data: str | dict | list) -> str | dict | list:
        """Mask sensitive data for logging."""
        if isinstance(data, str):
            return self._mask_string(data)
        elif isinstance(data, dict):
            return {k: self.mask_sensitive_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.mask_sensitive_data(item) for item in data]
        else:
            return data

    def _mask_string(self, value: str) -> str:
        """Mask sensitive string values."""
        if not value or len(value) < 4:
            return '***'
        if any((pattern in value.upper() for pattern in ['KEY', 'SECRET', 'TOKEN', 'PASSWORD'])):
            return f'{value[:4]}***{value[-4:]}'
        if len(value) > 20 and value.replace('-', '').replace('_', '').isalnum():
            return f'{value[:4]}***{value[-4:]}'
        return value

    def log_audit_event(self, event_type: AuditEventType, action: str, resource: str, details: dict[str, Any] | None=None, severity: SecurityLevel=SecurityLevel.INFO, user_id: str='system', result: str='success') -> None:
        """Log audit event for compliance."""
        event = AuditEvent(timestamp=datetime.now(UTC), event_type=event_type, severity=severity, user_id=user_id, action=action, resource=resource, details=self.mask_sensitive_data(details or {}), result=result)
        audit_data = {'timestamp': event.timestamp.isoformat(), 'event_type': event.event_type.value, 'severity': event.severity.value, 'user_id': event.user_id, 'action': event.action, 'resource': event.resource, 'details': event.details, 'result': event.result}
        self.audit_logger.info(json.dumps(audit_data))
        if event.severity in [SecurityLevel.ERROR, SecurityLevel.CRITICAL]:
            self.logger.warning(f'Security event: {event.action} on {event.resource} - {event.result}')

class SafeLogger:
    """Logger wrapper that prevents API key exposure."""

    def __init__(self, logger: logging.Logger, secure_config: SecureConfig | None=None):
        self.logger = logger
        self.secure_config = secure_config or SecureConfig()
        self._sensitive_patterns = ['[A-Z0-9]{20,}', 'sk-[A-Za-z0-9]{20,}', '[A-Za-z0-9+/]{32,}={0,2}']

    def _clean_message(self, message: str) -> str:
        """Clean sensitive data from log message."""
        if not isinstance(message, str):
            message = str(message)
        return str(self.secure_config.mask_sensitive_data(message))

    def debug(self, message: str, *args, **kwargs) -> None:
        """Safe debug logging."""
        clean_message = self._clean_message(message)
        self.logger.debug(clean_message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs) -> None:
        """Safe info logging."""
        clean_message = self._clean_message(message)
        self.logger.info(clean_message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        """Safe warning logging."""
        clean_message = self._clean_message(message)
        self.logger.warning(clean_message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs) -> None:
        """Safe error logging."""
        clean_message = self._clean_message(message)
        self.logger.error(clean_message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs) -> None:
        """Safe critical logging."""
        clean_message = self._clean_message(message)
        self.logger.critical(clean_message, *args, **kwargs)

class SecurityManager:
    """Central security management for the trading platform."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.secure_config = SecureConfig()
        self.safe_logger = SafeLogger(self.logger, self.secure_config)
        self._security_events: list[AuditEvent] = []
        self._failed_access_attempts = 0
        self._last_security_check = datetime.now(UTC)
        self.safe_logger.info('SecurityManager initialized')

    def get_api_key(self, service: str) -> str | None:
        """Securely retrieve API key for a service."""
        key_mapping = {'alpaca': 'ALPACA_API_KEY', 'alpaca_secret': 'ALPACA_SECRET_KEY', 'news': 'NEWS_API_KEY', 'finnhub': 'FINNHUB_API_KEY', 'fundamental': 'FUNDAMENTAL_API_KEY', 'iex': 'IEX_API_TOKEN'}
        env_key = key_mapping.get(service.lower())
        if not env_key:
            self.secure_config.log_audit_event(event_type=AuditEventType.API_KEY_ACCESS, action='get_api_key', resource=service, result='failure', severity=SecurityLevel.WARNING, details={'reason': 'unknown_service'})
            return None
        api_key = self.secure_config.get_secure_config(env_key)
        self.secure_config.log_audit_event(event_type=AuditEventType.API_KEY_ACCESS, action='get_api_key', resource=service, result='success' if api_key else 'failure', details={'has_key': bool(api_key)})
        return api_key

    def validate_api_key(self, service: str, key: str) -> bool:
        """Validate API key format and basic checks."""
        if not key or len(key) < 10:
            return False
        validation_rules = {'alpaca': lambda k: len(k) >= 20 and k.replace('-', '').isalnum(), 'news': lambda k: len(k) >= 32, 'finnhub': lambda k: len(k) >= 20}
        validator = validation_rules.get(service.lower())
        if validator:
            return validator(key)
        return len(key) >= 20 and any((c.isalnum() for c in key))

    def mask_sensitive_data(self, payload: dict) -> dict:
        """Mask sensitive fields via SecureConfig."""
        return self.secure_config.mask_sensitive_data(payload)

    def check_security_health(self) -> dict[str, Any]:
        """Perform security health check."""
        health_status = {'encryption_available': _CRYPTOGRAPHY_AVAILABLE, 'audit_logging_active': bool(self.secure_config.audit_logger.handlers), 'failed_access_attempts': self._failed_access_attempts, 'recent_security_events': len([e for e in self._security_events if e.timestamp > datetime.now(UTC) - timedelta(hours=24)]), 'last_security_check': self._last_security_check.isoformat()}
        critical_issues = []
        if not _CRYPTOGRAPHY_AVAILABLE:
            critical_issues.append('Encryption library not available')
        if self._failed_access_attempts > 10:
            critical_issues.append('High number of failed access attempts')
        health_status['critical_issues'] = critical_issues
        health_status['overall_health'] = 'healthy' if not critical_issues else 'degraded'
        self._last_security_check = datetime.now(UTC)
        return health_status

    def rotate_encryption_key(self) -> bool:
        """Rotate the master encryption key."""
        try:
            new_key = self._generate_new_key()
            old_config = self.secure_config
            self.secure_config = SecureConfig(new_key)
            for key, value in old_config._config_cache.items():
                if key in old_config._sensitive_keys:
                    decrypted = old_config.decrypt_value(value)
                    self.secure_config.set_secure_config(key, decrypted)
            self.secure_config.log_audit_event(event_type=AuditEventType.SYSTEM_CHANGE, action='rotate_encryption_key', resource='master_key', severity=SecurityLevel.INFO)
            return True
        except (ValueError, TypeError) as e:
            self.safe_logger.error(f'Failed to rotate encryption key: {e}')
            return False

    def _generate_new_key(self) -> str:
        """Generate a new secure encryption key."""
        return secrets.token_urlsafe(32)
_security_manager: SecurityManager | None = None

def get_security_manager() -> SecurityManager:
    """Get or create global security manager instance."""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager

def get_safe_logger(name: str) -> SafeLogger:
    """Get a safe logger that masks sensitive data."""
    base_logger = logging.getLogger(name)
    security_manager = get_security_manager()
    return SafeLogger(base_logger, security_manager.secure_config)

def mask_sensitive_data(payload: dict) -> dict:
    """Module-level wrapper used by tests: from security import mask_sensitive_data"""
    return SecurityManager().mask_sensitive_data(payload)

def log_security_event(event_type: AuditEventType, action: str, resource: str, **kwargs) -> None:
    """Convenience function to log security events."""
    security_manager = get_security_manager()
    security_manager.secure_config.log_audit_event(event_type=event_type, action=action, resource=resource, **kwargs)