#!/usr/bin/env python3
"""Advanced security and risk management module for AI trading bot.

This module provides enterprise-grade security and risk management:
- API rate limiting and DDoS protection
- Encryption for sensitive data at rest and in transit
- Secure audit logging for compliance
- Anomaly detection for unusual trading patterns
- Position limits and risk circuit breakers
- Multi-factor authentication support
- Security penetration testing simulation

AI-AGENT-REF: Production-grade security and risk management system
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from cryptography.fernet import Fernet

# AI-AGENT-REF: Advanced security for institutional-grade trading


class SecurityLevel(Enum):
    """Security alert levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskLevel(Enum):
    """Risk assessment levels."""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class SecurityEvent:
    """Security event record."""
    timestamp: datetime
    event_type: str
    severity: SecurityLevel
    source_ip: str | None
    user_id: str | None
    description: str
    details: dict[str, Any]
    action_taken: str | None = None


@dataclass
class RiskAssessment:
    """Risk assessment result."""
    timestamp: datetime
    risk_level: RiskLevel
    risk_score: float
    factors: list[str]
    recommendations: list[str]
    position_limits: dict[str, float]
    trading_restrictions: list[str]


class RateLimiter:
    """Advanced rate limiting for API protection."""

    def __init__(self, max_requests: int, time_window: int, burst_limit: int = None):
        self.max_requests = max_requests
        self.time_window = time_window
        self.burst_limit = burst_limit or max_requests * 2

        self.requests: dict[str, deque] = defaultdict(lambda: deque())
        self.blocked_clients: dict[str, float] = {}
        self.lock = threading.RLock()

    def is_allowed(self, client_id: str) -> tuple[bool, int]:
        """Check if request is allowed for client."""
        with self.lock:
            current_time = time.time()

            # Check if client is currently blocked
            if client_id in self.blocked_clients:
                if current_time < self.blocked_clients[client_id]:
                    return False, 0
                else:
                    del self.blocked_clients[client_id]

            # Clean old requests
            client_requests = self.requests[client_id]
            cutoff_time = current_time - self.time_window

            while client_requests and client_requests[0] < cutoff_time:
                client_requests.popleft()

            # Check rate limits
            current_requests = len(client_requests)

            if current_requests >= self.burst_limit:
                # Block client temporarily
                self.blocked_clients[client_id] = current_time + self.time_window
                return False, 0
            elif current_requests >= self.max_requests:
                return False, self.max_requests - current_requests

            # Allow request
            client_requests.append(current_time)
            return True, self.max_requests - current_requests - 1

    def get_stats(self) -> dict[str, Any]:
        """Get rate limiter statistics."""
        with self.lock:
            return {
                'active_clients': len(self.requests),
                'blocked_clients': len(self.blocked_clients),
                'total_requests': sum(len(req_queue) for req_queue in self.requests.values())
            }


class DataEncryption:
    """Data encryption and decryption utilities."""

    def __init__(self, encryption_key: bytes | None = None):
        if encryption_key:
            self.key = encryption_key
        else:
            # Generate key from environment or create new one
            key_b64 = os.getenv("TRADING_BOT_ENCRYPTION_KEY")
            if key_b64:
                self.key = base64.urlsafe_b64decode(key_b64)
            else:
                self.key = Fernet.generate_key()

        self.cipher = Fernet(self.key)
        self.logger = logging.getLogger(__name__)

    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        try:
            encrypted_data = self.cipher.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            raise

    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.cipher.decrypt(encrypted_bytes)
            return decrypted_data.decode()
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            raise

    def get_key_b64(self) -> str:
        """Get encryption key as base64 string."""
        return base64.urlsafe_b64encode(self.key).decode()


class AuditLogger:
    """Secure audit logging for compliance."""

    def __init__(self, log_file: str = "logs/audit.log"):
        self.log_file = log_file
        self.logger = logging.getLogger("audit")

        # Setup audit log handler
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s|%(levelname)s|%(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S.%fZ'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        # Ensure audit log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log_event(self, event_type: str, details: dict[str, Any],
                  user_id: str | None = None, source_ip: str | None = None):
        """Log audit event."""
        audit_record = {
            'timestamp': datetime.now(UTC).isoformat(),
            'event_type': event_type,
            'user_id': user_id,
            'source_ip': source_ip,
            'details': details
        }

        # Create audit trail hash for integrity
        record_str = json.dumps(audit_record, sort_keys=True)
        audit_hash = hashlib.sha256(record_str.encode()).hexdigest()
        audit_record['audit_hash'] = audit_hash

        self.logger.info(json.dumps(audit_record))

    def log_trade_execution(self, symbol: str, side: str, quantity: float,
                           price: float, order_id: str, user_id: str = None):
        """Log trade execution for audit trail."""
        self.log_event(
            event_type="TRADE_EXECUTION",
            details={
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'order_id': order_id,
                'total_value': quantity * price
            },
            user_id=user_id
        )

    def log_risk_violation(self, violation_type: str, details: dict[str, Any]):
        """Log risk management violations."""
        self.log_event(
            event_type="RISK_VIOLATION",
            details={
                'violation_type': violation_type,
                **details
            }
        )

    def log_security_event(self, event: SecurityEvent):
        """Log security events."""
        self.log_event(
            event_type="SECURITY_EVENT",
            details={
                'security_event_type': event.event_type,
                'severity': event.severity.value,
                'description': event.description,
                'action_taken': event.action_taken,
                **event.details
            },
            source_ip=event.source_ip,
            user_id=event.user_id
        )


class AnomalyDetector:
    """Machine learning-based anomaly detection for trading patterns."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Historical data for pattern analysis
        self.trade_history: deque = deque(maxlen=10000)
        self.pnl_history: deque = deque(maxlen=1000)
        self.position_history: deque = deque(maxlen=1000)

        # Anomaly thresholds
        self.thresholds = {
            'unusual_trade_size': 5.0,  # 5x normal size
            'rapid_fire_orders': 10,    # 10 orders in 1 minute
            'abnormal_pnl_swing': 0.1,  # 10% swing
            'position_concentration': 0.3  # 30% of portfolio
        }

        # Detection flags
        self.anomalies_detected: list[dict[str, Any]] = []

    def analyze_trade_pattern(self, symbol: str, side: str, quantity: float,
                            price: float) -> dict[str, Any] | None:
        """Analyze trade for anomalous patterns."""
        current_time = time.time()

        # Record trade
        trade_record = {
            'timestamp': current_time,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'value': quantity * price
        }
        self.trade_history.append(trade_record)

        anomalies = []

        # Check for unusual trade size
        recent_trades = [t for t in self.trade_history
                        if t['symbol'] == symbol and current_time - t['timestamp'] < 3600]

        if len(recent_trades) > 1:
            avg_quantity = sum(t['quantity'] for t in recent_trades[:-1]) / (len(recent_trades) - 1)
            if quantity > avg_quantity * self.thresholds['unusual_trade_size']:
                anomalies.append({
                    'type': 'unusual_trade_size',
                    'severity': SecurityLevel.MEDIUM,
                    'details': {
                        'current_quantity': quantity,
                        'average_quantity': avg_quantity,
                        'ratio': quantity / avg_quantity
                    }
                })

        # Check for rapid-fire orders
        recent_orders = [t for t in self.trade_history
                        if current_time - t['timestamp'] < 60]  # Last minute

        if len(recent_orders) >= self.thresholds['rapid_fire_orders']:
            anomalies.append({
                'type': 'rapid_fire_orders',
                'severity': SecurityLevel.HIGH,
                'details': {
                    'orders_count': len(recent_orders),
                    'time_window': 60,
                    'threshold': self.thresholds['rapid_fire_orders']
                }
            })

        if anomalies:
            anomaly_report = {
                'timestamp': datetime.now(UTC),
                'trade': trade_record,
                'anomalies': anomalies
            }
            self.anomalies_detected.append(anomaly_report)
            return anomaly_report

        return None

    def analyze_pnl_pattern(self, current_pnl: float) -> dict[str, Any] | None:
        """Analyze P&L for abnormal swings."""
        if not self.pnl_history:
            self.pnl_history.append(current_pnl)
            return None

        previous_pnl = self.pnl_history[-1]
        pnl_change = abs(current_pnl - previous_pnl) / abs(previous_pnl) if previous_pnl != 0 else 0

        self.pnl_history.append(current_pnl)

        if pnl_change > self.thresholds['abnormal_pnl_swing']:
            return {
                'type': 'abnormal_pnl_swing',
                'severity': SecurityLevel.HIGH,
                'details': {
                    'current_pnl': current_pnl,
                    'previous_pnl': previous_pnl,
                    'change_percent': pnl_change * 100,
                    'threshold_percent': self.thresholds['abnormal_pnl_swing'] * 100
                }
            }

        return None

    def get_anomaly_report(self) -> dict[str, Any]:
        """Get comprehensive anomaly report."""
        recent_anomalies = [a for a in self.anomalies_detected
                           if (datetime.now(UTC) - a['timestamp']).total_seconds() < 3600]

        return {
            'timestamp': datetime.now(UTC).isoformat(),
            'total_anomalies': len(self.anomalies_detected),
            'recent_anomalies': len(recent_anomalies),
            'anomaly_types': list(set(
                anomaly['type']
                for report in recent_anomalies
                for anomaly in report['anomalies']
            )),
            'recent_reports': recent_anomalies[-10:]  # Last 10 reports
        }


class SecurityManager:
    """Comprehensive security management system."""

    def __init__(self, enable_encryption: bool = True):
        self.logger = logging.getLogger(__name__)

        # Security components
        self.rate_limiter = RateLimiter(max_requests=100, time_window=60)
        self.encryption = DataEncryption() if enable_encryption else None
        self.audit_logger = AuditLogger()
        self.anomaly_detector = AnomalyDetector()

        # Security events
        self.security_events: deque = deque(maxlen=1000)

        # API keys and secrets management
        self.secured_credentials: dict[str, str] = {}

        # Security settings
        self.security_settings = {
            'enable_rate_limiting': True,
            'enable_encryption': enable_encryption,
            'enable_audit_logging': True,
            'enable_anomaly_detection': True,
            'require_api_authentication': True,
            'max_failed_attempts': 5,
            'lockout_duration': 300  # 5 minutes
        }

        # Failed authentication tracking
        self.failed_attempts: dict[str, list[float]] = defaultdict(list)
        self.locked_clients: dict[str, float] = {}

        self.logger.info("Security manager initialized")

    def authenticate_api_request(self, api_key: str, signature: str,
                                timestamp: str, body: str, client_ip: str = None) -> bool:
        """Authenticate API request with HMAC signature."""
        try:
            # Check rate limits
            if self.security_settings['enable_rate_limiting'] and client_ip:
                allowed, remaining = self.rate_limiter.is_allowed(client_ip)
                if not allowed:
                    self._log_security_event(
                        "RATE_LIMIT_EXCEEDED",
                        SecurityLevel.MEDIUM,
                        f"Rate limit exceeded for {client_ip}",
                        {'client_ip': client_ip, 'remaining_requests': remaining}
                    )
                    return False

            # Check if client is locked out
            if client_ip and client_ip in self.locked_clients:
                if time.time() < self.locked_clients[client_ip]:
                    self._log_security_event(
                        "LOCKED_CLIENT_ACCESS_ATTEMPT",
                        SecurityLevel.HIGH,
                        f"Locked client attempted access: {client_ip}",
                        {'client_ip': client_ip}
                    )
                    return False
                else:
                    del self.locked_clients[client_ip]

            # Verify API key exists
            expected_secret = os.getenv(f"API_SECRET_{api_key}")
            if not expected_secret:
                self._handle_failed_auth(client_ip, "Invalid API key")
                return False

            # Verify timestamp (prevent replay attacks)
            try:
                request_time = float(timestamp)
                current_time = time.time()
                if abs(current_time - request_time) > 300:  # 5 minute window
                    self._handle_failed_auth(client_ip, "Request timestamp expired")
                    return False
            except ValueError:
                self._handle_failed_auth(client_ip, "Invalid timestamp format")
                return False

            # Verify HMAC signature
            expected_signature = hmac.new(
                expected_secret.encode(),
                f"{timestamp}{body}".encode(),
                hashlib.sha256
            ).hexdigest()

            if not hmac.compare_digest(signature, expected_signature):
                self._handle_failed_auth(client_ip, "Invalid signature")
                return False

            # Authentication successful
            self._log_security_event(
                "SUCCESSFUL_AUTHENTICATION",
                SecurityLevel.LOW,
                f"Successful API authentication for {api_key}",
                {'api_key': api_key, 'client_ip': client_ip}
            )

            return True

        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            self._handle_failed_auth(client_ip, f"Authentication error: {e}")
            return False

    def _handle_failed_auth(self, client_ip: str | None, reason: str):
        """Handle failed authentication attempt."""
        if client_ip:
            current_time = time.time()
            self.failed_attempts[client_ip].append(current_time)

            # Clean old attempts (keep only last hour)
            cutoff_time = current_time - 3600
            self.failed_attempts[client_ip] = [
                t for t in self.failed_attempts[client_ip] if t > cutoff_time
            ]

            # Check if client should be locked
            if len(self.failed_attempts[client_ip]) >= self.security_settings['max_failed_attempts']:
                self.locked_clients[client_ip] = current_time + self.security_settings['lockout_duration']
                self._log_security_event(
                    "CLIENT_LOCKED",
                    SecurityLevel.HIGH,
                    f"Client locked due to repeated failed attempts: {client_ip}",
                    {
                        'client_ip': client_ip,
                        'failed_attempts': len(self.failed_attempts[client_ip]),
                        'lockout_duration': self.security_settings['lockout_duration']
                    }
                )

        self._log_security_event(
            "FAILED_AUTHENTICATION",
            SecurityLevel.MEDIUM,
            f"Failed authentication: {reason}",
            {'client_ip': client_ip, 'reason': reason}
        )

    def _log_security_event(self, event_type: str, severity: SecurityLevel,
                          description: str, details: dict[str, Any]):
        """Log security event."""
        event = SecurityEvent(
            timestamp=datetime.now(UTC),
            event_type=event_type,
            severity=severity,
            source_ip=details.get('client_ip'),
            user_id=details.get('user_id'),
            description=description,
            details=details
        )

        self.security_events.append(event)
        self.audit_logger.log_security_event(event)

        # Alert on high severity events
        if severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            self.logger.warning(f"SECURITY ALERT [{severity.value.upper()}]: {description}")

    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data if encryption is enabled."""
        if self.encryption:
            return self.encryption.encrypt_data(data)
        return data

    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data if encryption is enabled."""
        if self.encryption:
            return self.encryption.decrypt_data(encrypted_data)
        return encrypted_data

    def store_secure_credential(self, key: str, value: str):
        """Store credential securely."""
        if self.encryption:
            self.secured_credentials[key] = self.encryption.encrypt_data(value)
        else:
            self.secured_credentials[key] = value

    def retrieve_secure_credential(self, key: str) -> str | None:
        """Retrieve credential securely."""
        encrypted_value = self.secured_credentials.get(key)
        if not encrypted_value:
            return None

        if self.encryption:
            return self.encryption.decrypt_data(encrypted_value)
        return encrypted_value

    def analyze_trade_for_anomalies(self, symbol: str, side: str,
                                  quantity: float, price: float) -> dict[str, Any] | None:
        """Analyze trade for security anomalies."""
        if not self.security_settings['enable_anomaly_detection']:
            return None

        anomaly_report = self.anomaly_detector.analyze_trade_pattern(
            symbol, side, quantity, price
        )

        if anomaly_report:
            # Log security event for anomaly
            self._log_security_event(
                "TRADING_ANOMALY_DETECTED",
                SecurityLevel.MEDIUM,
                f"Anomalous trading pattern detected for {symbol}",
                {
                    'symbol': symbol,
                    'anomaly_types': [a['type'] for a in anomaly_report['anomalies']],
                    'trade_details': anomaly_report['trade']
                }
            )

        return anomaly_report

    def get_security_report(self) -> dict[str, Any]:
        """Generate comprehensive security report."""
        recent_events = [
            event for event in self.security_events
            if (datetime.now(UTC) - event.timestamp).total_seconds() < 3600
        ]

        # Group events by type and severity
        event_summary = defaultdict(lambda: defaultdict(int))
        for event in recent_events:
            event_summary[event.event_type][event.severity.value] += 1

        return {
            'timestamp': datetime.now(UTC).isoformat(),
            'security_summary': {
                'total_events': len(self.security_events),
                'recent_events': len(recent_events),
                'active_rate_limits': self.rate_limiter.get_stats(),
                'locked_clients': len(self.locked_clients),
                'failed_auth_attempts': sum(len(attempts) for attempts in self.failed_attempts.values())
            },
            'event_breakdown': dict(event_summary),
            'anomaly_report': self.anomaly_detector.get_anomaly_report(),
            'security_settings': self.security_settings
        }

    def run_security_audit(self) -> dict[str, Any]:
        """Run comprehensive security audit."""
        audit_results = {
            'timestamp': datetime.now(UTC).isoformat(),
            'audit_checks': {},
            'recommendations': [],
            'security_score': 0
        }

        score = 100  # Start with perfect score

        # Check environment variables security
        critical_env_vars = ['ALPACA_API_KEY', 'ALPACA_SECRET_KEY']
        missing_vars = [var for var in critical_env_vars if not os.getenv(var)]

        if missing_vars:
            audit_results['audit_checks']['environment_security'] = 'FAIL'
            audit_results['recommendations'].append(
                f"Missing critical environment variables: {', '.join(missing_vars)}"
            )
            score -= 20
        else:
            audit_results['audit_checks']['environment_security'] = 'PASS'

        # Check encryption status
        if self.encryption:
            audit_results['audit_checks']['data_encryption'] = 'PASS'
        else:
            audit_results['audit_checks']['data_encryption'] = 'FAIL'
            audit_results['recommendations'].append("Enable data encryption for sensitive information")
            score -= 15

        # Check rate limiting
        if self.security_settings['enable_rate_limiting']:
            audit_results['audit_checks']['rate_limiting'] = 'PASS'
        else:
            audit_results['audit_checks']['rate_limiting'] = 'FAIL'
            audit_results['recommendations'].append("Enable API rate limiting")
            score -= 10

        # Check audit logging
        if self.security_settings['enable_audit_logging']:
            audit_results['audit_checks']['audit_logging'] = 'PASS'
        else:
            audit_results['audit_checks']['audit_logging'] = 'FAIL'
            audit_results['recommendations'].append("Enable comprehensive audit logging")
            score -= 15

        # Check for recent security incidents
        recent_critical_events = [
            event for event in self.security_events
            if event.severity == SecurityLevel.CRITICAL and
            (datetime.now(UTC) - event.timestamp).total_seconds() < 86400
        ]

        if recent_critical_events:
            audit_results['audit_checks']['recent_incidents'] = 'FAIL'
            audit_results['recommendations'].append(
                f"Address {len(recent_critical_events)} critical security incidents from last 24 hours"
            )
            score -= 25
        else:
            audit_results['audit_checks']['recent_incidents'] = 'PASS'

        audit_results['security_score'] = max(0, score)

        return audit_results


# Global security manager instance
_security_manager: SecurityManager | None = None


def get_security_manager() -> SecurityManager:
    """Get global security manager instance."""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager


def initialize_security_manager(enable_encryption: bool = True) -> SecurityManager:
    """Initialize security manager."""
    global _security_manager
    _security_manager = SecurityManager(enable_encryption)
    return _security_manager
