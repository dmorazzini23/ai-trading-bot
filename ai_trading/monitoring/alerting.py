"""
Multi-channel alerting system for production trading.

Implements comprehensive alerting with email, SMS, and team notifications
for critical trading events, system failures, and performance issues.
"""

import queue
import smtplib
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any

# Use the centralized logger as per AGENTS.md
from ai_trading.logging import logger
from ai_trading.utils import http
from ai_trading.utils.timing import (
    HTTP_TIMEOUT,
)  # AI-AGENT-REF: avoid circular import


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertChannel(Enum):
    """Alert delivery channels."""

    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    DISCORD = "discord"
    WEBHOOK = "webhook"


class Alert:
    """Alert message container."""

    def __init__(
        self,
        title: str,
        message: str,
        severity: AlertSeverity,
        source: str = "",
        metadata: dict = None,
    ):
        """Initialize alert."""
        self.id = f"alert_{int(time.time() * 1000)}"
        self.title = title
        self.message = message
        self.severity = severity
        self.source = source
        self.metadata = metadata or {}
        self.timestamp = datetime.now(UTC)
        self.channels_sent = []
        self.delivery_attempts = 0
        self.max_attempts = 3


class EmailAlerter:
    """
    Email alerting system for trading notifications.

    Sends email alerts for critical trading events and system status.
    """

    def __init__(
        self,
        smtp_server: str = None,
        smtp_port: int = 587,
        username: str = None,
        password: str = None,
    ):
        """Initialize email alerter."""
        # AI-AGENT-REF: Email alerting system
        self.smtp_server = smtp_server or "smtp.gmail.com"
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.enabled = all([smtp_server, username, password])

        if not self.enabled:
            logger.warning(
                "Email alerter not fully configured - alerts will be logged only"
            )
        else:
            logger.info(
                f"Email alerter configured for {self.smtp_server}:{self.smtp_port}"
            )

    def send_alert(self, alert: Alert, recipients: list[str]) -> bool:
        """
        Send email alert to recipients.

        Args:
            alert: Alert object to send
            recipients: List of email addresses

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            if not self.enabled:
                logger.warning(f"Email not configured - logging alert: {alert.title}")
                return False

            if not recipients:
                logger.warning("No email recipients configured")
                return False

            # Create email message
            msg = MIMEMultipart()
            msg["From"] = self.username
            msg["To"] = ", ".join(recipients)
            msg["Subject"] = f"[{alert.severity.value.upper()}] {alert.title}"

            # Create email body
            body = self._create_email_body(alert)
            msg.attach(MIMEText(body, "html"))

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)

            logger.info(
                f"Email alert sent: {alert.title} to {len(recipients)} recipients"
            )
            return True

        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
            return False

    def _create_email_body(self, alert: Alert) -> str:
        """Create HTML email body from alert."""
        severity_colors = {
            AlertSeverity.INFO: "#17a2b8",
            AlertSeverity.WARNING: "#ffc107",
            AlertSeverity.CRITICAL: "#dc3545",
            AlertSeverity.EMERGENCY: "#721c24",
        }

        color = severity_colors.get(alert.severity, "#6c757d")

        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <div style="border-left: 4px solid {color}; padding: 10px; margin: 10px 0;">
                <h2 style="color: {color}; margin: 0 0 10px 0;">
                    [{alert.severity.value.upper()}] {alert.title}
                </h2>
                <p><strong>Source:</strong> {alert.source}</p>
                <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                <div style="background-color: #f8f9fa; padding: 10px; border-radius: 4px; margin: 10px 0;">
                    <pre style="margin: 0; white-space: pre-wrap;">{alert.message}</pre>
                </div>
        """

        if alert.metadata:
            html_body += """
                <h3>Additional Information:</h3>
                <table style="border-collapse: collapse; width: 100%;">
            """
            for key, value in alert.metadata.items():
                html_body += f"""
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">{key}</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">{value}</td>
                    </tr>
                """
            html_body += "</table>"

        html_body += """
            </div>
        </body>
        </html>
        """

        return html_body


class SlackAlerter:
    """
    Slack alerting system for team notifications.

    Sends Slack messages for trading alerts and system status.
    """

    def __init__(self, webhook_url: str = None, channel: str = None):
        """Initialize Slack alerter."""
        # AI-AGENT-REF: Slack alerting system
        self.webhook_url = webhook_url
        self.channel = channel or "#trading-alerts"
        self.enabled = bool(webhook_url)

        if not self.enabled:
            logger.warning("Slack alerter not configured - alerts will be logged only")
        else:
            logger.info(f"Slack alerter configured for channel {self.channel}")

    def send_alert(self, alert: Alert) -> bool:
        """
        Send Slack alert.

        Args:
            alert: Alert object to send

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            if not self.enabled:
                logger.warning(f"Slack not configured - logging alert: {alert.title}")
                return False

            # Create Slack message
            color_map = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ff9500",
                AlertSeverity.CRITICAL: "#ff0000",
                AlertSeverity.EMERGENCY: "#8b0000",
            }

            color = color_map.get(alert.severity, "#808080")

            payload = {
                "channel": self.channel,
                "username": "Trading Bot",
                "icon_emoji": ":robot_face:",
                "attachments": [
                    {
                        "color": color,
                        "title": f"[{alert.severity.value.upper()}] {alert.title}",
                        "text": alert.message,
                        "fields": [
                            {"title": "Source", "value": alert.source, "short": True},
                            {
                                "title": "Time",
                                "value": alert.timestamp.strftime(
                                    "%Y-%m-%d %H:%M:%S UTC"
                                ),
                                "short": True,
                            },
                        ],
                        "footer": "AI Trading Bot",
                        "ts": int(alert.timestamp.timestamp()),
                    }
                ],
            }

            # Add metadata fields
            if alert.metadata:
                for key, value in alert.metadata.items():
                    payload["attachments"][0]["fields"].append(
                        {"title": key, "value": str(value), "short": True}
                    )

            # Send to Slack
            response = http.post(
                self.webhook_url,
                json=payload,
                timeout=HTTP_TIMEOUT,  # AI-AGENT-REF: explicit timeout
            )
            response.raise_for_status()

            logger.info(f"Slack alert sent: {alert.title}")
            return True

        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
            return False


class AlertManager:
    """
    Comprehensive alert management system.

    Coordinates multiple alert channels, manages alert routing,
    and provides escalation and rate limiting capabilities.
    """

    def __init__(self):
        """Initialize alert manager."""
        # AI-AGENT-REF: Comprehensive alert management system
        self.email_alerter = EmailAlerter()
        self.slack_alerter = SlackAlerter()

        # Alert configuration
        self.alert_routing = {
            AlertSeverity.INFO: [AlertChannel.SLACK],
            AlertSeverity.WARNING: [AlertChannel.SLACK, AlertChannel.EMAIL],
            AlertSeverity.CRITICAL: [AlertChannel.SLACK, AlertChannel.EMAIL],
            AlertSeverity.EMERGENCY: [
                AlertChannel.SLACK,
                AlertChannel.EMAIL,
                AlertChannel.SMS,
            ],
        }

        # Rate limiting
        self.rate_limits = {
            AlertSeverity.INFO: timedelta(minutes=15),  # Max 1 per 15 minutes
            AlertSeverity.WARNING: timedelta(minutes=5),  # Max 1 per 5 minutes
            AlertSeverity.CRITICAL: timedelta(minutes=1),  # Max 1 per minute
            AlertSeverity.EMERGENCY: timedelta(seconds=0),  # No rate limiting
        }

        # Alert history for rate limiting
        self.alert_history = []
        self.max_history_size = 1000

        # Recipients configuration
        self.email_recipients = []
        self.escalation_recipients = []

        # Processing queue
        self.alert_queue = queue.Queue()
        self.processing_thread = None
        self.is_running = False

        # Callbacks for custom alert handling
        self.custom_handlers = {}

        logger.info("AlertManager initialized")

    def configure_email(
        self,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        recipients: list[str],
    ):
        """Configure email alerting."""
        try:
            self.email_alerter = EmailAlerter(
                smtp_server, smtp_port, username, password
            )
            self.email_recipients = recipients
            logger.info(f"Email alerting configured with {len(recipients)} recipients")
        except Exception as e:
            logger.error(f"Error configuring email alerts: {e}")

    def configure_slack(self, webhook_url: str, channel: str = None):
        """Configure Slack alerting."""
        try:
            self.slack_alerter = SlackAlerter(webhook_url, channel)
            logger.info("Slack alerting configured")
        except Exception as e:
            logger.error(f"Error configuring Slack alerts: {e}")

    def start_processing(self):
        """Start alert processing thread."""
        try:
            if self.is_running:
                logger.warning("Alert processing already running")
                return

            self.is_running = True
            self.processing_thread = threading.Thread(
                target=self._process_alerts, daemon=True, name="AlertProcessor"
            )
            self.processing_thread.start()

            logger.info("Alert processing started")

        except Exception as e:
            logger.error(f"Error starting alert processing: {e}")

    def stop_processing(self):
        """Stop alert processing thread."""
        try:
            self.is_running = False
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5.0)
            logger.info("Alert processing stopped")
        except Exception as e:
            logger.error(f"Error stopping alert processing: {e}")

    def send_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        source: str = "",
        metadata: dict = None,
        force: bool = False,
    ) -> str:
        """
        Send an alert through the system.

        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity level
            source: Source of the alert
            metadata: Additional metadata
            force: Force sending even if rate limited

        Returns:
            Alert ID
        """
        try:
            # Create alert
            alert = Alert(title, message, severity, source, metadata)

            # Check rate limiting
            if not force and self._is_rate_limited(alert):
                logger.debug(f"Alert rate limited: {alert.title}")
                return alert.id

            # Add to processing queue
            self.alert_queue.put(alert)

            # Update history
            self._update_alert_history(alert)

            logger.debug(f"Alert queued: {alert.id} - {alert.title}")
            return alert.id

        except Exception as e:
            logger.error(f"Error sending alert: {e}")
            return ""

    def send_trading_alert(
        self,
        event_type: str,
        symbol: str = "",
        details: dict = None,
        severity: AlertSeverity = AlertSeverity.INFO,
    ):
        """Send trading-specific alert."""
        try:
            title = f"Trading Event: {event_type}"
            if symbol:
                title += f" ({symbol})"

            message = f"Trading event occurred: {event_type}"
            if details:
                message += "\n\nDetails:\n"
                for key, value in details.items():
                    message += f"{key}: {value}\n"

            metadata = {"event_type": event_type, "symbol": symbol}
            if details:
                metadata.update(details)

            return self.send_alert(title, message, severity, "TradingEngine", metadata)

        except Exception as e:
            logger.error(f"Error sending trading alert: {e}")
            return ""

    def send_system_alert(
        self,
        component: str,
        status: str,
        details: str = "",
        severity: AlertSeverity = AlertSeverity.WARNING,
    ):
        """Send system status alert."""
        try:
            title = f"System Alert: {component} - {status}"
            message = f"System component status change:\n\nComponent: {component}\nStatus: {status}"

            if details:
                message += f"\n\nDetails:\n{details}"

            metadata = {"component": component, "status": status}

            return self.send_alert(title, message, severity, "SystemMonitor", metadata)

        except Exception as e:
            logger.error(f"Error sending system alert: {e}")
            return ""

    def send_performance_alert(
        self,
        metric: str,
        current_value: Any,
        threshold: Any,
        severity: AlertSeverity = AlertSeverity.WARNING,
    ):
        """Send performance-related alert."""
        try:
            title = f"Performance Alert: {metric} Threshold Exceeded"
            message = "Performance metric outside acceptable range:\n\n"
            message += f"Metric: {metric}\n"
            message += f"Current Value: {current_value}\n"
            message += f"Threshold: {threshold}"

            metadata = {
                "metric": metric,
                "current_value": current_value,
                "threshold": threshold,
            }

            return self.send_alert(
                title, message, severity, "PerformanceMonitor", metadata
            )

        except Exception as e:
            logger.error(f"Error sending performance alert: {e}")
            return ""

    def _process_alerts(self):
        """Main alert processing loop."""
        try:
            while self.is_running:
                try:
                    # Get alert from queue with timeout
                    alert = self.alert_queue.get(timeout=1.0)

                    # Route alert to appropriate channels
                    channels = self.alert_routing.get(
                        alert.severity, [AlertChannel.SLACK]
                    )

                    for channel in channels:
                        success = self._send_to_channel(alert, channel)
                        if success:
                            alert.channels_sent.append(channel)

                    # Handle custom handlers
                    if alert.severity in self.custom_handlers:
                        try:
                            self.custom_handlers[alert.severity](alert)
                        except Exception as e:
                            logger.error(f"Error in custom alert handler: {e}")

                    # Mark task as done
                    self.alert_queue.task_done()

                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error processing alert: {e}")

        except Exception as e:
            logger.error(f"Error in alert processing loop: {e}")

    def _send_to_channel(self, alert: Alert, channel: AlertChannel) -> bool:
        """Send alert to specific channel."""
        try:
            if channel == AlertChannel.EMAIL:
                return self.email_alerter.send_alert(alert, self.email_recipients)
            elif channel == AlertChannel.SLACK:
                return self.slack_alerter.send_alert(alert)
            elif channel == AlertChannel.SMS:
                # SMS implementation would go here
                logger.warning("SMS alerting not implemented")
                return False
            else:
                logger.warning(f"Unknown alert channel: {channel}")
                return False

        except Exception as e:
            logger.error(f"Error sending to channel {channel}: {e}")
            return False

    def _is_rate_limited(self, alert: Alert) -> bool:
        """Check if alert is rate limited."""
        try:
            rate_limit = self.rate_limits.get(alert.severity, timedelta(0))
            if rate_limit.total_seconds() == 0:
                return False

            # Check for recent similar alerts
            cutoff_time = datetime.now(UTC) - rate_limit
            recent_alerts = [
                a
                for a in self.alert_history
                if a.timestamp >= cutoff_time
                and a.severity == alert.severity
                and a.title == alert.title
            ]

            return len(recent_alerts) > 0

        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return False

    def _update_alert_history(self, alert: Alert):
        """Update alert history for rate limiting."""
        try:
            self.alert_history.append(alert)

            # Trim history to max size
            if len(self.alert_history) > self.max_history_size:
                self.alert_history = self.alert_history[-self.max_history_size :]

        except Exception as e:
            logger.error(f"Error updating alert history: {e}")

    def add_custom_handler(
        self, severity: AlertSeverity, handler: Callable[[Alert], None]
    ):
        """Add custom alert handler for specific severity."""
        self.custom_handlers[severity] = handler

    def get_alert_stats(self) -> dict[str, Any]:
        """Get alert statistics."""
        try:
            now = datetime.now(UTC)
            last_hour = now - timedelta(hours=1)
            last_day = now - timedelta(days=1)

            recent_alerts_hour = [
                a for a in self.alert_history if a.timestamp >= last_hour
            ]
            recent_alerts_day = [
                a for a in self.alert_history if a.timestamp >= last_day
            ]

            stats = {
                "total_alerts": len(self.alert_history),
                "alerts_last_hour": len(recent_alerts_hour),
                "alerts_last_day": len(recent_alerts_day),
                "queue_size": self.alert_queue.qsize(),
                "processing_active": self.is_running,
                "severity_counts": {},
            }

            # Count by severity
            for severity in AlertSeverity:
                count = len([a for a in recent_alerts_day if a.severity == severity])
                stats["severity_counts"][severity.value] = count

            return stats

        except Exception as e:
            logger.error(f"Error getting alert stats: {e}")
            return {"error": str(e)}
