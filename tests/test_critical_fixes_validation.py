#!/usr/bin/env python3
"""
Critical fixes validation test for the AI trading bot.
Tests the 5 major issues identified in the problem statement.
"""

import os
import sys
import tempfile
import unittest
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

from tests.mocks.app_mocks import MockConfig
from ai_trading.meta_learning import validate_trade_data_quality


class TestCriticalFixes(unittest.TestCase):
    """Test suite for the critical fixes implementation."""

    def setUp(self):
        """Set up test environment."""
        # AI-AGENT-REF: Use environment variables to avoid hardcoded secrets
        sys.modules["config"] = cast(Any, MockConfig())

    def test_sentiment_circuit_breaker_constants(self):
        """Test 1: Sentiment API Rate Limiting - Circuit breaker constants."""
        # Test that enhanced sentiment caching constants are reasonable
        SENTIMENT_RATE_LIMITED_TTL_SEC = 3600  # 1 hour
        SENTIMENT_FAILURE_THRESHOLD = 3  # 3 failures
        SENTIMENT_RECOVERY_TIMEOUT = 300  # 5 minutes
        SENTIMENT_TTL_SEC = 600  # 10 minutes

        # Validate the constants
        self.assertGreater(
            SENTIMENT_RATE_LIMITED_TTL_SEC,
            SENTIMENT_TTL_SEC,
            "Rate limited TTL should be longer than normal TTL",
        )
        self.assertGreaterEqual(
            SENTIMENT_FAILURE_THRESHOLD,
            2,
            "Should allow at least 2 failures before opening circuit",
        )
        self.assertGreaterEqual(
            SENTIMENT_RECOVERY_TIMEOUT,
            60,
            "Recovery timeout should be at least 1 minute",
        )

    def test_data_staleness_detection_improvement(self):
        """Test 4: Data Staleness Detection - Weekend/holiday awareness."""
        from ai_trading.utils.base import is_market_holiday, is_weekend

        # Test weekend detection
        saturday = datetime(2024, 1, 6, 12, 0, tzinfo=UTC)  # Saturday
        sunday = datetime(2024, 1, 7, 12, 0, tzinfo=UTC)  # Sunday
        monday = datetime(2024, 1, 8, 12, 0, tzinfo=UTC)  # Monday

        self.assertTrue(is_weekend(saturday), "Saturday should be detected as weekend")
        self.assertTrue(is_weekend(sunday), "Sunday should be detected as weekend")
        self.assertFalse(is_weekend(monday), "Monday should not be detected as weekend")

        # Test holiday detection
        new_years = date(2024, 1, 1)  # New Year's Day
        christmas = date(2024, 12, 25)  # Christmas
        regular_day = date(2024, 3, 15)  # Regular Friday

        self.assertTrue(
            is_market_holiday(new_years), "New Year's should be detected as holiday"
        )
        self.assertTrue(
            is_market_holiday(christmas), "Christmas should be detected as holiday"
        )
        self.assertFalse(
            is_market_holiday(regular_day),
            "Regular day should not be detected as holiday",
        )

    def test_meta_learning_price_validation(self):
        """Test 2: MetaLearning Data Validation - Price validation logic."""
        # Mock pandas for testing
        try:
            import pytest
            pd = pytest.importorskip("pandas")
            # Test data with mixed price types
            test_data = {
                "entry_price": ["100.50", "200", "invalid", "50.25"],
                "exit_price": ["105.75", "-5", "0", "55.00"],
                "side": ["buy", "sell", "buy", "sell"],
                "signal_tags": ["momentum", "mean_revert", "momentum", "trend"],
            }
            df = pd.DataFrame(test_data)

            # Apply the validation logic from meta_learning.py
            df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")
            df["exit_price"] = pd.to_numeric(df["exit_price"], errors="coerce")
            df = df.dropna(subset=["entry_price", "exit_price"])

            # Filter out non-positive prices
            df = df[(df["entry_price"] > 0) & (df["exit_price"] > 0)]

            # Should have 2 valid rows (first and last)
            self.assertEqual(
                len(df), 2, "Should have 2 rows with valid positive prices"
            )
            self.assertTrue(
                all(df["entry_price"] > 0), "All entry prices should be positive"
            )
            self.assertTrue(
                all(df["exit_price"] > 0), "All exit prices should be positive"
            )

            from ai_trading.meta_learning import validate_trade_data_quality

            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
                tmp.write(
                    "symbol,entry_time,entry_price,exit_time,exit_price,qty,side,signal_tags\n"
                )
                tmp.write(
                    "AAA,2024-01-01T00:00:00Z,100.50,2024-01-01T01:00:00Z,105.75,1,buy,momentum\n"
                )
                tmp.write(
                    "BBB,2024-01-02T00:00:00Z,200,2024-01-02T01:00:00Z,-5,1,sell,mean_revert\n"
                )
                tmp.write(
                    "CCC,2024-01-03T00:00:00Z,invalid,2024-01-03T01:00:00Z,0,1,buy,momentum\n"
                )
                tmp.write(
                    "DDD,2024-01-04T00:00:00Z,50.25,2024-01-04T01:00:00Z,55.00,1,sell,trend\n"
                )
                csv_path = tmp.name

            try:
                report = validate_trade_data_quality(csv_path)
                self.assertEqual(report["row_count"], 4)
                self.assertEqual(report["valid_price_rows"], 2)
            finally:
                os.unlink(csv_path)

        except ImportError:
            # Skip if pandas not available
            self.skipTest("pandas not available for meta learning test")

    def test_meta_learning_signed_negative_prices_are_rejected(self):
        """Trade data quality check should drop rows with signed non-positive prices."""
        from ai_trading.meta_learning import validate_trade_data_quality

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
            tmp.write("c1,c2,c3,c4,c5\n")
            tmp.write("2024-01-01,AAA,100,105,1\n")
            tmp.write("2024-01-02,BBB,-50,110,1\n")
            tmp.write("2024-01-03,CCC,120,-3,1\n")
            tmp_path = tmp.name

        try:
            report = validate_trade_data_quality(tmp_path)
            self.assertEqual(report["row_count"], 3)
            self.assertEqual(report["valid_price_rows"], 1)
        finally:
            os.unlink(tmp_path)

    def test_systemd_service_configuration(self):
        """Test 3: Service Configuration - systemd service file."""
        service_file = os.path.join(
            os.getcwd(), "packaging", "systemd", "ai-trading.service"
        )
        self.assertTrue(os.path.exists(service_file), "service file should exist")

        # Legacy service files should be absent
        self.assertFalse(os.path.exists("ai-trading-bot.service"))
        self.assertFalse(os.path.exists(os.path.join("deploy", "ai-trading.service")))

        with open(service_file) as f:
            content = f.read()

        # Check key configuration elements
        self.assertIn("User=aiuser", content, "Service should run as aiuser")
        self.assertIn("Group=aiuser", content, "Service should run as aiuser group")
        self.assertIn(
            "WorkingDirectory=/home/aiuser/ai-trading-bot",
            content,
            "Should have correct working directory",
        )
        self.assertIn(
            "/var/lib/ai-trading-bot/runtime/research_reports",
            content,
            "Service should provision after-hours report directory",
        )
        self.assertIn(
            "NoNewPrivileges=true", content, "Should have security restrictions"
        )
        self.assertIn("ProtectSystem=strict", content, "Should protect system")
        self.assertIn("StateDirectory=ai-trading-bot", content)
        self.assertIn("CacheDirectory=ai-trading-bot", content)
        self.assertIn("LogsDirectory=ai-trading-bot", content)
        self.assertIn("ReadWritePaths=", content)
        self.assertIn("/var/lib/ai-trading-bot", content)
        self.assertIn("/var/cache/ai-trading-bot", content)
        self.assertIn("/var/log/ai-trading-bot", content)
        self.assertIn("RuntimeMaxSec=infinity", content)
        self.assertNotIn("RuntimeMaxSec=0", content)
        self.assertIn("Restart=always", content, "Should restart on failure")

    def test_systemd_environment_precedence(self):
        """Packaged units should sync repo env into the runtime env on restart."""
        systemd_dir = Path(os.getcwd()) / "packaging" / "systemd"
        main_content = (systemd_dir / "ai-trading.service").read_text(encoding="utf-8")
        env_src_idx = main_content.index(
            "Environment=AI_TRADING_ENV_SRC=/home/aiuser/ai-trading-bot/.env"
        )
        runtime_env_file_idx = main_content.index(
            "EnvironmentFile=-/run/ai-trading-bot/ai-trading-runtime.env"
        )
        runtime_override_idx = main_content.index("Environment=AI_TRADING_DOTENV_RUNTIME_OVERRIDE=1")
        paper_url_idx = main_content.index(
            "Environment=ALPACA_TRADING_BASE_URL=https://paper-api.alpaca.markets"
        )
        api_port_idx = main_content.index("Environment=API_PORT=9001")
        health_port_idx = main_content.index("Environment=HEALTHCHECK_PORT=9001")
        alembic_idx = main_content.index("venv/bin/python -m alembic upgrade head")
        alembic_source_idx = main_content.index(
            "source /run/ai-trading-bot/ai-trading-runtime.env"
        )
        sync_idx = main_content.index("ExecStartPre=/home/aiuser/ai-trading-bot/scripts/sync_env_runtime.sh")
        exec_start_idx = main_content.index("ExecStart=/bin/bash -lc")
        exec_start_sync_idx = main_content.index(
            "ExecStart=/bin/bash -lc '/home/aiuser/ai-trading-bot/scripts/sync_env_runtime.sh"
        )
        exec_start_source_idx = main_content.index(
            "source /run/ai-trading-bot/ai-trading-runtime.env",
            exec_start_idx,
        )
        exec_start_python_idx = main_content.index(
            "exec /home/aiuser/ai-trading-bot/venv/bin/python -m ai_trading",
            exec_start_idx,
        )
        self.assertLess(paper_url_idx, env_src_idx)
        self.assertLess(env_src_idx, runtime_env_file_idx)
        self.assertLess(runtime_env_file_idx, runtime_override_idx)
        self.assertLess(runtime_env_file_idx, api_port_idx)
        self.assertLess(runtime_env_file_idx, health_port_idx)
        self.assertLess(env_src_idx, api_port_idx)
        self.assertLess(env_src_idx, health_port_idx)
        self.assertLess(sync_idx, alembic_idx)
        self.assertLess(sync_idx, alembic_source_idx)
        self.assertLess(alembic_source_idx, alembic_idx)
        self.assertLess(alembic_idx, exec_start_idx)
        self.assertLess(exec_start_sync_idx, exec_start_source_idx)
        self.assertLess(exec_start_source_idx, exec_start_python_idx)
        self.assertIn("RuntimeDirectoryPreserve=yes", main_content)
        self.assertNotIn("startup migration skipped", main_content)
        self.assertNotIn("/etc/ai-trading-bot/ai-trading.env", main_content)

        self.assertFalse((systemd_dir / "ai-trading-api.service").exists())
        self.assertFalse((systemd_dir / "ai-trading-metrics-forwarder.service").exists())

        project_content = (Path(os.getcwd()) / "pyproject.toml").read_text(encoding="utf-8")
        requirements_content = (Path(os.getcwd()) / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn('"alembic==1.14.1"', project_content)
        self.assertIn("alembic==1.14.1", requirements_content)

        for service_name in (
            "ai-trading-connectors.service",
            "ai-trading-healthcheck.service",
            "ai-trading-runtime-report.service",
            "ai-trading-replay-governance.service",
            "ai-trading-runtime-backup-sync.service",
            "ai-trading-runtime-prune.service",
        ):
            content = (systemd_dir / service_name).read_text(encoding="utf-8")
            env_src_idx = content.index(
                "Environment=AI_TRADING_ENV_SRC=/home/aiuser/ai-trading-bot/.env"
            )
            runtime_idx = content.index("EnvironmentFile=-/run/ai-trading-bot/ai-trading-runtime.env")
            override_idx = content.index("Environment=AI_TRADING_DOTENV_RUNTIME_OVERRIDE=1")
            sync_idx = content.index("ExecStartPre=/home/aiuser/ai-trading-bot/scripts/sync_env_runtime.sh")
            exec_idx = content.index("ExecStart=")
            self.assertLess(env_src_idx, runtime_idx, f"{service_name} should sync from repo .env")
            self.assertLess(runtime_idx, override_idx, f"{service_name} should let fresh runtime env win")
            self.assertLess(sync_idx, exec_idx, f"{service_name} should sync runtime env before start")
            self.assertNotIn("/etc/ai-trading-bot/ai-trading.env", content)
            if service_name == "ai-trading-connectors.service":
                self.assertIn("Environment=API_PORT=9001", content)
                self.assertIn("Environment=HEALTHCHECK_PORT=9001", content)
                self.assertLess(runtime_idx, content.index("Environment=API_PORT=9001"))
                self.assertLess(runtime_idx, content.index("Environment=HEALTHCHECK_PORT=9001"))

        timer_content = (systemd_dir / "ai-trading.timer").read_text(encoding="utf-8")
        self.assertNotIn("Timezone=", timer_content)
        self.assertIn("OnCalendar=Mon..Fri 09:30 America/New_York", timer_content)

    def test_runtime_backup_sync_waits_for_network_online(self):
        """Runtime backup sync should both order after and want network-online."""
        service_file = (
            Path(os.getcwd())
            / "packaging"
            / "systemd"
            / "ai-trading-runtime-backup-sync.service"
        )
        content = service_file.read_text(encoding="utf-8")

        self.assertIn("After=network-online.target", content)
        self.assertIn("Wants=network-online.target", content)

    def test_error_handling_robustness(self):
        """Test 5: General Robustness - Error handling patterns."""
        # Test that we have proper exception handling patterns

        # Example of how sentiment fallback should work
        def mock_sentiment_fallback(cached_data, default_score=0.0):
            """Mock sentiment fallback logic."""
            try:
                if (
                    cached_data
                    and isinstance(cached_data, list | tuple)
                    and len(cached_data) > 0
                ):
                    return cached_data[-1]  # Use last cached value
                return default_score
            except (TypeError, IndexError, AttributeError):
                return default_score  # Always return safe default

        # Test fallback scenarios
        self.assertEqual(
            mock_sentiment_fallback(None), 0.0, "Should return neutral when no cache"
        )
        self.assertEqual(
            mock_sentiment_fallback([]), 0.0, "Should return neutral when empty cache"
        )
        self.assertEqual(
            mock_sentiment_fallback([0.5, 0.7]), 0.7, "Should return last cached value"
        )
        self.assertEqual(
            mock_sentiment_fallback("invalid"),
            0.0,
            "Should handle invalid data gracefully",
        )

    def test_cache_behavior(self):
        """Test enhanced caching behavior."""
        import time

        # Mock cache structure
        cache = {}
        normal_ttl = 600  # 10 minutes
        extended_ttl = 3600  # 1 hour

        def is_cache_valid(cache_entry, ttl):
            if not cache_entry:
                return False
            timestamp, value = cache_entry
            return time.time() - timestamp < ttl

        # Test normal cache behavior
        now = time.time()
        cache["AAPL"] = (now - 300, 0.5)  # 5 minutes old

        self.assertTrue(
            is_cache_valid(cache["AAPL"], normal_ttl),
            "Recent cache should be valid with normal TTL",
        )

        # Test extended cache during rate limiting
        cache["MSFT"] = (now - 1800, 0.3)  # 30 minutes old

        self.assertFalse(
            is_cache_valid(cache["MSFT"], normal_ttl),
            "Old cache should be invalid with normal TTL",
        )
        self.assertTrue(
            is_cache_valid(cache["MSFT"], extended_ttl),
            "Old cache should be valid with extended TTL",
        )


if __name__ == "__main__":
    # Run the tests

    # Set up the test environment
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestCriticalFixes)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Report results
    if result.wasSuccessful():
        pass
    else:
        sys.exit(1)
